#!/usr/bin/env python
#
# This is a Telegram bot that moderates chats with the help of an LLM.
#
# Copyright (C) 2026 PurkkaKoodari
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import asyncio, json, re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, cast

import httpx

from telegram import Bot, Message, MessageEntity, ReplyParameters, User
from telegram.error import TelegramError
from telegram.helpers import escape_markdown

import openrouter
from openrouter.components import (
    ChatResult,
    ChatFormatJSONSchemaConfig,
    ChatJSONSchemaConfig,
    Reasoning,
)
from openrouter.errors import (
    BadRequestResponseErrorData,
    UnauthorizedResponseError,
    PaymentRequiredResponseError,
    OpenRouterError,
)

from janitorbot.config import Rule, cfg
from janitorbot.db import (
    CACHED_REASONING,
    MessageRecord,
    cached_verdict,
    chat_mode,
    chat_purpose,
    content_hash,
    effective_admins,
    effective_moderated_chats,
    is_established_user,
    is_new_user,
    mentioning_message_ids,
    pop_mirrored_message_ids,
    recent_spam_count,
    record_message,
    system_prompt_hash,
    user_message_ids,
)
from janitorbot.log import logger

or_client = openrouter.OpenRouter(api_key=cfg.llm.openrouter_key)

# Errors that are generally fatal and the bot operator's fault.
OPERATOR_ERRORS = (
    UnauthorizedResponseError,
    PaymentRequiredResponseError,
)


def expand_links(text: str, entities: Sequence[MessageEntity]) -> str:
    """
    Expand hidden-URL link entities (``text_link``) in ``text`` to Markdown ``[label](url)``.

    Telegram lets spammers hide a URL behind innocuous display text, so the raw message text
    never contains the actual link. Expanding these entities surfaces the URL to both the spam
    regexes and the LLM.
    """
    links = sorted(
        (e for e in entities if e.type == MessageEntity.TEXT_LINK and e.url),
        key=lambda e: e.offset,
    )
    if not links:
        return text
    # Entity offsets/lengths are UTF-16 code units (per the Bot API), so slice the UTF-16 encoding
    # rather than the Python string to stay correct for emoji and other non-BMP characters.
    utf16 = text.encode("utf-16-le")
    parts: list[str] = []
    last = 0
    for entity in links:
        start, end = entity.offset * 2, (entity.offset + entity.length) * 2
        # Leave non-link text verbatim so the existing plain-text regexes keep matching; only
        # escape the structural Markdown characters needed to keep the link well-formed.
        parts.append(utf16[last:start].decode("utf-16-le"))
        label = re.sub(r"([\\\[\]])", r"\\\1", utf16[start:end].decode("utf-16-le"))
        url = escape_markdown(cast(str, entity.url), version=2, entity_type=MessageEntity.TEXT_LINK)
        parts.append(f"[{label}]({url})")
        last = end
    parts.append(utf16[last:].decode("utf-16-le"))
    return "".join(parts)


def parse_message(msg: Message) -> tuple[str, bool, str]:
    """Return (text, has_media, media_info) for the message."""
    # Describe any attachments to the message.
    if msg.contact is not None:  # Contact → "contact: <name>"
        name = f"{msg.contact.first_name or ""} {msg.contact.last_name or ""}".strip()
        media_info = (
            f"The message contains a contact with the name:\n\n{name}"
            if name
            else "The message contains a contact."
        )
    elif msg.photo:  # photo → list[PhotoSize]
        media_info = "The message contains a photo."
    elif msg.video_note:  # VideoNote → "videonote"
        media_info = "The message contains a video note."
    elif (attachment := msg.effective_attachment) is not None:
        media_info = f"The message contains a {type(attachment).__name__.lower()}."
    else:
        media_info = ""

    has_media = bool(media_info)

    # Expand link entities in the message to Markdown.
    if msg.text is not None:
        text = expand_links(msg.text, msg.entities)
    elif msg.caption is not None:
        text = expand_links(msg.caption, msg.caption_entities)
    else:
        text = ""

    # Spammers can hide a URL in the link preview instead of the text, so append it if missing.
    preview = msg.link_preview_options
    if preview and preview.url and str(preview.url) not in text:
        text += f"\n{preview.url}"

    return text, has_media, media_info


def extract_mentions(msg: Message) -> list[str]:
    """
    Return the users and URLs mentioned in the message. Users are lowercase usernames or numeric
    user IDs, URLs include hidden link targets and the link preview URL.
    """
    types: list[str | None] = [
        MessageEntity.MENTION,
        MessageEntity.TEXT_MENTION,
        MessageEntity.URL,
        MessageEntity.TEXT_LINK,
    ]
    entities = msg.parse_entities(types) | msg.parse_caption_entities(types)
    mentions: list[str] = []
    for entity, text in entities.items():
        if entity.type == MessageEntity.TEXT_MENTION and entity.user:
            mentions.append(str(entity.user.id))
        elif entity.type == MessageEntity.MENTION:
            mentions.append(text.removeprefix("@").lower())
        elif entity.type == MessageEntity.TEXT_LINK and entity.url:
            mentions.append(entity.url)
        elif entity.type == MessageEntity.URL:
            mentions.append(text)
    preview = msg.link_preview_options
    if preview and preview.url:
        mentions.append(str(preview.url))
    return sorted(set(mentions))


def record_checked_message(
    msg: Message,
    spam: bool,
    *,
    sp_hash: bytes | None = None,
    c_hash: bytes | None = None,
    samples: Sequence[tuple[float, str | None]] | None = None,
    reasoning: str | None = None,
) -> None:
    """
    Store the result of spam checking the message in the database, with either the LLM samples or
    the non-LLM reasoning.
    """
    record_message(
        MessageRecord(
            chat_id=msg.chat.id,
            message_id=msg.message_id,
            user_id=cast(User, msg.from_user).id,
            timestamp=int(msg.date.timestamp()),
            system=False,
            spam=spam,
            mentions=extract_mentions(msg),
            system_prompt_hash=sp_hash,
            content_hash=c_hash,
            llm_samples=samples,
            reasoning=reasoning,
        )
    )


def record_joins(msg: Message) -> None:
    """Store a join message in the database as a system message from each joined user."""
    for member in msg.new_chat_members:
        record_message(
            MessageRecord(
                chat_id=msg.chat.id,
                message_id=msg.message_id,
                user_id=member.id,
                timestamp=int(msg.date.timestamp()),
                system=True,
                spam=False,
                mentions=[],
            )
        )


TME_LINK_REGEX = re.compile(r"https?://t\.me/[^\s)\]]+")
CJK_REGEX = re.compile(r"[\u4E00-\u9FFF]")


class _OGDescriptionParser(HTMLParser):
    """Minimal HTML parser that extracts the content of the first og:description meta tag."""

    def __init__(self) -> None:
        super().__init__()
        self.description: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta" or self.description is not None:
            return
        attrs_dict = dict(attrs)
        if attrs_dict.get("property") == "og:description":
            self.description = attrs_dict.get("content") or ""


async def fetch_tme_description(url: str) -> str:
    """Fetch the og:description meta tag of a t.me link, or "" if unavailable."""
    logger.info("Fetching description for link: %s", url)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("Failed to fetch og:description from %s: %s", url, e)
        return ""
    parser = _OGDescriptionParser()
    parser.feed(response.text)
    return (parser.description or "").strip()


async def prepare_llm_check(text: str, media_info: str, *, force: bool = False) -> str | None:
    """Build the user message content to send to the LLM, embedding attachment info and any
    linked Telegram post description as an <attachments> tag, as described in the system
    prompt's <message_format> section.

    Return None if the message has too little data to be worth checking via LLM, unless forced."""
    # A media with a description always gets forwarded to the LLM.
    # This is checked via formatting, as any media_info usually exceeds cfg.llm.min_length.
    force_llm = force or "\n\n" in media_info or ":" in media_info

    tme_link = TME_LINK_REGEX.search(text)
    if tme_link:
        description = await fetch_tme_description(tme_link.group())
        if description:
            media_info = f"{media_info}\n\nThe message contains a Telegram link with the description:\n\n{description}".strip()
            # A Telegram link with a description always gets forwarded to the LLM.
            force_llm = True

    # CJK spam is often very short, so always check messages containing CJK characters.
    if CJK_REGEX.search(text):
        force_llm = True

    if media_info:
        media_info = f"<attachments>{media_info}</attachments>"

    # Short-circuit the LLM check for short messages, as they are unlikely to be spam
    # and it's not worth the cost and false positives to check them.
    if len(text) < cfg.llm.min_length and not force_llm:
        return None

    return f"{text}\n\n{media_info}".strip()


def matches(rule: Rule, has_media: bool, text: str) -> bool:
    if rule.media is not None and has_media != rule.media:
        return False
    return bool(re.search(rule.regex, text, re.I))


def extract_provider(e: Exception) -> str | None:
    """
    Extract the name of the upstream provider that caused an OpenRouter error, if available.

    OpenRouter reports the provider responsible for many upstream failures (e.g. 502/503/429s
    from the provider itself) in the ``provider_name`` key of the error's ``metadata``.
    """
    if not isinstance(e, OpenRouterError):
        return None
    data = cast(BadRequestResponseErrorData | None, getattr(e, "data", None))
    if not data:
        return None
    if data.error.metadata:
        return data.error.metadata.get("provider_name")
    return None


async def resolve_provider(e: Exception, response: ChatResult | None) -> str | None:
    """
    Determine the upstream provider responsible for a failed LLM check.

    HTTP-level failures carry the provider on the exception itself (see ``extract_provider``).
    But a provider can also return a 200 response that fails our JSON schema (e.g. a null or
    malformed completion) -- OpenRouter doesn't treat that as an error, so no exception carries
    the provider. In that case, look it up via the generation ID instead, so misbehaving
    providers can still be identified and excluded via ``cfg.llm.model`` routing preferences.
    """
    provider = extract_provider(e)
    if provider or response is None:
        return provider
    try:
        generation = await or_client.generations.get_generation_async(id=response.id)
        return generation.data.provider_name
    except Exception:
        logger.warning("Failed to look up provider for generation %s", response.id, exc_info=True)
        return None


async def llm_api(user_message: str, purpose: str) -> ChatResult:
    """
    Call the LLM API with the given user message and purpose, and return the response.
    """
    return await or_client.chat.send_async(
        model=cfg.llm.model,
        messages=[
            {
                "role": "system",
                "content": cfg.llm.system_prompt.format(purpose=purpose),
            },
            {"role": "user", "content": user_message},
        ],
        max_tokens=3000,
        temperature=cfg.llm.temperature,
        reasoning=Reasoning(
            effort=cfg.llm.reasoning_effort,
            summary="auto",
        ),
        response_format=ChatFormatJSONSchemaConfig(
            type="json_schema",
            json_schema=ChatJSONSchemaConfig(
                name="spam_prob",
                strict=True,
                schema_={
                    "type": "object",
                    "properties": {
                        "prob": {
                            "type": "number",
                            "description": "Probability that the message is spam",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["prob"],
                    "additionalProperties": False,
                },
            ),
        ),
    )


def parse_llm_response(response: ChatResult) -> tuple[float, str, int | None]:
    """
    Parse the LLM response and return (probability, reasoning, usage).

    This should pass for any valid responses.
    """
    content = cast(str, response.choices[0].message.content)
    prob = json.loads(content)["prob"]
    if not isinstance(prob, (float, int)):
        raise ValueError(f"Invalid prob value: {type(prob).__name__}: {prob}")

    reasoning = response.choices[0].message.reasoning or ""
    usage = response.usage.total_tokens if response.usage else None

    return float(prob), reasoning, usage


@dataclass(eq=False)
class LLMError(Exception):
    """An error occurred while checking a message with the LLM."""

    causes: list[Exception]
    fatal: bool
    is_provider_issue: bool
    provider: str | None
    usage: int | None
    finish_reason: str | None
    content: str | None
    reasoning: str | None

    def __str__(self) -> str:
        provider_info = f" (provider: {self.provider})" if self.provider else ""
        usage_info = f" after {self.usage} tokens" if self.usage is not None else ""
        finish_info = f" (finish reason: {self.finish_reason})" if self.finish_reason else ""
        result = f"LLM check{provider_info} failed{usage_info}{finish_info}"
        for exc in self.causes:
            result += f"\n{type(exc).__name__}: {exc}"
        if self.content is not None:
            result += f"\nContent: {self.content}"
        if self.reasoning is not None:
            result += f"\nReasoning: {self.reasoning}"
        return result


class DeferredLLMError(Exception):
    """An LLM error that will be enriched asynchronously and reported once ready."""

    def __init__(self, message: str, enriched: asyncio.Task[LLMError]) -> None:
        super().__init__(message)
        self.enriched = enriched


async def handle_llm_error(
    fetch: asyncio.Task[ChatResult],
    exc: Exception,
    fatal: bool,
    report: Callable[[LLMError], Awaitable[None]] | None,
) -> LLMError:
    """
    Handle an LLM error, reporting it to the super admin if configured.
    """
    # The successful response.
    response: ChatResult | None = None
    # All errors to be reported on this request.
    errors: list[Exception] = []
    # The exception that might be an OpenRouterError with provider info.
    err_response = exc
    if isinstance(exc, TimeoutError):
        # Timeout. Ditch the original stack trace, those are ugly with asyncio.
        exc = TimeoutError(
            f"LLM check timed out after {cfg.llm.timeout} seconds "
            f"- slow provider or too low max tokens?"
        )
        try:
            # Wait a while longer to get a response.
            response = await asyncio.wait_for(fetch, timeout=20 * cfg.llm.timeout)
            parse_llm_response(response)
        except httpx.HTTPError as later_exc:
            # The fetch failed on the HTTP layer. Ignore the timeout which is likely a side effect.
            errors = [later_exc]
        except TimeoutError:
            # Timed out even with the longer wait. Report that as a provider issue.
            errors = [
                TimeoutError(
                    f"LLM check timed out, no response within {20 * cfg.llm.timeout} seconds "
                    f"- slow provider or too low max tokens?"
                )
            ]
        except Exception as later_exc:
            # The fetch or parse failed for some reason other than a timeout.
            # Report both the timeout and that error.
            errors = [exc, later_exc]
            err_response = later_exc
        else:
            # The result was valid but came in way too late.
            # Only report the timeout and hope that a retry or another multisample handled it.
            errors = [exc]
    else:
        # The fetch or parsing didn't time out and returned another error in exc.
        errors = [exc]
        # Try getting the response for extra info if we have one.
        try:
            response = fetch.result()
        except Exception:
            # The fetch failed, exception is already in exc.
            pass

    # If fetch failed on the HTTP layer, that may be a network or OpenRouter issue.
    # Log these and obvious operator problems, but don't report as a provider issue.
    is_provider_issue = not all(
        isinstance(e, httpx.HTTPError) or isinstance(e, OPERATOR_ERRORS) for e in errors
    )

    # Attempt to extract as much useful information as possible.
    usage = None
    finish_reason = None
    content = None
    reasoning = None
    provider = await resolve_provider(err_response, response)
    if response is not None:
        try:
            if response.usage:
                usage = response.usage.total_tokens
            if response.choices:
                first = response.choices[0]
                reasoning = first.message.reasoning or None
                finish_reason = first.finish_reason
                if isinstance(first.message.content, str):
                    content = first.message.content
                elif first.message.content is not None:
                    content = first.message.model_dump_json(
                        exclude={"reasoning", "reasoning_details"} if reasoning else None
                    )
        except Exception:
            logger.warning("Failed to extract info from LLM response", exc_info=True)

    error = LLMError(
        causes=errors,
        fatal=fatal,
        is_provider_issue=is_provider_issue,
        provider=provider,
        usage=usage,
        finish_reason=finish_reason,
        content=content,
        reasoning=reasoning,
    )
    error.__cause__ = exc
    logger.error("LLM error occurred", exc_info=error)
    if callable(report):
        try:
            await report(error)
        except Exception:
            logger.error("Error reporting LLM error", exc_info=True)
    return error


_error_handling_tasks = set[asyncio.Task[Any]]()


def handle_llm_error_async(
    fetch: asyncio.Task[ChatResult],
    exc: Exception,
    fatal: bool,
    report: Callable[[LLMError], Awaitable[None]] | None,
) -> asyncio.Task[LLMError]:
    """
    Handle an LLM error asynchronously, reporting it to the super admin if configured.

    This is a fire-and-forget wrapper around ``handle_llm_error`` that runs in the background.
    """
    handle = asyncio.create_task(handle_llm_error(fetch, exc, fatal=fatal, report=report))
    _error_handling_tasks.add(handle)
    handle.add_done_callback(_error_handling_tasks.remove)
    return handle


async def llm_check(
    user_message: str,
    purpose: str,
    *,
    retry: int = cfg.llm.retry_count,
    recursive: bool = False,
    report_error: Callable[[LLMError], Awaitable[None]] | None = None,
) -> tuple[float, str, list[tuple[float, str]]]:
    """
    Check the message with the LLM and react to spam.

    If the result reaches the multisample threshold and this is not itself a recursive
    (multisample) call, the check is repeated in parallel and the median result is used.

    Returns a tuple of (spam_probability, reasoning, samples).
    """
    # Wrap the call in a task so we can await it multiple times.
    task = asyncio.create_task(llm_api(user_message, purpose))
    try:
        response = await asyncio.wait_for(asyncio.shield(task), timeout=cfg.llm.timeout)
        prob, reasoning, usage = parse_llm_response(response)
        logger.info("LLM result: %s tokens, spam probability: %.2f", usage or "?", prob)
    except Exception as exc:
        # If the error is obviously the operator's problem, don't bother retrying.
        if isinstance(exc, OPERATOR_ERRORS):
            retry = 0

        # Process the error further asynchronously.
        handle = handle_llm_error_async(
            task, exc, fatal=not retry and not recursive, report=report_error
        )

        if retry:
            await asyncio.sleep(cfg.llm.retry_delay)
            return await llm_check(
                user_message,
                purpose,
                retry=retry - 1,
                recursive=recursive,
                report_error=report_error,
            )
        else:
            raise DeferredLLMError(f"{type(exc).__name__}: {str(exc)}", handle) from exc

    # Complain if the provider returned no reasoning.
    if cfg.llm.report_no_reasoning and not reasoning.strip():
        exc = ValueError("No reasoning returned by OpenRouter")
        handle_llm_error_async(task, exc, fatal=False, report=report_error)

    samples = [(prob, reasoning)]

    # Multisample if the probability is above the threshold and this is the initial call.
    if not recursive and cfg.llm.multisample_count >= 2 and prob >= cfg.llm.multisample_threshold:
        logger.info(
            "Spam probability %.2f reached multisample threshold, sampling %d more time(s)...",
            prob,
            cfg.llm.multisample_count - 1,
        )
        further = await asyncio.gather(
            *(
                llm_check(user_message, purpose, recursive=True, report_error=report_error)
                for _ in range(cfg.llm.multisample_count - 1)
            ),
            return_exceptions=True,
        )
        samples += [result[:2] for result in further if isinstance(result, tuple)]
        samples.sort(key=lambda sample: sample[0])
        prob, reasoning = samples[(len(samples) - 1) // 2]
        logger.info(
            "Median spam probability: %.2f (%d sample(s))",
            prob,
            len(samples),
        )

    return (prob, reasoning, samples)


def sampled_prob_range(samples: list[float]) -> str:
    """
    Formats the range of spam probabilities in the samples if there are multiple samples.
    """
    if len(samples) < 2:
        return ""
    return f" ({min(samples):.2f}\u2026{max(samples):.2f})"


async def check_spam(bot: Bot, msg: Message) -> bool:
    """
    Check the message for spam and react to spam.

    Returns True if the message is likely not spam, False if it is likely spam.
    """
    chat_id = msg.chat.id
    sender = cast(User, msg.from_user)
    text, _, media_info = parse_message(msg)

    # Handle a couple hardcoded cases of spam before even running the LLM, to save costs and reduce false negatives.
    if chat_id in effective_moderated_chats() and text and re.search(cfg.spam_pattern, text, re.I):
        logger.info("Message matches spam regex, banning without LLM check!")
        matching_regex = next(
            (regex for regex in cfg.spam.regex if re.search(regex, text, re.I)), None
        )
        await attempt_delete_ban(bot, msg, reasoning=f"Matched spam regex: {matching_regex}")
        return False

    # For other messages, run the LLM check and then process them in the after function.
    # Allow admins to override the purpose.
    if sender.id in effective_admins() and text.startswith("purpose: ") and "\n" in text:
        purpose, _, text = text[len("purpose: ") :].partition("\n")
        logger.info("Using custom purpose from admin: %s", purpose)
    else:
        purpose = chat_purpose(msg.chat.id)

    system_prompt = cfg.llm.system_prompt.format(purpose=purpose)
    sp_hash = system_prompt_hash(system_prompt)

    llm_user_message = await prepare_llm_check(text, media_info)

    # If the message has too little data to check, skip the LLM and assume it's not spam.
    if llm_user_message is None:
        logger.info("Short message, skipping LLM check!")
        record_checked_message(msg, False, reasoning="Short message, skipped LLM check")
        return True

    if sender.id in effective_admins() and text.startswith("debug:\n"):
        logger.info("Debug mode, sending system prompt to admin.")
        await bot.send_message(
            chat_id=sender.id,
            reply_parameters=ReplyParameters(chat_id=msg.chat.id, message_id=msg.message_id),
            text=f"System prompt:\n{system_prompt}\n\nMessage:\n{llm_user_message}"[:4095],
        )
        return True

    # Reuse the verdict of an identical earlier message checked with the same system prompt.
    c_hash = content_hash(llm_user_message)
    cached_probs = cached_verdict(sp_hash, c_hash)

    # Now run the LLM check and react to the result.
    try:
        samples: Sequence[tuple[float, str]]
        if cached_probs:
            # Only record the reused median, not the reasoning or the individual samples.
            prob = cached_probs[(len(cached_probs) - 1) // 2]
            samples = [(prob, CACHED_REASONING)]
            # Still show the range of the cached samples in the monitor chat.
            sample_probs = cached_probs
            reasoning = "Cached LLM verdict from previous message"
            logger.info("Using cached verdict, spam probability: %.2f", prob)
        else:

            async def report_error(error: LLMError) -> None:
                # Record the error in the database.
                if error.fatal:
                    record_checked_message(msg, False, c_hash=c_hash, reasoning=str(error))
                # Report to the super admin if configured.
                if cfg.quiet:
                    return
                if error.fatal or (cfg.llm.report_provider_issues and error.is_provider_issue):
                    await bot.send_message(
                        chat_id=cfg.admins[0],
                        reply_parameters=ReplyParameters(
                            chat_id=msg.chat.id, message_id=msg.message_id
                        ),
                        text=str(error)[:4095],
                    )

            logger.info("LLM checking message...")
            prob, reasoning, samples = await llm_check(
                llm_user_message, purpose, report_error=report_error
            )
            sample_probs = [sample[0] for sample in samples]

        # Store the verdict.
        spam = prob >= cfg.llm.threshold
        record_checked_message(msg, spam, sp_hash=sp_hash, c_hash=c_hash, samples=samples)

        if spam:
            # Likely spam, delete and ban, and send the reasoning to the monitor chat for review.
            await attempt_delete_ban(
                bot,
                msg,
                reasoning=reasoning,
                prob=prob,
                sample_probs=sample_probs,
            )
        elif cfg.debug:
            # Not spam, but still send the reasoning to the admins for debugging.
            sampled = sampled_prob_range(sample_probs)
            await bot.send_message(
                chat_id=cfg.admins[0],
                reply_parameters=ReplyParameters(chat_id=msg.chat.id, message_id=msg.message_id),
                text=f"Spam probability: {prob:.2f}{sampled}\nReasoning: {reasoning}"[:4095],
            )

        if (
            not cfg.quiet
            and cfg.llm.outlier_threshold < 1.0
            and not cached_probs
            and len(samples) >= 2
        ):
            # Attempt to detect outliers in the sampling in case there's room for improvement in the system prompt.
            for sample_prob, sample_reasoning in (samples[0], samples[-1]):
                if abs(sample_prob - prob) >= cfg.llm.outlier_threshold:
                    await bot.send_message(
                        chat_id=cfg.admins[0],
                        reply_parameters=ReplyParameters(
                            chat_id=msg.chat.id, message_id=msg.message_id
                        ),
                        text=f"Outlier in sampling: {sample_prob:.2f} vs median {prob:.2f}\n"
                        f"Reasoning: {sample_reasoning}"[:4095],
                    )

        return not spam
    except DeferredLLMError:
        # Fail open to avoid false positives. Will be reported asynchronously by report_error.
        return True


async def bulk_delete(bot: Bot, chat_id: int, message_ids: list[int], what: str) -> None:
    """Delete the given messages in the chat, ignoring messages that no longer exist."""
    if not message_ids:
        return
    logger.info("Deleting %d %s", len(message_ids), what)
    for start in range(0, len(message_ids), 100):
        try:
            await bot.delete_messages(chat_id=chat_id, message_ids=message_ids[start : start + 100])
        except TelegramError:
            logger.warning("Failed to delete %s", what, exc_info=True)


async def delete_mirrors(bot: Bot, chat_id: int, message_ids: list[int]) -> None:
    """Delete the channel mirrors of the given messages, if the chat is a source chat."""
    if chat_id not in cfg.chats.source or not message_ids:
        return
    mirror_ids = pop_mirrored_message_ids(message_ids)
    await bulk_delete(bot, cfg.chats.channel, mirror_ids, "mirrored message(s) in channel")


async def delete_messages(bot: Bot, chat_id: int, message_ids: list[int], what: str) -> None:
    """Delete the given messages in the chat and their channel mirrors."""
    await bulk_delete(bot, chat_id, message_ids, what)
    await delete_mirrors(bot, chat_id, message_ids)


async def ban_user(bot: Bot, msg: Message, user: User) -> None:
    """
    Ban the user from the message's chat and delete their join messages there, or all their
    messages there if they are a new user.
    """
    await msg.chat.ban_member(user_id=user.id)
    if is_new_user(msg.chat.id, user.id):
        await delete_messages(
            bot,
            msg.chat.id,
            user_message_ids(msg.chat.id, user.id),
            "message(s) of banned new user",
        )
    else:
        await delete_messages(
            bot,
            msg.chat.id,
            user_message_ids(msg.chat.id, user.id, joins_only=True),
            "join message(s) of banned user",
        )


async def attempt_delete_ban(
    bot: Bot,
    msg: Message,
    reasoning: str,
    prob: float | None = None,
    sample_probs: list[float] = [],
):
    """
    Attempt to delete the message and ban the sender, and forward the message to the monitor chat with the reasoning.

    Without sample_probs, the message is recorded as spam with the non-LLM reasoning; LLM checks
    record their samples themselves.
    """
    if not sample_probs:
        record_checked_message(msg, True, reasoning=reasoning)

    sender = cast(User, msg.from_user)
    caller = msg.guest_bot_caller_user
    monitor = sender.id if sender.id in effective_admins() else cfg.chats.monitor

    fwd_msg = None
    if monitor:
        try:
            fwd_msg = await msg.forward(chat_id=monitor)
        except TelegramError:
            logger.warning("Failed to forward message to monitor chat", exc_info=True)

    no_delete = ""
    no_ban = ""
    simulation = (
        sender.id in effective_admins()
        or (caller and caller.id in effective_admins())
        or msg.chat.id not in effective_moderated_chats()
    )
    if simulation:
        no_ban = "\n(simulation, would have been deleted/banned)"
    else:
        mode = chat_mode(msg.chat.id)
        if mode == "test":
            no_ban = "\n(test mode, no action taken)"
        else:
            try:
                await msg.delete()
            except TelegramError as ex:
                no_delete = f"\nFailed to delete: {ex}"
            # The message may have been mirrored before being edited into spam.
            await delete_mirrors(bot, msg.chat.id, [msg.message_id])
            if caller and sender.username:
                # Also make sure to delete the calling messages.
                await delete_messages(
                    bot,
                    msg.chat.id,
                    mentioning_message_ids(msg.chat.id, caller.id, sender.username.lower()),
                    "message(s) of caller mentioning the bot",
                )
            if mode != "ban":
                no_ban = "\n(banning disabled for this chat)"
            elif (
                is_established_user(sender.id)
                and recent_spam_count(sender.id) < cfg.established_user.ban_threshold_messages
            ):
                # TODO: Once python-telegram-bot supports ephemeral messages
                # (bot API 10.3), send one to warn the user and contact cfg.admin_name.
                no_ban = "\n(established user, not banned)"
            else:
                try:
                    await ban_user(bot, msg, sender)
                except TelegramError as ex:
                    no_ban = f"\nFailed to ban: {ex}"

    caller_ban_text = None
    if caller:
        caller_name = caller.name
        if simulation:
            caller_ban_text = (
                f"(simulation, would have also banned {caller_name} for calling the bot)"
            )
        else:
            try:
                await ban_user(bot, msg, caller)
                caller_ban_text = f"Also banned {caller_name} for calling the bot!"
            except TelegramError as ex:
                caller_ban_text = f"Failed to ban caller {caller_name}: {ex}"

    if monitor:
        user_name = sender.name
        chat_name = (
            (msg.chat.username and "@" + msg.chat.username)
            or msg.chat.effective_name
            or str(msg.chat.id)
        )
        sampled = sampled_prob_range(sample_probs)
        text = (
            f"Posted by {user_name} in {chat_name}"
            + (f"\nSpam probability: {prob:.2f}{sampled}" if prob is not None else "")
            + f"{no_delete}{no_ban}"
            + (f"\nReasoning: {reasoning}" if reasoning else "")
        )
        await bot.send_message(
            chat_id=monitor,
            reply_to_message_id=fwd_msg.message_id if fwd_msg else None,
            text=text[:4095],
        )

    if monitor and caller_ban_text:
        await bot.send_message(
            chat_id=monitor,
            reply_to_message_id=fwd_msg.message_id if fwd_msg else None,
            text=caller_ban_text,
        )
