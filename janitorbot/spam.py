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
from html.parser import HTMLParser
from typing import cast
from collections.abc import Sequence

import httpx

from telegram import Bot, Message, MessageEntity, ReplyParameters, User
from telegram.error import TelegramError
from telegram.helpers import escape_markdown

import openrouter
from openrouter.components import ChatResult, ChatFormatJSONSchemaConfig, ChatJSONSchemaConfig

from janitorbot.config import Rule, cfg
from janitorbot.db import (
    chat_mode,
    chat_purpose,
    effective_admins,
    effective_moderated_chats,
)
from janitorbot.log import logger

or_client = openrouter.OpenRouter(api_key=cfg.llm.openrouter_key)


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


TME_LINK_REGEX = re.compile(r"https?://t\.me/[^\s)\]]+")


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


async def prepare_llm_check(text: str, media_info: str) -> str | None:
    """Build the user message content to send to the LLM, embedding attachment info and any
    linked Telegram post description as an <attachments> tag, as described in the system
    prompt's <message_format> section.

    Return None if the message has too little data to be worth checking via LLM."""
    # A media with a description always gets forwarded to the LLM.
    # This is checked via formatting, as any media_info usually exceeds cfg.llm.min_length.
    force_llm = "\n\n" in media_info or ":" in media_info

    tme_link = TME_LINK_REGEX.search(text)
    if tme_link:
        description = await fetch_tme_description(tme_link.group())
        if description:
            media_info = f"{media_info}\n\nThe message contains a Telegram link with the description:\n\n{description}".strip()
            # A Telegram link with a description always gets forwarded to the LLM.
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


async def llm_check(
    user_message: str,
    purpose: str,
    *,
    retry: int = cfg.llm.retry_count,
    recursive: bool = False,
) -> tuple[float, str, list[tuple[float, str]]]:
    """
    Check the message with the LLM and react to spam.

    If the result reaches the multisample threshold and this is not itself a recursive
    (multisample) call, the check is repeated in parallel and the median result is used.

    Returns a tuple of (spam_probability, reasoning, samples).
    """
    response = None
    usage: str | int = "?"
    content = "?"
    try:
        response = await asyncio.wait_for(llm_api(user_message, purpose), timeout=cfg.llm.timeout)
        usage = int(response.usage.total_tokens) if response.usage else "?"
        content = cast(str, response.choices[0].message.content)
        prob = cast(float, json.loads(content)["prob"])
        logger.info("LLM check complete, %s tokens, spam probability: %.2f", usage, prob)
    except Exception as e:
        reasoning = ""
        if response is not None:
            try:
                reasoning = response.choices[0].message.reasoning or ""
            except Exception:
                pass
        logger.error(
            "LLM check failed, %s tokens, result: %s, reasoning: %s",
            usage,
            content,
            reasoning,
            exc_info=True,
        )
        if retry:
            await asyncio.sleep(cfg.llm.retry_delay)
            return await llm_check(user_message, purpose, retry=retry - 1, recursive=recursive)
        else:
            raise RuntimeError(
                f"Openrouter failed with {str(e)} after {usage} tokens\nReasoning: {reasoning}"
            ) from e

    reasoning = response.choices[0].message.reasoning or ""
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
                llm_check(user_message, purpose, recursive=True)
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


def sampled_prob_range(samples: list[tuple[float, str]]) -> str:
    if len(samples) < 2:
        return ""
    return f" ({samples[0][0]:.2f}\u2026{samples[-1][0]:.2f})"


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

    llm_user_message = await prepare_llm_check(text, media_info)

    # If the message has too little data to check, skip the LLM and assume it's not spam.
    if llm_user_message is None:
        logger.info("Short message, skipping LLM check!")
        return True

    if sender.id in effective_admins() and text.startswith("debug:\n"):
        text = text[len("debug:\n") :]
        system_prompt = cfg.llm.system_prompt.format(purpose=purpose)
        logger.info("Debug mode, sending system prompt to admin.")
        await bot.send_message(
            chat_id=sender.id,
            reply_parameters=ReplyParameters(chat_id=msg.chat.id, message_id=msg.message_id),
            text=f"System prompt:\n{system_prompt}\n\nMessage:\n{llm_user_message}"[:4095],
        )
        return True

    # Now run the LLM check and react to the result.
    logger.info("LLM checking message...")
    try:
        prob, reasoning, samples = await llm_check(llm_user_message, purpose)
        if prob >= cfg.llm.threshold:
            # Likely spam, delete and ban, and send the reasoning to the monitor chat for review.
            await attempt_delete_ban(bot, msg, prob=prob, reasoning=reasoning, samples=samples)
            return False
        if cfg.debug:
            # Not spam, but still send the reasoning to the admins for debugging.
            sampled = sampled_prob_range(samples)
            await bot.send_message(
                chat_id=cfg.admins[0],
                reply_parameters=ReplyParameters(chat_id=msg.chat.id, message_id=msg.message_id),
                text=f"Spam probability: {prob:.2f}{sampled}\nReasoning: {reasoning}"[:4095],
            )

        if cfg.debug and len(samples) >= 2:
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
        return True
    except RuntimeError as e:
        if not cfg.quiet:
            await bot.send_message(
                chat_id=cfg.admins[0],
                reply_parameters=ReplyParameters(chat_id=msg.chat.id, message_id=msg.message_id),
                text=str(e)[:4095],
            )
        # Fail open to avoid false positives.
        return True


async def attempt_delete_ban(
    bot: Bot,
    msg: Message,
    prob: float | None = None,
    reasoning: str | None = None,
    samples: list[tuple[float, str]] = [],
):
    """
    Attempt to delete the message and ban the sender, and forward the message to the monitor chat with the reasoning.
    """
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
            if mode != "ban":
                no_ban = "\n(banning disabled for this chat)"
            else:
                try:
                    await msg.chat.ban_member(user_id=sender.id)
                except TelegramError as ex:
                    no_ban = f"\nFailed to ban: {ex}"

    caller_ban_text = None
    if caller:
        caller_name = (
            (caller.username and "@" + caller.username) or caller.full_name or str(caller.id)
        )
        if simulation:
            caller_ban_text = (
                f"(simulation, would have also banned {caller_name} for calling the bot)"
            )
        else:
            try:
                await msg.chat.ban_member(user_id=caller.id)
                caller_ban_text = f"Also banned {caller_name} for calling the bot!"
            except TelegramError as ex:
                caller_ban_text = f"Failed to ban caller {caller_name}: {ex}"

    if monitor:
        user_name = (
            (sender.username and "@" + sender.username) or sender.full_name or str(sender.id)
        )
        chat_name = (
            (msg.chat.username and "@" + msg.chat.username)
            or msg.chat.effective_name
            or str(msg.chat.id)
        )
        sampled = sampled_prob_range(samples)
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
