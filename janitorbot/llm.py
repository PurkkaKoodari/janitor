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

import asyncio, json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

import httpx

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

from janitorbot.config import cfg
from janitorbot.log import logger

or_client = openrouter.OpenRouter(api_key=cfg.llm.openrouter_key)

# Errors that are generally fatal and the bot operator's fault.
OPERATOR_ERRORS = (
    UnauthorizedResponseError,
    PaymentRequiredResponseError,
)


def _extract_provider(e: Exception) -> str | None:
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


async def _resolve_provider(e: Exception, response: ChatResult | None) -> str | None:
    """
    Determine the upstream provider responsible for a failed LLM check.

    HTTP-level failures carry the provider on the exception itself (see ``extract_provider``).
    But a provider can also return a 200 response that fails our JSON schema (e.g. a null or
    malformed completion) -- OpenRouter doesn't treat that as an error, so no exception carries
    the provider. In that case, look it up via the generation ID instead, so misbehaving
    providers can still be identified and excluded via ``cfg.llm.model`` routing preferences.
    """
    provider = _extract_provider(e)
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


def _parse_llm_response(response: ChatResult) -> tuple[float, str, int | None]:
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


async def _handle_llm_error(
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
            _parse_llm_response(response)
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
    provider = await _resolve_provider(err_response, response)
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


def _handle_llm_error_async(
    fetch: asyncio.Task[ChatResult],
    exc: Exception,
    fatal: bool,
    report: Callable[[LLMError], Awaitable[None]] | None,
) -> asyncio.Task[LLMError]:
    """
    Handle an LLM error asynchronously, reporting it to the super admin if configured.

    This is a fire-and-forget wrapper around ``handle_llm_error`` that runs in the background.
    """
    handle = asyncio.create_task(_handle_llm_error(fetch, exc, fatal=fatal, report=report))
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
        prob, reasoning, usage = _parse_llm_response(response)
        logger.info("LLM result: %s tokens, spam probability: %.2f", usage or "?", prob)
    except Exception as exc:
        # If the error is obviously the operator's problem, don't bother retrying.
        if isinstance(exc, OPERATOR_ERRORS):
            retry = 0

        # Process the error further asynchronously.
        handle = _handle_llm_error_async(
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
        _handle_llm_error_async(task, exc, fatal=False, report=report_error)

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
