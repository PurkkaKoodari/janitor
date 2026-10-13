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

from datetime import datetime
from typing import cast

from telegram import Bot, Message, MessageOriginUser

from janitorbot.commands import fetch_chat_name
from janitorbot.db import (
    RecordedCheck,
    content_hash,
    find_checks,
    latest_uncached_check,
)
from janitorbot.spam import parse_message, prepare_llm_check


async def _llm_content_hash(text: str, media_info: str) -> bytes:
    """Return the content hash the message would be cached under."""
    llm_user_message = cast(str, await prepare_llm_check(text, media_info, force=True))
    return content_hash(llm_user_message)


async def describe_check(bot: Bot, check: RecordedCheck) -> list[str]:
    """Describe a recorded check: a header and each LLM sample with its reasoning."""
    chat = await fetch_chat_name(bot, check.chat_id)
    date = datetime.fromtimestamp(check.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    verdict = "spam" if check.spam else "not spam"
    parts = [f"Message {check.message_id} in {chat} sent at {date}: {verdict}"]

    if check.cached:
        original = latest_uncached_check(
            c_hash=cast(bytes, check.content_hash),
            sp_hash=check.system_prompt_hash,
            before=check.rowid,
        )
        if original and not original.cached:  # Should never be cached, but sanity check
            header, *reasoning = await describe_check(bot, original)
            parts[0] += "\nCached reasoning used:\n" + header
            parts.extend(reasoning)
        else:
            parts[0] += "\nCached reasoning used, original not found."
    else:
        if check.purged:
            parts[0] += "\nReasoning has been purged due to age."
        if not check.llm_samples:
            if check.reasoning:
                parts[0] += f"\nReasoning: {check.reasoning}"
            else:
                parts[0] += "\nReasoning not available."
        for index, (prob, reasoning) in enumerate(check.llm_samples, 1):
            sample = (
                f"Sample {index}/{len(check.llm_samples)}: {prob:.2f}"
                if len(check.llm_samples) > 1
                else f"Spam probability: {prob:.2f}"
            )
            if not reasoning:
                reasoning = "Not available, may be purged due to age"
            if not check.purged:
                sample += f"\nReasoning: {reasoning}"
            parts.append(sample)

    return parts


async def reply_checks(bot: Bot, msg: Message, checks: list[RecordedCheck]) -> None:
    """
    Reply with the description of each check as separate messages.

    Cached verdicts are dropped if their message also has an uncached LLM verdict,
    otherwise described with their original.
    """
    messages_with_useful_checks = set(
        (check.chat_id, check.message_id)
        for check in checks
        if check.llm_samples and not check.purged and not check.cached
    )
    replies: list[str] = []
    for check in checks:
        if check.cached and (check.chat_id, check.message_id) in messages_with_useful_checks:
            continue
        parts = await describe_check(bot, check)
        for part in parts:
            if not replies or len(replies[-1]) + 2 + len(part) > 4095:
                replies.append(part[:4095])
            else:
                replies[-1] += f"\n\n{part}"
    for reply in replies:
        await msg.reply_text(reply)


async def search_forwarded(bot: Bot, msg: Message) -> bool:
    """
    Reply to a message forwarded by an admin with the recorded checks of the original message.

    Matches on sender, send date and content, then without content in case the message was
    edited. Returns False if nothing matched, so the message should be simulated instead.
    """
    origin = msg.forward_origin
    assert origin is not None
    # The sender is only known for users who allow linking forwards to their account.
    user_id = origin.sender_user.id if isinstance(origin, MessageOriginUser) else None
    timestamp = int(origin.date.timestamp())
    text, _, media_info = parse_message(msg)
    c_hash = await _llm_content_hash(text, media_info)

    # Search with user + timestamp + content, or just the timestamp + content if user is hidden.
    checks = find_checks(user_id, timestamp, c_hash)
    if not checks and user_id is not None:
        # User is visible, but no match with content hash. May be an edit we haven't seen,
        # or perhaps a link description changed. Retry with just user + timestamp.
        checks = find_checks(user_id, timestamp, None)
        if checks:
            await msg.reply_text(
                "Original spam check not found, showing messages/edits with that user and timestamp."
            )
    if not checks:
        # No match with timestamp. Try with just the content hash, which may match a different message,
        # but still produces a "search" result rather than a simulation.
        check = latest_uncached_check(c_hash=c_hash)
        if check:
            checks = [check]
            await msg.reply_text(
                "Original spam check not found, showing latest message with the same content."
            )
    if not checks:
        # Fall back to normal simulation.
        await msg.reply_text("No matching spam checks found, rechecking the message.")
        return False
    await reply_checks(bot, msg, checks)
    return True


async def search_text(bot: Bot, msg: Message, text: str) -> None:
    """Reply to a ``search:`` message with the latest uncached LLM check of the same content."""
    _, _, media_info = parse_message(msg)
    c_hash = await _llm_content_hash(text, media_info)
    check = latest_uncached_check(c_hash=c_hash)
    if check is None:
        await msg.reply_text("No matching spam checks found.")
        return
    await reply_checks(bot, msg, [check])
