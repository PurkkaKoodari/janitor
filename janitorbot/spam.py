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

import random, re
from dataclasses import dataclass
from typing import cast

from telegram import Bot, Message, ReplyParameters, User
from telegram.error import TelegramError

from janitorbot.channel import bulk_delete, delete_mirrors
from janitorbot.config import LLMConfig, cfg
from janitorbot.db import (
    cached_verdict,
    chat_mode,
    chat_purpose,
    content_hash,
    effective_admins,
    effective_spam_checked_chats,
    is_established_user,
    is_new_user,
    mentioning_message_ids,
    recent_spam_count,
    record_message_check,
    system_prompt_hash,
    text_hash,
    user_message_ids,
)
from janitorbot.llm import DeferredLLMError, LLMError, Verdict, llm_check
from janitorbot.log import logger, message_logging
from janitorbot.messages import (
    CYRILLIC_WORD_REGEX,
    ParsedMessage,
    extract_mentions,
    llm_user_message,
)
from janitorbot.util import spawn_task


@dataclass
class LLMContent:
    """A message prepared for LLM checking, with the user message and its content hash."""

    text: str
    user_message: str
    c_hash: bytes
    llm_viable: bool


async def prepare_llm_content(parsed: ParsedMessage) -> LLMContent:
    """Prepare the message for LLM checking."""
    user_message, llm_viable = await llm_user_message(parsed)
    return LLMContent(
        text=parsed.text,
        user_message=user_message,
        c_hash=content_hash(user_message),
        llm_viable=llm_viable,
    )


@dataclass
class LLMInput(LLMContent):
    """A message prepared for LLM checking, with the user/system prompts and hashes."""

    system_prompt: str
    sp_hash: bytes


def prepare_llm_input(prepared: LLMContent, purpose: str, llm_cfg: LLMConfig) -> LLMInput:
    """Prepare the message for LLM checking, with the system prompt and its hash."""
    system_prompt = llm_cfg.system_prompt.format(purpose=purpose)
    return LLMInput(
        text=prepared.text,
        user_message=prepared.user_message,
        c_hash=prepared.c_hash,
        llm_viable=prepared.llm_viable,
        system_prompt=system_prompt,
        sp_hash=system_prompt_hash(system_prompt, llm_cfg),
    )


def record_checked_message(
    msg: Message,
    prepared: ParsedMessage | LLMContent,
    verdict: Verdict,
    *,
    primary: bool = True,
) -> None:
    """
    Store the result of spam checking the message in the database, with either the LLM samples or
    the non-LLM reasoning. Non-primary checks don't affect the message's verdict.
    """
    record_message_check(
        chat_id=msg.chat.id,
        message_id=msg.message_id,
        user_id=cast(User, msg.from_user).id,
        timestamp=int(msg.date.timestamp()),
        mentions=extract_mentions(msg),
        verdict=verdict,
        t_hash=text_hash(prepared.text),
        sp_hash=prepared.sp_hash if isinstance(prepared, LLMInput) else None,
        c_hash=prepared.c_hash if isinstance(prepared, LLMContent) else None,
        primary=primary,
    )


async def run_a_b_test(
    bot: Bot,
    msg: Message,
    prepared: LLMInput,
    b_cfg: LLMConfig,
    a_verdict: Verdict,
    reevaluate: bool,
    target: int,
) -> None:
    """
    Check the message with the B side LLM config, reusing cached verdicts like the A side, and
    report to the target if the verdict differs from the A side verdict.
    """

    async def report_error(error: LLMError) -> None:
        # Report to the super admin if configured.
        if error.should_report and not cfg.quiet:
            await bot.send_message(
                **msg.build_reply_arguments(
                    target_chat_id=cfg.admins[0], allow_sending_without_reply=True
                ),
                text=f"A/B test: {error}"[:4095],
            )

    b_verdict = (
        None if reevaluate else cached_verdict(prepared.sp_hash, prepared.c_hash, b_cfg.threshold)
    )

    if a_verdict.cached and b_verdict:
        # Both verdicts come from earlier checks, so this is likely a repeat of an earlier report.
        logger.info("Both verdicts of A/B test verdict cached, not reporting")
        return
    elif b_verdict:
        logger.info("Using cached A/B test verdict, spam probability: %.2f", b_verdict.prob)
    else:
        try:
            logger.info("LLM checking message with A/B test config...")
            with message_logging(msg.chat.id, msg.message_id, extra="a_b_test"):
                b_verdict = await llm_check(
                    prepared.user_message,
                    prepared.system_prompt,
                    llm_cfg=b_cfg,
                    report_error=report_error,
                )
        except DeferredLLMError:
            # Will be reported asynchronously by report_error.
            return
        record_checked_message(msg, prepared, b_verdict, primary=False)

    logger.info(
        "A/B test: A spam probability %.2f (%s), B spam probability %.2f (%s)",
        a_verdict.prob,
        a_verdict.spam_str,
        b_verdict.prob,
        b_verdict.spam_str,
    )
    if b_verdict.spam == a_verdict.spam:
        return

    await bot.send_message(
        **msg.build_reply_arguments(target_chat_id=target, allow_sending_without_reply=True),
        text=f"A/B test verdicts differ:\n"
        f"A: {a_verdict.spam_str}, probability {a_verdict.sample_range}\n"
        f"B: {b_verdict.spam_str}, probability {b_verdict.sample_range}\n"
        f"B reasoning: {b_verdict.reasoning}"[:4095],
    )


async def check_spam(bot: Bot, msg: Message, parsed: ParsedMessage) -> bool:
    """
    Check the message for spam and react to spam.

    Returns True if the message is likely not spam, False if it is likely spam.
    """
    chat_id = msg.chat.id
    sender = cast(User, msg.from_user)
    is_evaluation = msg.chat.id in effective_admins()

    # Check for forwarded special cases of spam, and react to them.
    # If the message is likely spam, don't process it further.
    if (
        cfg.spam.forwarded_story
        and msg.chat.id in effective_spam_checked_chats()
        and msg.story
        and msg.story.chat.id != sender.id
    ):
        await attempt_delete_ban(bot, msg, parsed, reasoning="Forwarded story")
        return False

    fwd_from_channel = msg.forward_origin and msg.forward_origin.type == "channel"
    if msg.chat.id in effective_spam_checked_chats() and fwd_from_channel:
        if cfg.spam.channel_forwarded_with_keyboard and msg.reply_markup:
            await attempt_delete_ban(
                bot, msg, parsed, reasoning="Forwarded message from channel with inline keyboard"
            )
            return False
        if (
            cfg.spam.channel_forwarded_cyrillic_and_links
            and parsed.text
            and re.search(cfg.spam.suspicious_link_regex, parsed.text)
            and re.search(CYRILLIC_WORD_REGEX, parsed.text, re.I)
        ):
            await attempt_delete_ban(
                bot,
                msg,
                parsed,
                reasoning="Forwarded message from channel with Cyrillic and suspicious links",
            )
            return False

    # Also check for spam regexes. These can save cost and time by avoiding LLM checks for obvious spam.
    if (
        not parsed.debug
        and chat_id in effective_spam_checked_chats()
        and parsed.text
        and re.search(cfg.spam_pattern, parsed.text, re.I)
    ):
        logger.info("Message matches spam regex, banning without LLM check!")
        matching_regex = next(
            (regex for regex in cfg.spam.regex if re.search(regex, parsed.text, re.I)), None
        )
        await attempt_delete_ban(
            bot, msg, parsed, reasoning=f"Matched spam regex: {matching_regex}"
        )
        return False

    # For other messages, run the LLM check and then process them in the after function.
    if parsed.purpose is not None:
        purpose = parsed.purpose
        logger.info("Using custom purpose from admin: %s", purpose)
    else:
        purpose = chat_purpose(msg.chat.id)

    prepared = prepare_llm_input(await prepare_llm_content(parsed), purpose, cfg.llm)

    if parsed.debug:
        logger.info("Debug mode, sending system prompt to admin.")
        await bot.send_message(
            chat_id=sender.id,
            reply_parameters=ReplyParameters(chat_id=msg.chat.id, message_id=msg.message_id),
            text=f"System prompt:\n{prepared.system_prompt}\n\nMessage:\n{prepared.user_message}"[
                :4095
            ],
        )
        return True

    # If the message has too little data to check, skip the LLM and assume it's not spam.
    if not prepared.llm_viable:
        logger.info("Short message, skipping LLM check!")
        record_checked_message(
            msg, prepared, Verdict(spam=False, reasoning="Short message, skipped LLM check")
        )
        if is_evaluation:
            await msg.reply_text(
                f"The message is short enough (< {cfg.llm.min_length} characters) to be assumed not spam.",
                do_quote=True,
            )
        return True

    # Reuse the verdict of an identical earlier message checked with the same system prompt.
    if parsed.reevaluate:
        logger.info("Reevaluation requested, bypassing verdict cache.")
        verdict = None
    else:
        verdict = cached_verdict(prepared.sp_hash, prepared.c_hash, cfg.llm.threshold)

    # Now run the LLM check and react to the result.
    try:
        if verdict:
            logger.info("Using cached verdict, spam probability: %.2f", verdict.prob)
        else:

            async def report_error(error: LLMError) -> None:
                # Record fatal errors in the database.
                if error.fatal:
                    record_checked_message(
                        msg,
                        prepared,
                        Verdict(spam=False, reasoning=str(error)),
                    )
                # Report to the super admin if configured.
                reported_to = None
                if error.should_report and not cfg.quiet:
                    reported_to = cfg.admins[0]
                    await bot.send_message(
                        chat_id=cfg.admins[0],
                        reply_parameters=ReplyParameters(
                            chat_id=msg.chat.id, message_id=msg.message_id
                        ),
                        text=str(error)[:4095],
                    )
                # Always report fatal errors in evaluations to the evaluating admin.
                if error.fatal and is_evaluation and sender.id != reported_to:
                    await msg.reply_text(str(error)[:4095], do_quote=True)

            logger.info("LLM checking message...")
            verdict = await llm_check(
                prepared.user_message, prepared.system_prompt, report_error=report_error
            )

        # Store the verdict.
        record_checked_message(msg, prepared, verdict)

        if verdict.spam:
            # Likely spam, delete and ban, and send the reasoning to the monitor chat for review.
            await attempt_delete_ban(bot, msg, prepared, verdict)
        else:
            if is_evaluation or cfg.debug:
                # Not spam, but still send the reasoning to the admin(s) for debugging/evaluation.
                # Reply to the sender if in DMs, otherwise send to the super admin.
                target = sender.id if is_evaluation else cfg.admins[0]
                await bot.send_message(
                    **msg.build_reply_arguments(
                        target_chat_id=target, allow_sending_without_reply=True
                    ),
                    text=f"Spam probability: {verdict.sample_range}"
                    f"\nReasoning: {verdict.reasoning}"[:4095],
                )

        if (
            not cfg.quiet
            and cfg.llm.outlier_threshold < 1.0
            and not verdict.cached
            and len(verdict.samples_to_store) >= 2
        ):
            # Attempt to detect outliers in the sampling in case there's room for improvement in the system prompt.
            for sample_prob, sample_reasoning in (
                verdict.samples_to_store[0],
                verdict.samples_to_store[-1],
            ):
                if abs(sample_prob - verdict.prob) >= cfg.llm.outlier_threshold:
                    await bot.send_message(
                        chat_id=cfg.admins[0],
                        reply_parameters=ReplyParameters(
                            chat_id=msg.chat.id, message_id=msg.message_id
                        ),
                        text=f"Outlier in sampling: {sample_prob:.2f} vs median {verdict.prob:.2f}\n"
                        f"Reasoning: {sample_reasoning}"[:4095],
                    )

        # Run an A/B test in the background if configured. Report to the evaluating admin if
        # this is an evaluation, otherwise to the super admin.
        if a_b_llm_cfg := cfg.llm.a_b_test_config():
            sample_rate = (
                cfg.llm.a_b_test.spam_sample_rate
                if verdict.spam
                else cfg.llm.a_b_test.nonspam_sample_rate
            )
            if is_evaluation or random.random() < sample_rate:
                spawn_task(
                    run_a_b_test(
                        bot,
                        msg,
                        prepare_llm_input(prepared, purpose, a_b_llm_cfg),
                        a_b_llm_cfg,
                        a_verdict=verdict,
                        reevaluate=parsed.reevaluate,
                        target=sender.id if is_evaluation else cfg.admins[0],
                    ),
                    "A/B test",
                )

        return not verdict.spam
    except DeferredLLMError:
        # Fail open to avoid false positives. Will be reported asynchronously by report_error.
        return True


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
    prepared: ParsedMessage | LLMContent,
    reasoning: str | Verdict,
):
    """
    Attempt to delete the message and ban the sender, and forward the message to the monitor chat with the reasoning.

    If reasoning is a string, the message is recorded as spam with the non-LLM reasoning; LLM checks
    record their samples themselves.
    """
    if isinstance(reasoning, str):
        verdict = Verdict(spam=True, reasoning=reasoning)
        record_checked_message(msg, prepared, verdict)
    else:
        # The LLM path also records non-spam verdicts, so it already records all checks itself.
        verdict = reasoning

    sender = cast(User, msg.from_user)
    caller = msg.guest_bot_caller_user
    from_admin = (
        sender.id
        if sender.id in effective_admins()
        else caller.id if (caller and caller.id in effective_admins()) else None
    )
    monitor = from_admin or cfg.chats.monitor

    reply_to = None
    if monitor == msg.chat.id:
        # Evaluations are replied to directly, no need to forward the message to the admin.
        reply_to = msg.message_id
    elif monitor:
        try:
            fwd_msg = await msg.forward(chat_id=monitor)
            reply_to = fwd_msg.message_id
        except TelegramError:
            logger.warning("Failed to forward message to monitor chat", exc_info=True)

    no_delete = ""
    no_ban = ""
    simulation = from_admin or msg.chat.id not in effective_spam_checked_chats()
    if simulation:
        no_ban = "(simulation, would have been deleted/banned)"
    else:
        mode = chat_mode(msg.chat.id)
        if mode == "test":
            no_ban = "(test mode, no action taken)"
        else:
            try:
                await msg.delete()
            except TelegramError as ex:
                no_delete = f"Failed to delete: {ex}"
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
                no_ban = "(banning disabled for this chat)"
            elif (
                is_established_user(sender.id)
                and recent_spam_count(sender.id) < cfg.established_user.ban_threshold_messages
            ):
                # TODO: Once python-telegram-bot supports ephemeral messages
                # (bot API 10.3), send one to warn the user and contact cfg.admin_name.
                no_ban = "(established user, not banned)"
            else:
                try:
                    await ban_user(bot, msg, sender)
                except TelegramError as ex:
                    no_ban = f"Failed to ban: {ex}"

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
        parts: list[str] = []
        if monitor != msg.chat.id:
            parts.append(f"Posted by {user_name} in {chat_name}")
        if verdict.sample_probs:
            parts.append(f"Spam probability: {verdict.sample_range}")
        if no_delete:
            parts.append(no_delete)
        if no_ban:
            parts.append(no_ban)
        if verdict.reasoning:
            parts.append(f"Reasoning: {verdict.reasoning}")
        await bot.send_message(
            chat_id=monitor,
            reply_to_message_id=reply_to,
            text="\n".join(parts)[:4095],
        )

    if monitor and caller_ban_text:
        await bot.send_message(
            chat_id=monitor,
            reply_to_message_id=reply_to,
            text=caller_ban_text,
        )
