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

import asyncio, glob, os, re, sys, threading, time
from html import escape
from typing import cast

from telegram import Bot, Message, Update, User
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from janitorbot.channel import edit_in_channel, forward_to_channel
from janitorbot.commands import (
    handle_admin,
    handle_callback_query,
    handle_chats,
    handle_hello,
    handle_mode,
    handle_moderate,
    handle_unadmin,
    handle_unmoderate,
    post_init,
)
from janitorbot.config import cfg
from janitorbot.db import (
    effective_admins,
    effective_member_chats,
    effective_moderated_chats,
    effective_spam_checked_chats,
)
from janitorbot.log import logger, message_logging
from janitorbot.spam import attempt_delete_ban, check_spam, parse_message


async def process_message(bot: Bot, msg: Message):
    """
    Route an incoming message: handle the bot being added to a chat, run spam checks, and
    forward the message to the CHANNEL only once it has passed all spam checks.
    """
    sender = cast(User, msg.from_user)

    # If the bot was just added to a new chat, check if it's a chat we want to be in.
    if msg.new_chat_members and any(user.id == bot.id for user in msg.new_chat_members):
        # If the chat is not known and the person who added the bot is not an admin, leave the chat.
        if msg.chat.id not in effective_member_chats() and sender.id not in effective_admins():
            await msg.chat.leave()
        elif sender.id in effective_admins():
            # Otherwise, send the chat id to the admins so they can add it to the config.
            sender_name = (
                (sender.username and "@" + sender.username) or sender.full_name or str(sender.id)
            )
            monitor = cfg.chats.monitor or sender.id
            await bot.send_message(
                chat_id=monitor,
                text=f'{sender_name} added the bot to a new chat "{escape(msg.chat.effective_name or "")}" (ID <code>{msg.chat.id}</code>).',
                parse_mode="HTML",
            )
        return

    if msg.chat.id not in effective_spam_checked_chats():
        # If the source chat is a private chat, reply with instructions.
        if msg.chat.type == "private":
            await msg.reply_text(
                text=f"Hello! This bot is for moderating groups.\n\nPlease talk to {cfg.admin_name} if you want to use this instance.\n\nSource code: https://github.com/PurkkaKoodari/janitor",
            )
        # Ignore messages from unknown chats.
        return

    text, _, _ = parse_message(msg)

    # Check for forwarded special cases of spam, and react to them. If the message is likely spam, don't process it further.
    if (
        cfg.spam.forwarded_story
        and msg.chat.id in effective_moderated_chats()
        and msg.story
        and msg.story.chat.id != sender.id
    ):
        await attempt_delete_ban(bot, msg, reasoning="Forwarded story")
        return

    fwd_from_channel = msg.forward_origin and msg.forward_origin.type == "channel"
    if msg.chat.id in effective_moderated_chats() and fwd_from_channel:
        if cfg.spam.channel_forwarded_with_keyboard and msg.reply_markup:
            await attempt_delete_ban(
                bot, msg, reasoning="Forwarded message from channel with inline keyboard"
            )
            return
        if (
            cfg.spam.channel_forwarded_cyrillic_and_links
            and text
            and re.search(cfg.spam.suspicious_link_regex, text)
            and re.search(r"[\u0400-\u052F]", text, re.I)
        ):
            await attempt_delete_ban(
                bot,
                msg,
                reasoning="Forwarded message from channel with Cyrillic and suspicious links",
            )
            return

    # Check for spam and react to it. If the message is likely spam, don't process it further.
    if not await check_spam(bot, msg):
        return

    # The message passed all spam checks; hand it off to the channel forwarding step.
    await forward_to_channel(bot, msg)


async def process_edit(bot: Bot, msg: Message):
    """
    Run spam checks on an edited message, then apply the edit to its CHANNEL copy.
    """
    # Ignore edits where we wouldn't have checked the original message for spam.
    if msg.chat.id not in effective_spam_checked_chats():
        return

    # Check for spam and react to it. If the message is likely spam, don't process it further.
    if not await check_spam(bot, msg):
        return

    # The edit passed all spam checks; hand it off to the channel edit step.
    await edit_in_channel(bot, msg)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = cast(Message, update.message)
        with message_logging(msg.chat.id, msg.message_id):
            asyncio.create_task(process_message(context.bot, msg))
    except Exception:
        logger.error("Error handling message", exc_info=True)


async def handle_edited_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = cast(Message, update.edited_message)
        with message_logging(msg.chat.id, msg.message_id):
            asyncio.create_task(process_edit(context.bot, msg))
    except Exception:
        logger.error("Error handling edited message", exc_info=True)


def _watch_and_restart():
    # Watch the launcher, the config, and every package module so any code change restarts the bot.
    watched = ["janitor.py", "config.toml"] + glob.glob("janitorbot/*.py")
    mtimes = {f: os.path.getmtime(f) for f in watched}
    while True:
        time.sleep(1)
        for f in watched:
            try:
                mtime = os.path.getmtime(f)
            except OSError:
                continue
            if mtime != mtimes[f]:
                logger.info("File %s changed, restarting...", f)
                os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    threading.Thread(target=_watch_and_restart, daemon=True).start()
    app = (
        Application.builder()
        .token(cfg.bot_token)
        .post_init(post_init)  # pyright: ignore[reportUnknownMemberType]
        .build()
    )
    app.add_handler(CommandHandler("moderate", handle_moderate))
    app.add_handler(CommandHandler("unmoderate", handle_unmoderate))
    app.add_handler(CommandHandler("mode", handle_mode))
    app.add_handler(CommandHandler("chats", handle_chats))
    app.add_handler(CommandHandler("hello", handle_hello))
    app.add_handler(CommandHandler("admin", handle_admin))
    app.add_handler(CommandHandler("unadmin", handle_unadmin))
    app.add_handler(MessageHandler(filters.UpdateType.MESSAGE, handle_message))
    app.add_handler(MessageHandler(filters.UpdateType.EDITED_MESSAGE, handle_edited_message))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.run_polling()
