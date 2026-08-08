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

import time
from typing import cast

from telegram import (
    Bot,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    Message,
    User,
)
from telegram.error import BadRequest

from janitorbot.config import cfg
from janitorbot.db import cursor, db
from janitorbot.spam import matches, parse_message

recent_senders: dict[int, float] = {}
recent_ignored: dict[int, float] = {}


class ReplyNotFound(Exception):
    pass


def make_keyboard(msg: Message) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("See message", url=msg.link),
                InlineKeyboardButton("Delete this", callback_data="delete:" + str(msg.message_id)),
            ]
        ]
    )


async def delete_from_channel(bot: Bot, msg_id: int):
    """
    Deletes a forwarded message from the channel and the DB.
    """
    await bot.delete_message(chat_id=cfg.chats.channel, message_id=msg_id)
    cursor.execute("DELETE FROM messages WHERE id = ?", [msg_id])
    db.commit()


async def forward_to_channel(bot: Bot, msg: Message):
    """
    Forward a spam-checked SOURCE message to the CHANNEL, applying ignore and recency heuristics.
    """
    global recent_senders, recent_ignored

    sender = cast(User, msg.from_user)

    # Ignore messages outside SOURCE chats.
    if msg.chat_id not in cfg.chats.source:
        return

    text, has_media, _ = parse_message(msg)

    # Ignore messages that have no text and no media, as they are unlikely to be useful.
    if not text and not has_media:
        return

    # Ignore messages that match ignore rules, and mark the user as a dibser.
    if next((rule for rule in cfg.channel_ignore.rules if matches(rule, has_media, text)), None):
        recent_ignored[sender.id] = time.time()
        return

    # Ignore short messages that have no media from users who haven't sent
    # anything useful in a while, as they are unlikely to be useful either.
    for user, ts in list(recent_senders.items()):
        if ts < time.time() - cfg.channel_ignore.recent_seconds:
            del recent_senders[user]
    if (
        sender.id not in recent_senders
        and not has_media
        and len(text) <= cfg.channel_ignore.length_min_from_silent
    ):
        return

    # Ignore short messages that have no media from recently ignored users,
    # as they are likely to be follow-ups to the original ignored message.
    for user, ts in list(recent_ignored.items()):
        if ts < time.time() - cfg.channel_ignore.recent_seconds:
            del recent_ignored[user]
    if (
        sender.id in recent_ignored
        and not has_media
        and len(text) <= cfg.channel_ignore.length_min_from_ignored
    ):
        return

    recent_senders[sender.id] = time.time()

    new_msg_id: int | None = None
    try:
        # If this message is a reply, try to find the corresponding message in the channel and reply to it.
        if not msg.reply_to_message:
            raise ReplyNotFound
        cursor.execute(
            "SELECT id FROM messages WHERE orig_id = ?", [msg.reply_to_message.message_id]
        )
        row = cursor.fetchone()
        if not row:
            raise ReplyNotFound

        result = await msg.copy(
            chat_id=cfg.chats.channel,
            reply_markup=make_keyboard(msg),
            reply_to_message_id=row[0],
        )
        new_msg_id = result.message_id
    except Exception as ex:
        # If the message is not a reply, or if we can't find the replied message in the channel, just post it without replying.
        if isinstance(ex, ReplyNotFound) or (
            isinstance(ex, BadRequest) and "replied message not found" in str(ex).lower()
        ):
            result = await msg.copy(
                chat_id=cfg.chats.channel,
                reply_markup=make_keyboard(msg),
            )
            new_msg_id = result.message_id

    if new_msg_id:
        cursor.execute(
            "INSERT INTO messages VALUES (?, ?, ?)", [new_msg_id, msg.message_id, sender.id]
        )
        db.commit()


async def edit_in_channel(bot: Bot, msg: Message):
    """
    Apply an edit of a spam-checked SOURCE message to its forwarded copy in the CHANNEL.
    """
    # Ignore edits outside SOURCE chats.
    if msg.chat.id not in cfg.chats.source:
        return

    # Try to find the corresponding message in the channel.
    cursor.execute("SELECT id FROM messages WHERE orig_id = ?", [msg.message_id])
    row = cursor.fetchone()
    if not row:
        return

    text, has_media, _ = parse_message(msg)

    # If the message now matches an ignore rule with delete=True, delete it from the channel.
    if any(rule.delete and matches(rule, has_media, text) for rule in cfg.channel_ignore.rules):
        await delete_from_channel(bot, row[0])
        return

    # Otherwise, update the message in the channel to reflect the edit.
    if msg.text is not None:
        await bot.edit_message_text(
            chat_id=cfg.chats.channel,
            message_id=row[0],
            text=msg.text,
            entities=msg.entities,
            reply_markup=make_keyboard(msg),
        )
    elif msg.caption is not None:
        await bot.edit_message_caption(
            chat_id=cfg.chats.channel,
            message_id=row[0],
            caption=msg.caption,
            caption_entities=msg.caption_entities,
            reply_markup=make_keyboard(msg),
        )
