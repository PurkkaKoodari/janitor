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

import asyncio, json, logging, os, re, sqlite3, sys, threading, time, tomllib
from functools import cached_property
from typing import cast, Callable, Coroutine, TypeAlias

from pydantic import BaseModel, computed_field, field_validator

from telegram import (
    Bot,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    Message,
    ReplyParameters,
    Update,
    User,
)
from telegram.error import TelegramError, BadRequest
from telegram.ext import Application, MessageHandler, CallbackQueryHandler, ContextTypes, filters

import openrouter
from openrouter.components import ResponseFormatJSONSchema, JSONSchemaConfig

format = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=format)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("JanitorBot")


AfterFunc: TypeAlias = Callable[[Bot, Message], Coroutine[None, None, None]]


class Rule(BaseModel):
    regex: str
    # None means "don't care", True means "must have media", False means "must not have media"
    media: bool | None = None
    # if True, messages matching this rule will be deleted if they are edited to match, even if they previously didn't match
    delete: bool = False


class ChatsConfig(BaseModel):
    source: list[int]
    channel: int
    moderated: list[int]
    monitor: int


class SpamConfig(BaseModel):
    regex: list[str]


class LLMConfig(BaseModel):
    openrouter_key: str
    model: str
    system_prompt: str
    threshold: float
    chat_purposes: dict[str, str]

    @field_validator("openrouter_key")
    @classmethod
    def or_key_not_empty(cls, v: str) -> str:
        assert v != "" and v != "FILLME", "OpenRouter key must be set"
        return v

    @field_validator("chat_purposes")
    @classmethod
    def purposes_has_default(cls, v: dict[str, str]) -> dict[str, str]:
        assert "default" in v, "llm.chat_purposes must have a 'default' key"
        return v


class ChannelIgnoreConfig(BaseModel):
    recent_seconds: int
    length_min_from_silent: int
    length_min_from_ignored: int
    rules: list[Rule]

    @field_validator("rules", mode="after")
    @classmethod
    def filter_invalid_rules(cls, v: list[Rule]) -> list[Rule]:
        valid: list[Rule] = []
        for rule in v:
            try:
                re.compile(rule.regex, re.I)
                valid.append(rule)
            except Exception as e:
                logger.error("Ignoring invalid regex %s: %s", rule.regex, e)
        return valid


class Config(BaseModel):
    bot_token: str
    db_name: str
    admins: list[int]
    admin_name: str
    spam: SpamConfig
    chats: ChatsConfig
    llm: LLMConfig
    channel_ignore: ChannelIgnoreConfig

    @field_validator("bot_token")
    @classmethod
    def token_not_empty(cls, v: str) -> str:
        assert v != "" and v != "FILLME", "Token must be set"
        return v

    @computed_field
    @cached_property
    def spam_pattern(self) -> str:
        return "|".join(self.spam.regex) or "a^"

    @computed_field
    @cached_property
    def moderated_chats(self) -> frozenset[int]:
        """Source and moderated chats — where spam moderation is applied."""
        return frozenset(self.chats.source) | frozenset(self.chats.moderated)

    @computed_field
    @cached_property
    def spam_checked_chats(self) -> frozenset[int]:
        """Moderated chats plus admin DM IDs — chats the bot checks for spam."""
        return self.moderated_chats | frozenset(self.admins)

    @computed_field
    @cached_property
    def member_chats(self) -> frozenset[int]:
        """Moderated chats plus the channel — chats the bot is intentionally a member of."""
        return self.moderated_chats | {self.chats.channel}


with open("config.toml", "rb") as file:
    cfg = Config.model_validate(tomllib.load(file))

recent_senders: dict[int, float] = {}
recent_ignored: dict[int, float] = {}


db = sqlite3.connect(cfg.db_name)
cursor = db.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS messages (
        id BIGINT PRIMARY KEY,
        orig_id BIGINT,
        author BIGINT
    )
    """
)

or_client = openrouter.OpenRouter(api_key=cfg.llm.openrouter_key)


def parse_message(msg: Message) -> tuple[str, bool]:
    has_media = bool(msg.photo) or msg.document is not None or msg.video is not None or msg.video_note is not None
    text = msg.text or msg.caption or ""
    return text, has_media


def matches(rule: Rule, has_media: bool, text: str) -> bool:
    if rule.media is not None and has_media != rule.media:
        return False
    return bool(re.search(rule.regex, text, re.I))


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


async def process_message(bot: Bot, msg: Message):
    """
    After the LLM check, filter messages from the SOURCE chats and forward them to the CHANNEL if they pass.
    """
    global recent_senders, recent_ignored

    sender = cast(User, msg.from_user)

    # If the bot was just added to a new chat, check if it's a chat we want to be in.
    if msg.new_chat_members and any(user.id == bot.id for user in msg.new_chat_members):
        # If the chat is not known and the person who added the bot is not an admin, leave the chat.
        if msg.chat.id not in cfg.member_chats and sender.id not in cfg.admins:
            await msg.chat.leave()
        elif sender.id in cfg.admins:
            # Otherwise, send the chat id to the admins so they can add it to the config.
            await sender.send_message(
                text=f'Added to a new chat "{msg.chat.effective_name}" ({msg.chat.id})',
            )
        return

    if msg.chat.id not in cfg.spam_checked_chats:
        # If the source chat is a private chat, reply with instructions.
        if msg.chat.type == "private":
            await msg.reply_text(
                text=f"Hello! This bot is for moderating groups.\n\nPlease talk to {cfg.admin_name} if you want to use this instance.\n\nSource code: https://github.com/PurkkaKoodari/janitor",
            )
        # Ignore messages from unknown chats.
        return

    text, has_media = parse_message(msg)

    # Check for forwarded special cases of spam, and react to them. If the message is likely spam, don't process it further.
    fwd_from_channel = msg.forward_origin and msg.forward_origin.type == "channel"
    if (
        msg.chat.id in cfg.moderated_chats
        and fwd_from_channel
        and (
            # Forwarded from channel with inline keyboard
            msg.reply_markup
            # Forwarded from channel, contains a link to VK or Telegram, and contains Cyrillic
            or (
                text
                and re.search(r"@\w+|https?://t\.me|https?://vk\.", text)
                and re.search(r"[\u0400-\u052F]", text, re.I)
            )
        )
    ):
        await msg.delete()
        await msg.chat.ban_member(user_id=sender.id)
        return

    # Check for spam and react to it. If the message is likely spam, don't process it further.
    if not await check_spam(bot, msg):
        return

    # Ignore messages outside SOURCE chats.
    if msg.chat_id not in cfg.chats.source:
        return

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
    if sender.id not in recent_senders and not has_media and len(text) <= cfg.channel_ignore.length_min_from_silent:
        return

    # Ignore short messages that have no media from recently ignored users,
    # as they are likely to be follow-ups to the original ignored message.
    for user, ts in list(recent_ignored.items()):
        if ts < time.time() - cfg.channel_ignore.recent_seconds:
            del recent_ignored[user]
    if sender.id in recent_ignored and not has_media and len(text) <= cfg.channel_ignore.length_min_from_ignored:
        return

    recent_senders[sender.id] = time.time()

    new_msg_id: int | None = None
    try:
        # If this message is a reply, try to find the corresponding message in the channel and reply to it.
        if not msg.reply_to_message:
            raise ReplyNotFound
        cursor.execute("SELECT id FROM messages WHERE orig_id = ?", [msg.reply_to_message.message_id])
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
        cursor.execute("INSERT INTO messages VALUES (?, ?, ?)", [new_msg_id, msg.message_id, sender.id])
        db.commit()


async def process_edit(bot: Bot, msg: Message):
    """
    After the LLM check, handle edits for messages forwarded to the CHANNEL.
    """
    # Ignore edits where we wouldn't have checked the original message for spam.
    if msg.chat.id not in cfg.spam_checked_chats:
        return

    # Check for spam and react to it. If the message is likely spam, don't process it further.
    if not await check_spam(bot, msg):
        return

    # Ignore edits outside SOURCE chats.
    if msg.chat.id not in cfg.chats.source:
        return

    # Try to find the corresponding message in the channel.
    cursor.execute("SELECT id FROM messages WHERE orig_id = ?", [msg.message_id])
    row = cursor.fetchone()
    if not row:
        return

    text, has_media = parse_message(msg)

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


async def llm_check(bot: Bot, msg: Message, *, retry: int = 2) -> bool:
    """
    Check the message with the LLM and react to spam.

    Returns True if the message is likely not spam, False if it is likely spam.
    """
    sender = cast(User, msg.from_user)
    text, _ = parse_message(msg)

    purpose = cfg.llm.chat_purposes.get(str(msg.chat.id), cfg.llm.chat_purposes["default"])

    response = None
    usage: str | int = "?"
    content = "?"
    try:
        response = await or_client.chat.send_async(
            model=cfg.llm.model,
            messages=[
                {"role": "system", "content": cfg.llm.system_prompt.format(purpose=purpose)},
                {"role": "user", "content": text},
            ],
            max_tokens=3000,
            temperature=0.5,
            response_format=ResponseFormatJSONSchema(
                json_schema=JSONSchemaConfig(
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
        usage = int(response.usage.total_tokens) if response.usage else "?"
        content = cast(str, response.choices[0].message.content)
        prob = json.loads(content)["prob"]
        logger.info("LLM check complete, %s tokens, spam probability: %.2f", usage, prob)
    except Exception as e:
        reasoning = ""
        if response is not None:
            try:
                reasoning = response.choices[0].message.reasoning or ""
            except Exception:
                pass
        logger.error("LLM check failed, %s tokens, result: %s, reasoning: %s", usage, content, reasoning, exc_info=True)
        if retry:
            await asyncio.sleep(1)
            return await llm_check(bot, msg, retry=retry - 1)
        else:
            text = f"Openrouter failed with {str(e)} after {usage} tokens\nReasoning: {reasoning}"
            await bot.send_message(
                chat_id=cfg.admins[0],
                reply_parameters=ReplyParameters(chat_id=msg.chat.id, message_id=msg.message_id),
                text=text[:4095],
            )
            # Fail open to avoid false positives.
            return True

    reasoning = response.choices[0].message.reasoning or ""

    if prob >= cfg.llm.threshold:
        # Likely spam, delete and ban, and send the reasoning to the monitor chat for review.
        user_name = (sender.username and "@" + sender.username) or sender.full_name or str(sender.id)
        chat_name = (msg.chat.username and "@" + msg.chat.username) or msg.chat.effective_name or str(msg.chat.id)

        fwd_msg = None
        if cfg.chats.monitor:
            try:
                fwd_msg = await msg.forward(chat_id=cfg.chats.monitor)
            except TelegramError:
                logger.warning("Failed to forward message to monitor chat", exc_info=True)

        no_delete = ""
        no_ban = ""
        if sender.id in cfg.admins or msg.chat.id not in cfg.moderated_chats:
            no_ban = "(simulation, would have been banned)\n"
        else:
            try:
                await msg.delete()
            except TelegramError as ex:
                no_delete = f"Failed to delete: {ex}\n"
            try:
                await msg.chat.ban_member(user_id=sender.id)
            except TelegramError as ex:
                no_ban = f"Failed to ban: {ex}\n"

        if cfg.chats.monitor and fwd_msg:
            text = (
                f"Posted by {user_name} in {chat_name}\n"
                f"Spam probability: {prob:.2f}\n"
                f"{no_delete}{no_ban}"
                f"Reasoning: {reasoning}"
            )
            await bot.send_message(
                chat_id=cfg.chats.monitor,
                reply_to_message_id=fwd_msg.message_id,
                text=text[:4095],
            )
        return False
    else:
        # Not spam, but still send the reasoning to the admins for debugging.
        await bot.send_message(
            chat_id=cfg.admins[0],
            reply_parameters=ReplyParameters(chat_id=msg.chat.id, message_id=msg.message_id),
            text=f"Spam probability: {prob:.2f}\nReasoning: {reasoning}"[:4095],
        )
        return True


async def check_spam(bot: Bot, msg: Message):
    """
    Check the message for spam and react to spam.

    Returns True if the message is likely not spam, False if it is likely spam.
    """
    chat_id = msg.chat.id
    sender = cast(User, msg.from_user)
    text, _ = parse_message(msg)

    # Handle a couple hardcoded cases of spam before even running the LLM, to save costs and reduce false negatives.
    if chat_id in cfg.moderated_chats and text and re.search(cfg.spam_pattern, text, re.I):
        await msg.delete()
        await msg.chat.ban_member(user_id=sender.id)
        return False

    # Short-circuit the LLM check for short messages, as they are unlikely to be spam
    # and it's not worth the cost to check them.
    if len(text) < 20:
        logger.info("Short message, skipping LLM check!")
        return True
    else:
        logger.info("LLM checking message...")
        # For other messages, run the LLM check and then process them in the after function.
        return await llm_check(bot, msg)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = cast(Message, update.message)
        asyncio.create_task(process_message(context.bot, msg))
    except Exception:
        logger.error("Error handling message", exc_info=True)


async def handle_edited_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = cast(Message, update.edited_message)
        asyncio.create_task(process_edit(context.bot, msg))
    except Exception:
        logger.error("Error handling edited message", exc_info=True)


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        qry = cast(CallbackQuery, update.callback_query)
        if not (qry.data or "").startswith("delete:"):
            await qry.answer(text="Invalid button")
            return

        if not qry.message:
            # This should fall back on the DB, but nobody uses it so I don't care.
            await qry.answer(text="Message not found, maybe it's too old?")
            return

        msg_id = qry.message.message_id

        if qry.from_user.id not in cfg.admins:
            # If the user is not an admin, check if they are the author of the message.
            cursor.execute("SELECT author FROM messages WHERE id = ?", [msg_id])
            row = cursor.fetchone()
            author = row[0] if row else None

            if qry.from_user.id != author:
                # If not, check if they are an admin of the source chat.
                member = None
                try:
                    member = await context.bot.get_chat_member(
                        chat_id=cfg.chats.source[0],  # TODO: multiple sources?
                        user_id=qry.from_user.id,
                    )
                except TelegramError:
                    member = None
                if (not member) or member.status not in ("creator", "administrator"):
                    # If not, they have no business deleting this message.
                    await qry.answer(
                        text="You can only delete your own messages. Please ask an admin to delete this.",
                        show_alert=True,
                    )
                    return

        await delete_from_channel(context.bot, msg_id)
        await qry.answer(text="Deleted!")
    except Exception:
        logger.error("Error handling callback query", exc_info=True)


def _watch_and_restart():
    watched = ["janitor.py", "config.toml"]
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
    app = Application.builder().token(cfg.bot_token).build()
    app.add_handler(MessageHandler(filters.UpdateType.MESSAGE, handle_message))
    app.add_handler(MessageHandler(filters.UpdateType.EDITED_MESSAGE, handle_edited_message))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.run_polling()


if __name__ == "__main__":
    main()
