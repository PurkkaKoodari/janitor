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

import asyncio
from dataclasses import dataclass
from collections.abc import Callable, Coroutine, Sequence
from html import escape
from typing import Any, Literal, cast, get_args

from telegram import (
    Bot,
    BotCommand,
    BotCommandScopeChat,
    CallbackQuery,
    ChatMemberAdministrator,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    Message,
    Update,
    User,
)
from telegram.error import TelegramError
from telegram.ext import Application, ContextTypes

from janitorbot.channel import delete_from_channel
from janitorbot.config import Mode, cfg
from janitorbot.db import (
    add_chat,
    chat_deletes_join_leave,
    chat_mode,
    chat_purpose,
    chat_settings,
    cursor,
    db,
    db_admins,
    db_chats,
    effective_admins,
    effective_moderated_chats,
    purge_old_reasonings,
    remove_chat,
    set_chat_setting,
)
from janitorbot.log import logger


class CommandError(Exception):
    """Aborts a command, replying to the sender with the exception message."""


type CommandCallback = Callable[[Message, User, list[str], Bot], Coroutine[Any, Any, None]]
type CommandHandlerCallback = Callable[
    [Update, ContextTypes.DEFAULT_TYPE], Coroutine[Any, Any, None]
]


def command(
    name: str,
    usage: str = "",
    min_args: int = 0,
    max_args: int | None = None,
    access: Literal["anyone", "admin", "superadmin"] = "admin",
) -> Callable[[CommandCallback], CommandHandlerCallback]:
    """
    Wrap a command implementation into a handler that checks access and argument count,
    replies with `CommandError` messages and logs other errors.
    """

    def decorator(func: CommandCallback) -> CommandHandlerCallback:
        async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            try:
                msg = cast(Message, update.message)
                sender = cast(User, msg.from_user)
                if access == "admin" and sender.id not in effective_admins():
                    return
                if access == "superadmin" and sender.id != cfg.admins[0]:
                    return
                args = context.args or []
                if len(args) < min_args or (max_args is not None and len(args) > max_args):
                    await msg.reply_text(f"Usage: /{name} {usage}")
                    return
                try:
                    await func(msg, sender, args, context.bot)
                except CommandError as e:
                    await msg.reply_text(str(e))
            except Exception:
                logger.error(f"Error handling /{name}", exc_info=True)

        return handler

    return decorator


def parse_chat_id(s: str) -> int:
    """Parse a group/supergroup chat ID (must be a negative integer)."""
    try:
        chat_id = int(s)
    except ValueError:
        chat_id = 0
    if chat_id >= 0:
        raise CommandError("Invalid chat ID.")
    return chat_id


def parse_moderated_chat_id(s: str) -> int:
    """Parse the ID of a chat that is currently moderated."""
    chat_id = parse_chat_id(s)
    if chat_id not in effective_moderated_chats():
        raise CommandError(f"Chat {chat_id} is not a moderated chat.")
    return chat_id


def parse_user_id(s: str) -> int:
    """Parse a user ID."""
    try:
        return int(s)
    except ValueError:
        raise CommandError("Invalid user ID.")


def parse_choice(s: str, choices: Sequence[str], what: str) -> str:
    """Ensure the argument is one of the given choices."""
    if s not in choices:
        raise CommandError(f"Invalid {what}, must be one of: {', '.join(choices)}.")
    return s


async def notify_monitor(bot: Bot, msg: Message, sender: User, html: str) -> None:
    """Report an admin action to the monitor chat, unless the command was sent there."""
    if cfg.chats.monitor and cfg.chats.monitor != msg.chat.id:
        await bot.send_message(
            chat_id=cfg.chats.monitor,
            text=f"{escape(sender.name)} {html}",
            parse_mode="HTML",
        )


@dataclass
class ChatName:
    """
    A chat's ID and its @handle or title (if known), formatted for display in replies and HTML
    messages. The handle is preferred, as Telegram makes it clickable.
    """

    id: int
    username: str | None
    title: str | None

    def __str__(self) -> str:
        if self.username:
            return f"@{self.username} ({self.id})"
        return f"{self.title} ({self.id})" if self.title else str(self.id)

    @property
    def html(self) -> str:
        """The chat name formatted as HTML."""
        if self.username:
            return f"@{escape(self.username)} (<code>{self.id}</code>)"
        if self.title:
            return f"<b>{escape(self.title)}</b> (<code>{self.id}</code>)"
        return f"<code>{self.id}</code>"


async def fetch_chat_name(bot: Bot, chat_id: int) -> ChatName:
    """Fetch the chat's handle and title for display, falling back to just the ID on failure."""
    try:
        chat = await bot.get_chat(chat_id)
    except TelegramError:
        return ChatName(chat_id, None, None)
    return ChatName(chat_id, chat.username, chat.title)


def default_marker(value: object) -> str:
    """Return " (default)" if the chat's setting value is None (not overridden), otherwise ""."""
    return " (default)" if value is None else ""


def describe_purpose(chat_id: int) -> str:
    """Return the chat's purpose for display, marking it if it's not overridden for the chat."""
    return chat_purpose(chat_id) + default_marker(chat_settings(chat_id).purpose)


def describe_mode(chat_id: int) -> str:
    """Return the chat's moderation mode for display, marking it if it's the default."""
    return chat_mode(chat_id) + default_marker(chat_settings(chat_id).mode)


def describe_delete_join_leave(chat_id: int) -> str:
    """Return the chat's join/leave deletion state for display, marking it if it's the default."""
    state = "on" if chat_deletes_join_leave(chat_id) else "off"
    return state + default_marker(chat_settings(chat_id).delete_join_leave)


async def warn_missing_permissions(bot: Bot, msg: Message, chat_id: int, mode: str) -> None:
    """Warn the sender if the bot lacks the admin permissions needed for the moderation mode."""
    try:
        bot_member = await bot.get_chat_member(chat_id, bot.id)
    except TelegramError:
        await msg.reply_text(
            f"Warning: could not check bot permissions in chat {chat_id} "
            "(bot may not be a member). Moderation may not work correctly."
        )
        return
    if bot_member.status not in ("administrator", "creator"):
        await msg.reply_text(
            f"Warning: bot is not an admin in chat {chat_id}. Moderation may not work correctly."
        )
    elif bot_member.status == "administrator" and mode != "test":
        bot_member = cast(ChatMemberAdministrator, bot_member)
        missing: list[str] = []
        if not bot_member.can_delete_messages:
            missing.append("delete messages")
        if mode == "ban" and not bot_member.can_restrict_members:
            missing.append("ban members")
        if missing:
            await msg.reply_text(
                f"Warning: bot is missing permissions in chat {chat_id} for '{mode}' mode: "
                f"{', '.join(missing)}. Moderation may not work correctly."
            )


ADMIN_COMMANDS = [
    BotCommand("moderate", "Enable moderation for a chat"),
    BotCommand("unmoderate", "Remove a chat from moderated chats"),
    BotCommand("purpose", "Show/set the LLM chat purpose for a chat"),
    BotCommand("mode", "Show/set moderation mode for a chat (test/delete/ban)"),
    BotCommand("deletejoins", "Show/set deletion of join/leave messages for a chat"),
]
SUPERADMIN_COMMANDS = ADMIN_COMMANDS + [
    BotCommand("chats", "List all moderated chats"),
    BotCommand("admin", "Add a user as admin"),
    BotCommand("unadmin", "Remove a user from admins"),
]


async def set_admin_commands(bot: Bot, user_id: int) -> None:
    commands = SUPERADMIN_COMMANDS if user_id == cfg.admins[0] else ADMIN_COMMANDS
    try:
        await bot.set_my_commands(commands, scope=BotCommandScopeChat(chat_id=user_id))
    except TelegramError:
        logger.warning("Could not set commands for user %s", user_id, exc_info=True)


async def purge_old_reasonings_loop() -> None:
    """Periodically delete expired LLM reasonings from the database."""
    while True:
        await asyncio.sleep(3600)
        try:
            purge_old_reasonings()
        except Exception:
            logger.error("Failed to purge old reasonings", exc_info=True)


async def post_init(app: Application[Bot, Any, Any, Any, Any, Any]) -> None:
    for user_id in effective_admins():
        await set_admin_commands(app.bot, user_id)
    app.create_task(purge_old_reasonings_loop())


@command("moderate", "<chat_id>", min_args=1, max_args=1)
async def handle_moderate(msg: Message, sender: User, args: list[str], bot: Bot) -> None:
    chat_id = parse_chat_id(args[0])
    await warn_missing_permissions(bot, msg, chat_id, chat_mode(chat_id))
    already_moderated = chat_id in effective_moderated_chats()
    if not already_moderated:
        add_chat(chat_id)
    name = await fetch_chat_name(bot, chat_id)
    status = "is already moderated" if already_moderated else "moderation enabled"
    await msg.reply_text(
        f"Chat {name} {status}.\n"
        f"Mode: {describe_mode(chat_id)} (change with /mode)\n"
        f"Purpose: {describe_purpose(chat_id)} (change with /purpose)\n"
        f"Join/leave message deletion: {describe_delete_join_leave(chat_id)} (change with /deletejoins)"
    )
    if not already_moderated:
        await notify_monitor(bot, msg, sender, f"enabled moderation for chat {name.html}.")


@command("unmoderate", "<chat_id>", min_args=1, max_args=1)
async def handle_unmoderate(msg: Message, sender: User, args: list[str], bot: Bot) -> None:
    chat_id = parse_chat_id(args[0])
    if chat_id in cfg.moderated_chats:
        raise CommandError(
            f"Chat {chat_id} is in the static config and cannot be removed via this command."
        )
    if chat_id not in db_chats:
        raise CommandError(f"Chat {chat_id} is not in the moderated chats database.")
    remove_chat(chat_id)
    name = await fetch_chat_name(bot, chat_id)
    await msg.reply_text(f"Chat {name} removed from moderated chats.")
    await notify_monitor(bot, msg, sender, f"removed chat {name.html} from moderated chats.")


@command("purpose", "<chat_id> [default|purpose...]", min_args=1)
async def handle_purpose(msg: Message, sender: User, args: list[str], bot: Bot) -> None:
    chat_id = parse_moderated_chat_id(args[0])
    if len(args) > 1:
        purpose: str | None = " ".join(args[1:])
        if purpose == "default":
            purpose = None
        elif len(purpose) > 200:
            raise CommandError("Purpose is too long, must be at most 200 characters.")
        set_chat_setting(chat_id, "purpose", purpose)
    name = await fetch_chat_name(bot, chat_id)
    purpose_display = describe_purpose(chat_id)
    llm_text = f'The LLM will be told it\'s in "a chat group where {purpose_display}".'
    if len(args) == 1:
        await msg.reply_text(f"Purpose for chat {name}: {purpose_display}\n{llm_text}")
        return
    await msg.reply_text(f"Purpose for chat {name} updated.\n{llm_text}")
    await notify_monitor(
        bot,
        msg,
        sender,
        f"set the purpose for chat {name.html} to: {escape(purpose_display)}",
    )


@command("mode", "<chat_id> [test|delete|ban|default]", min_args=1, max_args=2)
async def handle_mode(msg: Message, sender: User, args: list[str], bot: Bot) -> None:
    chat_id = parse_moderated_chat_id(args[0])
    name = await fetch_chat_name(bot, chat_id)
    if len(args) == 1:
        await msg.reply_text(f"Moderation mode for chat {name} is {describe_mode(chat_id)}.")
        await warn_missing_permissions(bot, msg, chat_id, chat_mode(chat_id))
        return
    choice = parse_choice(args[1], (*get_args(Mode), "default"), "mode")
    mode = None if choice == "default" else cast(Mode, choice)
    await warn_missing_permissions(bot, msg, chat_id, mode or cfg.spam.mode)
    set_chat_setting(chat_id, "mode", mode)
    await msg.reply_text(f"Moderation mode for chat {name} set to {describe_mode(chat_id)}.")
    await notify_monitor(
        bot,
        msg,
        sender,
        f"set moderation mode to <b>{escape(describe_mode(chat_id))}</b> for chat {name.html}.",
    )


@command("deletejoins", "<chat_id> [on|off|default]", min_args=1, max_args=2)
async def handle_deletejoins(msg: Message, sender: User, args: list[str], bot: Bot) -> None:
    chat_id = parse_moderated_chat_id(args[0])
    name = await fetch_chat_name(bot, chat_id)
    if len(args) == 1:
        await msg.reply_text(
            f"Join/leave message deletion for chat {name} is {describe_delete_join_leave(chat_id)}."
        )
        return
    choice = parse_choice(args[1], ("on", "off", "default"), "value")
    set_chat_setting(chat_id, "delete_join_leave", None if choice == "default" else choice == "on")
    state = describe_delete_join_leave(chat_id)
    await msg.reply_text(f"Join/leave message deletion for chat {name} set to {state}.")
    await notify_monitor(
        bot,
        msg,
        sender,
        f"turned join/leave message deletion <b>{escape(state)}</b> for chat {name.html}.",
    )


@command("admin", "<user_id>", min_args=1, max_args=1, access="superadmin")
async def handle_admin(msg: Message, sender: User, args: list[str], bot: Bot) -> None:
    user_id = parse_user_id(args[0])
    if user_id in cfg.admins:
        raise CommandError(f"User {user_id} is already a static admin.")
    cursor.execute("INSERT OR IGNORE INTO admins (id) VALUES (?)", [user_id])
    db.commit()
    db_admins.add(user_id)
    await set_admin_commands(bot, user_id)
    await msg.reply_text(f"User {user_id} added as admin.")


@command("unadmin", "<user_id>", min_args=1, max_args=1, access="superadmin")
async def handle_unadmin(msg: Message, sender: User, args: list[str], bot: Bot) -> None:
    user_id = parse_user_id(args[0])
    if user_id in cfg.admins:
        raise CommandError(
            f"User {user_id} is a static admin and cannot be removed via this command."
        )
    if user_id not in db_admins:
        raise CommandError(f"User {user_id} is not in the admins database.")
    cursor.execute("DELETE FROM admins WHERE id = ?", [user_id])
    db.commit()
    db_admins.discard(user_id)
    try:
        await bot.delete_my_commands(scope=BotCommandScopeChat(chat_id=user_id))
    except TelegramError:
        logger.warning("Could not clear commands for user %s", user_id, exc_info=True)
    await msg.reply_text(f"User {user_id} removed from admins.")


CHATS_PAGE_SIZE = 15


def build_chats_page(page: int) -> tuple[str, InlineKeyboardMarkup | None]:
    all_chats = sorted(effective_moderated_chats())
    total = len(all_chats)
    total_pages = max(1, (total + CHATS_PAGE_SIZE - 1) // CHATS_PAGE_SIZE)
    page = max(1, min(page, total_pages))

    start = (page - 1) * CHATS_PAGE_SIZE
    chats_page = all_chats[start : start + CHATS_PAGE_SIZE]

    lines = [f"Moderated chats (page {page}/{total_pages}):"]
    for chat_id in chats_page:
        purpose = chat_purpose(chat_id)
        mode = chat_mode(chat_id)
        lines.append(f"{chat_id}: {mode} — {purpose}")

    buttons: list[InlineKeyboardButton] = []
    if page > 1:
        buttons.append(InlineKeyboardButton("◄ Previous", callback_data=f"chats:{page - 1}"))
    if page < total_pages:
        buttons.append(InlineKeyboardButton("Next ►", callback_data=f"chats:{page + 1}"))

    keyboard = InlineKeyboardMarkup([buttons]) if buttons else None
    return "\n".join(lines), keyboard


@command("chats", access="superadmin")
async def handle_chats(msg: Message, sender: User, args: list[str], bot: Bot) -> None:
    text, keyboard = build_chats_page(1)
    await msg.reply_text(text, reply_markup=keyboard)


@command("hello", access="anyone")
async def handle_hello(msg: Message, sender: User, args: list[str], bot: Bot) -> None:
    await bot.send_message(
        chat_id=cfg.admins[0],
        text=f"User sent /hello: <code>{sender.id}</code>",
        parse_mode="HTML",
    )
    await msg.set_reaction("👍")


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        qry = cast(CallbackQuery, update.callback_query)
        data = qry.data or ""

        if data.startswith("chats:"):
            if qry.from_user.id != cfg.admins[0]:
                await qry.answer()
                return
            try:
                page = int(data.split(":", 1)[1])
            except (ValueError, IndexError):
                await qry.answer(text="Invalid page")
                return
            text, keyboard = build_chats_page(page)
            await qry.edit_message_text(text, reply_markup=keyboard)
            await qry.answer()
            return

        if not data.startswith("delete:"):
            await qry.answer(text="Invalid button")
            return

        if not qry.message:
            # This should fall back on the DB, but nobody uses it so I don't care.
            await qry.answer(text="Message not found, maybe it's too old?")
            return

        msg_id = qry.message.message_id

        if qry.from_user.id not in effective_admins():
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
