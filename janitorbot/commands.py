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

from html import escape
from typing import Any, cast

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
from janitorbot.config import cfg
from janitorbot.db import (
    chat_mode,
    chat_purpose,
    cursor,
    db,
    db_admins,
    db_chat_mode,
    db_chat_purposes,
    effective_admins,
    effective_moderated_chats,
)
from janitorbot.log import logger


def parse_chat_id(s: str) -> int | None:
    """Parse a group/supergroup chat ID (must be a negative integer)."""
    try:
        chat_id = int(s)
        if chat_id >= 0:
            return None
        return chat_id
    except ValueError:
        return None


ADMIN_COMMANDS = [
    BotCommand("moderate", "Enable/update moderation for a chat"),
    BotCommand("unmoderate", "Remove a chat from moderated chats"),
    BotCommand("mode", "Set moderation mode for a chat (test/delete/ban)"),
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


async def post_init(app: Application[Bot, Any, Any, Any, Any, Any]) -> None:
    for user_id in effective_admins():
        await set_admin_commands(app.bot, user_id)


async def handle_moderate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = cast(Message, update.message)
        sender = cast(User, msg.from_user)
        if sender.id not in effective_admins():
            return
        args = context.args or []
        if len(args) < 1:
            await msg.reply_text("Usage: /moderate <chat_id> [purpose...]")
            return
        chat_id = parse_chat_id(args[0])
        if chat_id is None:
            await msg.reply_text("Invalid chat ID.")
            return
        try:
            bot_member = await context.bot.get_chat_member(chat_id, context.bot.id)
            if bot_member.status not in ("administrator", "creator"):
                await msg.reply_text(
                    f"Warning: bot is not an admin in chat {chat_id}. "
                    "Moderation may not work correctly."
                )
            elif bot_member.status == "administrator":
                bot_member = cast(ChatMemberAdministrator, bot_member)
                missing: list[str] = []
                if not bot_member.can_delete_messages:
                    missing.append("delete messages")
                if not bot_member.can_restrict_members:
                    missing.append("ban members")
                if missing:
                    await msg.reply_text(
                        f"Warning: bot is missing permissions in chat {chat_id}: {', '.join(missing)}. "
                        "Moderation may not work correctly."
                    )
        except Exception:
            await msg.reply_text(
                f"Warning: could not check bot permissions in chat {chat_id} "
                "(bot may not be a member). Moderation may not work correctly."
            )
        purpose: str | None = " ".join(args[1:]) if args[1:] else None
        if purpose and len(purpose) > 200:
            await msg.reply_text("Purpose is too long, must be at most 200 characters.")
            return
        cursor.execute(
            """
            INSERT INTO chats (id, purpose)
            VALUES (?, ?)
            ON CONFLICT(id) DO UPDATE
            SET purpose = excluded.purpose
            """,
            [chat_id, purpose],
        )
        db.commit()
        db_chat_purposes[chat_id] = purpose
        purpose_display = purpose or chat_purpose(chat_id) + " (default)"
        await msg.reply_text(
            f'Chat {chat_id} moderation enabled/updated.\nThe LLM will be told it\'s in "a chat group where {purpose_display}".'
        )
        if cfg.chats.monitor and cfg.chats.monitor != msg.chat.id:
            sender_name = (
                (sender.username and "@" + sender.username) or sender.full_name or str(sender.id)
            )
            await context.bot.send_message(
                chat_id=cfg.chats.monitor,
                text=f"{escape(sender_name)} enabled/updated moderation for chat <code>{chat_id}</code>.\nNew purpose: {escape(purpose_display)}",
                parse_mode="HTML",
            )
    except Exception:
        logger.error("Error handling /moderate", exc_info=True)


async def handle_unmoderate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = cast(Message, update.message)
        sender = cast(User, msg.from_user)
        if sender.id not in effective_admins():
            return
        args = context.args or []
        if len(args) != 1:
            await msg.reply_text("Usage: /unmoderate <chat_id>")
            return
        chat_id = parse_chat_id(args[0])
        if chat_id is None:
            await msg.reply_text("Invalid chat ID.")
            return
        if chat_id in cfg.moderated_chats:
            await msg.reply_text(
                f"Chat {chat_id} is in the static config and cannot be removed via this command. Use /moderate to revert it to the default purpose instead."
            )
            return
        if chat_id not in db_chat_purposes:
            await msg.reply_text(f"Chat {chat_id} is not in the moderated chats database.")
            return
        cursor.execute("DELETE FROM chats WHERE id = ?", [chat_id])
        db.commit()
        del db_chat_purposes[chat_id]
        await msg.reply_text(f"Chat {chat_id} removed from moderated chats.")
        if cfg.chats.monitor and cfg.chats.monitor != msg.chat.id:
            sender_name = (
                (sender.username and "@" + sender.username) or sender.full_name or str(sender.id)
            )
            await context.bot.send_message(
                chat_id=cfg.chats.monitor,
                text=f"{escape(sender_name)} removed chat <code>{chat_id}</code> from moderated chats.",
                parse_mode="HTML",
            )
    except Exception:
        logger.error("Error handling /unmoderate", exc_info=True)


async def handle_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = cast(Message, update.message)
        sender = cast(User, msg.from_user)
        if sender.id not in effective_admins():
            return
        args = context.args or []
        if len(args) != 2:
            await msg.reply_text("Usage: /mode <chat_id> <test|delete|ban>")
            return
        chat_id = parse_chat_id(args[0])
        if chat_id is None:
            await msg.reply_text("Invalid chat ID.")
            return
        if args[1] not in ("test", "delete", "ban"):
            await msg.reply_text("Invalid mode, must be 'test', 'delete', or 'ban'.")
            return
        mode = args[1]
        if chat_id not in effective_moderated_chats():
            await msg.reply_text(f"Chat {chat_id} is not a moderated chat.")
            return
        cursor.execute(
            """
            INSERT INTO chats (id, purpose, mode)
            VALUES (?, NULL, ?)
            ON CONFLICT(id) DO UPDATE
            SET mode = excluded.mode
            """,
            [chat_id, mode],
        )
        db.commit()
        db_chat_mode[chat_id] = mode
        await msg.reply_text(f"Moderation mode for chat {chat_id} set to '{mode}'.")
        if cfg.chats.monitor and cfg.chats.monitor != msg.chat.id:
            sender_name = (
                (sender.username and "@" + sender.username) or sender.full_name or str(sender.id)
            )
            await context.bot.send_message(
                chat_id=cfg.chats.monitor,
                text=f"{escape(sender_name)} set moderation mode to <b>{mode}</b> for chat <code>{chat_id}</code>.",
                parse_mode="HTML",
            )
    except Exception:
        logger.error("Error handling /mode", exc_info=True)


async def handle_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = cast(Message, update.message)
        sender = cast(User, msg.from_user)
        if sender.id != cfg.admins[0]:
            return
        args = context.args or []
        if len(args) != 1:
            await msg.reply_text("Usage: /admin <user_id>")
            return
        try:
            user_id = int(args[0])
        except ValueError:
            await msg.reply_text("Invalid user ID.")
            return
        if user_id in cfg.admins:
            await msg.reply_text(f"User {user_id} is already a static admin.")
            return
        cursor.execute("INSERT OR IGNORE INTO admins (id) VALUES (?)", [user_id])
        db.commit()
        db_admins.add(user_id)
        await set_admin_commands(context.bot, user_id)
        await msg.reply_text(f"User {user_id} added as admin.")
    except Exception:
        logger.error("Error handling /admin", exc_info=True)


async def handle_unadmin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = cast(Message, update.message)
        sender = cast(User, msg.from_user)
        if sender.id != cfg.admins[0]:
            return
        args = context.args or []
        if len(args) != 1:
            await msg.reply_text("Usage: /unadmin <user_id>")
            return
        try:
            user_id = int(args[0])
        except ValueError:
            await msg.reply_text("Invalid user ID.")
            return
        if user_id in cfg.admins:
            await msg.reply_text(
                f"User {user_id} is a static admin and cannot be removed via this command."
            )
            return
        if user_id not in db_admins:
            await msg.reply_text(f"User {user_id} is not in the admins database.")
            return
        cursor.execute("DELETE FROM admins WHERE id = ?", [user_id])
        db.commit()
        db_admins.discard(user_id)
        try:
            await context.bot.delete_my_commands(scope=BotCommandScopeChat(chat_id=user_id))
        except TelegramError:
            logger.warning("Could not clear commands for user %s", user_id, exc_info=True)
        await msg.reply_text(f"User {user_id} removed from admins.")
    except Exception:
        logger.error("Error handling /unadmin", exc_info=True)


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


async def handle_chats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = cast(Message, update.message)
        sender = cast(User, msg.from_user)
        if sender.id != cfg.admins[0]:
            return
        text, keyboard = build_chats_page(1)
        await msg.reply_text(text, reply_markup=keyboard)
    except Exception:
        logger.error("Error handling /chats", exc_info=True)


async def handle_hello(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = cast(Message, update.message)
        sender = cast(User, msg.from_user)
        await context.bot.send_message(
            chat_id=cfg.admins[0],
            text=f"User sent /hello: <code>{sender.id}</code>",
            parse_mode="HTML",
        )
        await msg.set_reaction("👍")
    except Exception:
        logger.error("Error handling /hello", exc_info=True)


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
