#!/usr/bin/env python

import asyncio, json, logging, re, sqlite3, sys, time, tomllib
from typing import cast, TypedDict, NotRequired, Callable, Coroutine, TypeAlias

from telegram import Bot, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, Message, ReplyParameters, Update, User
from telegram.error import TelegramError, BadRequest
from telegram.ext import Application, MessageHandler, CallbackQueryHandler, ContextTypes, filters

import openrouter
from openrouter.components import ResponseFormatJSONSchema, JSONSchemaConfig

format = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=format)
logger = logging.getLogger("JanitorBot")


AfterFunc: TypeAlias = Callable[[Bot, Message], Coroutine[None, None, None]]


class Rule(TypedDict):
    regex: str
    # None means "don't care", True means "must have media", False means "must not have media"
    media: NotRequired[bool]
    # if True, messages matching this rule will be deleted if they are edited to match, even if they previously didn't match
    delete: NotRequired[bool]

class Config(TypedDict):
    token: str
    db_name: str
    source: list[int]
    moderated: list[int]
    admins: list[int]
    channel: int
    monitor_chat: int
    openrouter_key: str
    openrouter_model: str
    system_prompt: str
    spam_regex: list[str]
    purposes: dict[str, str]
    ignore_rules: list[Rule]

with open("config.toml", "rb") as file:
    cfg = cast(Config, tomllib.load(file))

token: str = cfg["token"]
assert token != ""

DB_NAME = cfg["db_name"]

SOURCE = cfg["source"]
CHANNEL = cfg["channel"]
MODERATED = cfg["moderated"]
ADMINS = cfg["admins"]
MONITOR_CHAT = cfg["monitor_chat"]

PURPOSES = cfg["purposes"]

try_ignore_rules = cfg["ignore_rules"]

spam_regexes = cfg["spam_regex"]
spam_regex = "|".join(spam_regexes) or "a^"

system_prompt = cfg["system_prompt"]

openrouter_key = cfg["openrouter_key"]
openrouter_model = cfg["openrouter_model"]

ignore_rules: list[Rule] = []

for rule in try_ignore_rules:
    try:
        re.compile(rule["regex"], re.I)
    except Exception as e:
        logger.error("Ignoring invalid regex %s: %s", rule["regex"], e)
    else:
        ignore_rules.append(rule)

recent_senders: dict[int, float] = {}
recent_dibsers: dict[int, float] = {}
RECENT = 15 * 60  # seconds


db = sqlite3.connect(DB_NAME)
cursor = db.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id BIGINT PRIMARY KEY,
    orig_id BIGINT,
    author BIGINT
)
""")

or_client = openrouter.OpenRouter(api_key=openrouter_key)


class ReplyNotFound(Exception):
    pass


def matches(rule: Rule, has_media: bool, text: str) -> bool:
    if (must_have_media := rule.get("media")) != None and has_media != must_have_media:
        return False
    return bool(re.search(rule["regex"], text, re.I))


def make_keyboard(msg: Message) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("See message", url=msg.link),
        InlineKeyboardButton("Delete this", callback_data="delete:" + str(msg.message_id)),
    ]])


def parse_message(msg: Message) -> tuple[str, bool]:
    has_media = bool(msg.photo) or msg.document is not None or msg.video is not None or msg.video_note is not None
    text = msg.text or msg.caption or ""
    return text, has_media


async def delete_from_channel(bot: Bot, chat_id: int, msg_id: int, qry_id: str | None = None):
    await bot.delete_message(chat_id=chat_id, message_id=msg_id)
    cursor.execute("DELETE FROM messages WHERE id = ?", [msg_id])
    db.commit()
    if qry_id:
        await bot.answer_callback_query(callback_query_id=qry_id, text="Deleted!")


async def process_message(bot: Bot, msg: Message):
    """
    After the LLM check, filter messages from the SOURCE chats and forward them to the CHANNEL if they pass.
    """
    global recent_senders, recent_dibsers

    sender = cast(User, msg.from_user)

    # If the bot was just added to a new chat, check if it's a chat we want to be in.
    if (
        msg.new_chat_members
        and any(user.id == bot.id for user in msg.new_chat_members)
    ):
        # If the chat is not known and the person who added the bot is not an admin, leave the chat.
        if (
            msg.chat.id not in [*SOURCE, *MODERATED, CHANNEL]
            and sender.id not in ADMINS
        ):
            await msg.chat.leave()
        else:
            # Otherwise, send the chat id to the admins so they can add it to the config.
            await bot.send_message(chat_id=ADMINS[0], text=f"Added to a new chat \"{msg.chat.effective_name}\" ({msg.chat.id})")
        return

    if msg.chat.id not in [*SOURCE, *MODERATED, *ADMINS]:
        # If the source chat is a private chat, reply with instructions.
        if msg.chat.type == "private":
            await msg.reply_text(
                text="Hello! This bot is for moderating groups. Please talk to @purkka if you need help from me.",
            )
        # Ignore messages from unknown chats.
        return

    text, has_media = parse_message(msg)

    # Check for forwarded special cases of spam, and react to them. If the message is likely spam, don't process it further.
    fwd_from_channel = msg.forward_origin and msg.forward_origin.type == "channel"
    if msg.chat.id in [*SOURCE, *MODERATED] and fwd_from_channel and (
        # Forwarded from channel with inline keyboard
        msg.reply_markup
        # Forwarded from channel, contains a link or mention, and contains Cyrillic
        or (fwd_from_channel and text and re.search(r"@\w+|https?://t\.me|https?://vk\.", text) and re.search(r"[\u0400-\u052F]", text, re.I))
    ):
        await msg.delete()
        await msg.chat.ban_member(user_id=sender.id)
        return

    # Check for spam and react to it. If the message is likely spam, don't process it further.
    if not await check_spam(bot, msg):
        return

    # Ignore messages outside SOURCE chats.
    if msg.chat_id not in SOURCE:
        return

    # Ignore messages that have no text and no media, as they are unlikely to be useful.
    if not text and not has_media:
        return

    # Ignore messages that match ignore rules, and mark the user as a dibser.
    if next((rule for rule in ignore_rules if matches(rule, has_media, text)), None):
        recent_dibsers[sender.id] = time.time()
        return

    # Ignore short messages that have no media from users who haven't sent
    # anything useful in a while, as they are unlikely to be useful either.
    recent_senders = {user: ts for user, ts in recent_senders.items() if ts >= time.time() - RECENT}
    if not has_media and (len(text or "") <= 15) and sender.id not in recent_senders:
        return

    # Ignore short messages that have no media from recent dibsers,
    # as they are likely to be follow-ups to the original dibs message.
    recent_dibsers = {user: ts for user, ts in recent_dibsers.items() if ts >= time.time() - RECENT}
    if not has_media and (len(text or "") <= 20) and sender.id in recent_dibsers:
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
            chat_id=CHANNEL,
            reply_markup=make_keyboard(msg),
            reply_to_message_id=row[0],
        )
        new_msg_id = result.message_id
    except Exception as ex:
        # If the message is not a reply, or if we can't find the replied message in the channel, just post it without replying.
        if isinstance(ex, ReplyNotFound) or (isinstance(ex, BadRequest) and "replied message not found" in str(ex).lower()):
            result = await msg.copy(
                chat_id=CHANNEL,
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
    # Ignore edits outside SOURCE, MODERATED, and ADMINS chats.
    if msg.chat.id not in [*SOURCE, *MODERATED, *ADMINS]:
        return

    # Check for spam and react to it. If the message is likely spam, don't process it further.
    if not await check_spam(bot, msg):
        return

    # Ignore edits outside SOURCE chats.
    if msg.chat.id not in SOURCE:
        return

    # Try to find the corresponding message in the channel.
    cursor.execute("SELECT id FROM messages WHERE orig_id = ?", [msg.message_id])
    row = cursor.fetchone()
    if not row:
        return

    text, has_media = parse_message(msg)

    # If the message now matches an ignore rule with delete=True, delete it from the channel.
    if any(rule.get("delete") and matches(rule, has_media, text) for rule in ignore_rules):
        await delete_from_channel(bot, CHANNEL, row[0])
        return

    # Otherwise, update the message in the channel to reflect the edit.
    if msg.text is not None:
        await bot.edit_message_text(
            chat_id=CHANNEL,
            message_id=row[0],
            text=msg.text,
            entities=msg.entities,
            reply_markup=make_keyboard(msg),
        )
    elif msg.caption is not None:
        await bot.edit_message_caption(
            chat_id=CHANNEL,
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

    purpose = PURPOSES.get(str(msg.chat.id), PURPOSES[""])

    response = None
    usage: str | int = "?"
    try:
        response = await or_client.chat.send_async(
            model=openrouter_model,
            messages=[
                {"role": "system", "content": system_prompt.format(purpose=purpose)},
                {"role": "user",   "content": text},
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
        logger.error("LLM check failed, %s tokens, reasoning: %s", usage, reasoning, exc_info=True)
        if retry:
            await asyncio.sleep(1)
            return await llm_check(bot, msg, retry=retry - 1)
        else:
            await bot.send_message(
                chat_id=ADMINS[0],
                reply_parameters=ReplyParameters(chat_id=msg.chat.id, message_id=msg.message_id),
                text=f"Openrouter failed with {str(e)} after {usage} tokens\nReasoning: {reasoning}"[:4095],
            )
            # Fail open to avoid false positives.
            return True

    reasoning = response.choices[0].message.reasoning or ""

    if prob >= 0.9:
        # Likely spam, delete and ban, and send the reasoning to the monitor chat for review.
        user_name = (
            (sender.username and "@" + sender.username)
            or sender.full_name
            or str(sender.id)
        )
        chat_name = (
            (msg.chat.username and "@" + msg.chat.username)
            or msg.chat.effective_name
            or str(msg.chat.id)
        )

        fwd_msg = await msg.forward(chat_id=MONITOR_CHAT)

        no_delete = ""
        no_ban = ""
        if sender.id in ADMINS or msg.chat.id not in [*SOURCE, *MODERATED]:
            no_ban = "(simulated ban in DMs)\n"
        else:
            try:
                await msg.delete()
            except TelegramError as ex:
                no_delete = f"Failed to delete: {ex}\n"
            try:
                await msg.chat.ban_member(user_id=sender.id)
            except TelegramError as ex:
                no_ban = f"Failed to ban: {ex}\n"

        await bot.send_message(
            chat_id=MONITOR_CHAT,
            reply_to_message_id=fwd_msg.message_id,
            text=f"Posted by {user_name} in {chat_name}\nSpam probability: {prob:.2f}\n{no_delete}{no_ban}Reasoning: {reasoning}"[:4095],
        )
        return False
    else:
        # Not spam, but still send the reasoning to the admins for debugging.
        await bot.send_message(
            chat_id=ADMINS[0],
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
    if chat_id in [*SOURCE, *MODERATED] and text and re.search(spam_regex, text, re.I):
        if sender.id not in ADMINS:
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
            await qry.answer(text="Message not found, maybe it's too old?")
            return

        msg_id = qry.message.message_id

        if qry.from_user.id not in ADMINS:
            cursor.execute("SELECT author FROM messages WHERE id = ?", [msg_id])
            row = cursor.fetchone()
            author = row[0] if row else None

            if qry.from_user.id != author:
                member = None
                try:
                    member = await context.bot.get_chat_member(
                        chat_id=SOURCE[0],
                        user_id=qry.from_user.id,
                    )
                except TelegramError:
                    member = None
                if (not member) or member.status not in ("creator", "administrator"):
                    await qry.answer(
                        text="You can only delete your own messages. Please ask an admin to delete this.",
                        show_alert=True,
                    )
                    return

        await delete_from_channel(context.bot, CHANNEL, msg_id, qry.id)
        await qry.answer()
    except Exception:
        logger.error("Error handling callback query", exc_info=True)


def main():
    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.UpdateType.MESSAGE, handle_message))
    app.add_handler(MessageHandler(filters.UpdateType.EDITED_MESSAGE, handle_edited_message))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.run_polling()


if __name__ == "__main__":
    main()
