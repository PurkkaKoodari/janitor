#!/usr/bin/env python

import json, logging, sys, re, sqlite3, time, threading, tomllib
from typing import cast, TypedDict, NotRequired, Any, Literal, Callable
import urllib.request

from telegram import telegram
import autorestart

format = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = format)
logger = logging.getLogger("JanitorBot")

autorestart.monitor(__file__)


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

bot: Any = telegram.Bot(token)

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
RECENT = 15 * 60

db = sqlite3.connect(DB_NAME)
cursor = db.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id BIGINT PRIMARY KEY,
    orig_id BIGINT,
    author BIGINT
)
""")

User = TypedDict("User", {
    "id": int,
    "username": str | None,
    "first_name": str,
    "last_name": str | None,
})
Chat = TypedDict("Chat", {
    "id": int,
    "type": Literal["private", "group", "supergroup", "channel"],
    "title": str | None,
    "first_name": str | None,
    "last_name": str | None,
})
MessageOrigin = TypedDict("MessageOrigin", {
    "type": Literal["user", "hidden_user", "channel", "chat"],
})
Message = TypedDict("Message", {
    "message_id": int,
    "from": User,
    "chat": Chat,
    "forward_origin": MessageOrigin | None,
})
ChatMember = TypedDict("ChatMember", {
    "status": Literal["creator", "administrator", "member", "restricted", "left", "kicked"],
    "user": User,
})

def matches(rule: Rule, has_media: bool, text: str) -> bool:
    if (must_have_media := rule.get("media")) != None and has_media != must_have_media:
        return False
    return bool(re.search(rule["regex"], text, re.I))

def make_keyboard(msg_id: int, chat_id: int):
    return {
        "inline_keyboard": [[
            {
                "text": "See message",
                "url": f"https://t.me/c/{-chat_id % 10000000000}/{msg_id}",
            },
            {
                "text": "Delete this",
                "callback_data": "delete:" + str(msg_id),
            },
        ]],
    }

def process_after_llm(msg: Message):
    global recent_senders, recent_dibsers
    msg_id = msg["message_id"]
    chat_id = msg["chat"]["id"]
    sender_id = msg["from"]["id"]
    reply_to_id = msg.get("reply_to_message", {}).get("message_id")

    if chat_id not in SOURCE:
        return

    text, has_media = parse_message(msg)

    if not text and not has_media:
        return

    if next((rule for rule in ignore_rules if matches(rule, has_media, text)), None):
        recent_dibsers[sender_id] = time.time()
        return

    recent_senders = {user: ts for user, ts in recent_senders.items() if ts >= time.time() - RECENT}
    if not has_media and (len(text or "") <= 15) and sender_id not in recent_senders:
        return

    recent_dibsers = {user: ts for user, ts in recent_dibsers.items() if ts >= time.time() - RECENT}
    if not has_media and (len(text or "") <= 20) and sender_id in recent_dibsers:
        return

    recent_senders[sender_id] = time.time()

    new_msg: Message | None = None
    try:
        if not reply_to_id:
            raise ReplyNotFound
        cursor.execute("SELECT id FROM messages WHERE orig_id = ?", [reply_to_id])
        row = cursor.fetchone()
        if not row:
            raise ReplyNotFound

        new_msg = bot.doRequest("copyMessage", {
            "chat_id": CHANNEL,
            "from_chat_id": chat_id,
            "message_id": msg_id,
            "reply_markup": make_keyboard(msg_id, chat_id),
            "reply_to_message_id": row[0],
        })
    except Exception as ex:
        if isinstance(ex, ReplyNotFound) or (isinstance(ex, telegram.ApiError) and "replied message not found" in ex.message):  # type: ignore
            new_msg = bot.doRequest("copyMessage", {
                "chat_id": CHANNEL,
                "from_chat_id": chat_id,
                "message_id": msg_id,
                "reply_markup": make_keyboard(msg_id, chat_id),
            })

    if new_msg:
        cursor.execute("INSERT INTO messages VALUES (?, ?, ?)", [
            new_msg["message_id"], msg_id, sender_id,
        ])
        db.commit()

def process_edit_after_llm(msg: Message):
    msg_id = msg["message_id"]
    chat_id = msg["chat"]["id"]

    if chat_id not in SOURCE:
        return

    cursor.execute("SELECT id FROM messages WHERE orig_id = ?", [msg_id])
    row = cursor.fetchone()
    if not row:
        return

    text, has_media = parse_message(msg)

    if any(rule.get("delete") and matches(rule, has_media, text) for rule in ignore_rules):
        delete_message(CHANNEL, row[0])
        return

    if "text" in msg:
        bot.makeRequest("editMessageText", {
            "chat_id": CHANNEL,
            "message_id": row[0],
            "text": msg["text"],
            "entities": msg.get("entities"),
            "reply_markup": make_keyboard(msg_id, chat_id),
        })
    elif "caption" in msg:
        bot.makeRequest("editMessageCaption", {
            "chat_id": CHANNEL,
            "message_id": row[0],
            "caption": msg["caption"],
            "caption_entities": msg.get("caption_entities"),
            "reply_markup": make_keyboard(msg_id, chat_id),
        })

def llm_check_thread(msg: Message, after: Callable[[Message], None], *, retry: int = 2):
    msg_id = msg["message_id"]
    chat_id = msg["chat"]["id"]
    sender_id = msg["from"]["id"]
    text, _ = parse_message(msg)

    purpose = PURPOSES.get(str(chat_id), PURPOSES[""])
    payload = json.dumps({
        "model": openrouter_model,
        "messages": [
            {"role": "system", "content": system_prompt.format(purpose=purpose)},
            {"role": "user",   "content": text},
        ],
        "max_tokens": 3000,
        "temperature": 0.5,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "spam_prob",
                "strict": True,
                "schema": {
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
            },
        },
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
        method="POST",
    )
    body: Any = None
    usage: str | int = "?"
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode())
        usage = body["usage"]["total_tokens"]
        prob = json.loads(body["choices"][0]["message"]["content"])["prob"]
        logger.info("LLM check complete, %s tokens, spam probability: %.2f", usage, prob)
    except Exception as e:
        try:
            reasoning = body["choices"][0]["message"]["reasoning"]
        except Exception:
            reasoning = ""
        logger.error("LLM check failed, %s tokens, reasoning: %s", usage, reasoning, exc_info=True)
        if retry:
            time.sleep(1)
            llm_check_thread(msg, after, retry=retry-1)
        else:
            bot.makeRequest("sendMessage", {
                "chat_id": ADMINS[0],
                "reply_parameters": {
                    "chat_id": chat_id,
                    "message_id": msg_id,
                },
                "text": f"Openrouter failed with {str(e)}",
            })
            after(msg)
        return

    if prob >= 0.9:
        reasoning = body["choices"][0]["message"]["reasoning"]
        user_name = (
            ((user := msg["from"].get("username")) and "@" + user)
            or f"{msg['from'].get('first_name', '')} {msg['from'].get('last_name', '')}".strip()
            or msg["from"]["id"]
        )
        chat_name = (
            ((user := msg["chat"].get("username")) and "@" + user)
            or msg["chat"].get("title")
            or f"{msg['chat'].get('first_name', '')} {msg['chat'].get('last_name', '')}".strip()
            or msg["chat"]["id"]
        )
        fwd_msg: Message = bot.doRequest("forwardMessage", {
            "chat_id": MONITOR_CHAT,
            "from_chat_id": chat_id,
            "message_id": msg_id,
        })
        no_delete = ""
        no_ban = ""
        if sender_id in ADMINS or chat_id not in [*SOURCE, *MODERATED]:
            no_ban = "Simulated ban in DMs\n"
        else:
            try:
                bot.doRequest("deleteMessage", {
                    "chat_id": chat_id,
                    "message_id": msg_id,
                })
            except telegram.ApiError as ex:
                no_delete = f"Failed to delete: {ex.message}\n"
            try:
                bot.doRequest("banChatMember", {
                    "chat_id": chat_id,
                    "user_id": sender_id,
                })
            except telegram.ApiError as ex:
                no_ban = f"Failed to ban: {ex.message}\n"
        bot.makeRequest("sendMessage", {
            "chat_id": MONITOR_CHAT,
            "reply_parameters": {
                "chat_id": MONITOR_CHAT,
                "message_id": fwd_msg["message_id"],
            },
            "text": f"Posted by {user_name} in {chat_name}\nSpam probability: {prob:.2f}\n{no_delete}{no_ban}Reasoning: {reasoning}"[:4095],
        })
    else:
        reasoning = body["choices"][0]["message"]["reasoning"]
        bot.makeRequest("sendMessage", {
            "chat_id": ADMINS[0],
            "reply_parameters": {
                "chat_id": chat_id,
                "message_id": msg_id,
            },
            "text": f"Spam probability: {prob:.2f}\nReasoning: {reasoning}"[:4095],
        })
        after(msg)

def process_message_or_edit(msg: Message, after: Callable[[Message], None]):
    msg_id = msg["message_id"]
    chat_id = msg["chat"]["id"]
    sender_id = msg["from"]["id"]
    text, _ = parse_message(msg)

    if chat_id in [*SOURCE, *MODERATED] and text and re.search(spam_regex, text, re.I):
        if sender_id not in ADMINS:
            bot.doRequest("deleteMessage", {
                "chat_id": chat_id,
                "message_id": msg_id,
            })
            bot.doRequest("banChatMember", {
                "chat_id": chat_id,
                "user_id": sender_id,
            })
        return

    if len(text) < 20:
        logger.info("Short message, skipping LLM check!")
        after(msg)
        return
    logger.info("LLM checking message...")
    threading.Thread(target=llm_check_thread, args=(msg, after), daemon=True).start()

def process_message(msg: Message):
    msg_id = msg["message_id"]
    chat_id = msg["chat"]["id"]
    sender_id = msg["from"]["id"]

    if (
        "new_chat_members" in msg
        and any(user["id"] == bot.getBotId() for user in msg["new_chat_members"])
    ):
        if (
            chat_id not in [*SOURCE, *MODERATED, CHANNEL]
            and sender_id not in ADMINS
        ):
            bot.makeRequest("leaveChat", {
                "chat_id": chat_id
            })
        else:
            bot.makeRequest("sendMessage", {
                "chat_id": ADMINS[0],
                "text": f"Added to a new chat: {chat_id}",
            })
        return

    text, _ = parse_message(msg)

    if chat_id not in [*SOURCE, *MODERATED, *ADMINS]:
        if msg["chat"]["type"] == "private":
            bot.makeRequest("sendMessage", {
                "chat_id": chat_id,
                "text": "Hello! This bot is for moderating groups. Please talk to @purkka if you need help from me.",
            })
        return

    fwd_from_channel = (orig := msg.get("forward_origin")) and orig["type"] == "channel"
    if chat_id in [*SOURCE, *MODERATED] and fwd_from_channel and (
        msg.get("reply_markup")
        or (fwd_from_channel and text and re.search(r"@\w+|https?://t\.me|https?://vk\.", text) and re.search(r"[\u0400-\u052F]", text, re.I))
    ):
        bot.doRequest("deleteMessage", {
            "chat_id": chat_id,
            "message_id": msg_id,
        })
        bot.doRequest("banChatMember", {
            "chat_id": chat_id,
            "user_id": sender_id,
        })
        return

    process_message_or_edit(msg, process_after_llm)


class ReplyNotFound(Exception):
    pass

def parse_message(msg: Message):
    has_media = "photo" in msg or "document" in msg or "video" in msg or "video_note" in msg
    text = ""
    if "text" in msg:
        text = msg["text"]
    elif "caption" in msg:
        text = msg["caption"]
    return text, has_media


def delete_message(chat_id: int, msg_id: int, qry_id: int | None = None):
    bot.makeRequest("deleteMessage", {
        "chat_id": chat_id,
        "message_id": msg_id,
    })
    cursor.execute("DELETE FROM messages WHERE id = ?", [msg_id])
    db.commit()
    if qry_id:
        bot.makeRequest("answerCallbackQuery", {
            "callback_query_id": qry_id,
            "text": "Deleted!",
        })

def main():
    while True:
        try:
            logger.debug("Getting updates...")
            updates = bot.getUpdates()
            logger.debug("Processing %d message(s)..." % len(updates))
            for update in updates:
                try:
                    if "message" in update:
                        msg = update["message"]
                        process_message(msg)

                    if "edited_message" in update:
                        msg = update["edited_message"]
                        process_message_or_edit(msg, process_edit_after_llm)

                    if "callback_query" in update:
                        qry = update["callback_query"]
                        if not qry.get("data", "").startswith("delete:"):
                            bot.makeRequest("answerCallbackQuery", {
                                "callback_query_id": qry["id"],
                                "text": "Invalid button",
                            })
                            continue

                        msg_id = qry.get("message", {}).get("message_id")

                        cursor.execute("SELECT author FROM messages WHERE id = ?", [msg_id])
                        row = cursor.fetchone()
                        author = row[0] if row else None

                        if qry["from"]["id"] != author and qry["from"]["id"] not in ADMINS:
                            user: ChatMember | None
                            try:
                                user = bot.doRequest("getChatMember", {
                                    "chat_id": SOURCE[0],
                                    "user_id": qry["from"]["id"],
                                })
                            except telegram.ApiError:
                                user = None
                            if (not user) or user["status"] not in ("creator", "administrator"):
                                bot.makeRequest("answerCallbackQuery", {
                                    "callback_query_id": qry["id"],
                                    "text": "You can only delete your own messages. Please ask an admin to delete this.",
                                    "show_alert": True,
                                })
                            else:
                                delete_message(qry["message"]["chat"]["id"], msg_id, qry["id"])
                        else:
                            delete_message(qry["message"]["chat"]["id"], msg_id, qry["id"])
                except Exception:
                    logger.error("Error handling message", exc_info=True)
        except KeyboardInterrupt:
            exit(0)
        except telegram.ApiError:
            logger.error("Error handling message", exc_info=True)

if __name__ == "__main__":
    logger.info("Confirming token...")
    bot.confirmToken()
    logger.info("Token confirmed.")
    # logger.info("Getting current update ID...")
    # bot.clearUpdates()
    # logger.info("Got current update ID.")
    main()
