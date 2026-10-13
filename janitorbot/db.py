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

import hashlib, sqlite3, time
from dataclasses import dataclass
from typing import Literal

from janitorbot.config import Mode, cfg

db = sqlite3.connect(cfg.db_name)
cursor = db.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id BIGINT PRIMARY KEY,
        orig_id BIGINT,
        author BIGINT
    )
    """)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS spam_messages (
        chat_id BIGINT NOT NULL,
        message_id BIGINT NOT NULL,
        user_id BIGINT NOT NULL,
        timestamp BIGINT NOT NULL,
        system BOOLEAN NOT NULL,
        spam BOOLEAN NOT NULL,
        mentions TEXT,
        system_prompt_hash BLOB,
        content_hash BLOB,
        llm_sample INTEGER,
        llm_prob REAL,
        llm_reasoning TEXT
    )
    """)
cursor.execute("""
    CREATE INDEX IF NOT EXISTS spam_messages_hashes
    ON spam_messages (content_hash, system_prompt_hash)
    """)
cursor.execute("""
    CREATE INDEX IF NOT EXISTS spam_messages_user
    ON spam_messages (user_id, timestamp)
    """)
cursor.execute("""
    CREATE INDEX IF NOT EXISTS spam_messages_user_chat
    ON spam_messages (user_id, chat_id)
    """)
cursor.execute("""
    CREATE INDEX IF NOT EXISTS spam_messages_reasoning_timestamp
    ON spam_messages (timestamp) WHERE llm_reasoning IS NOT NULL
    """)
_CHATS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS {table} (
        id BIGINT PRIMARY KEY,
        purpose TEXT,
        mode TEXT,
        delete_join_leave BOOLEAN
    )
    """
cursor.execute(_CHATS_SCHEMA.format(table="chats"))
# Migrate old chats tables: add delete_join_leave and allow NULL (= config default) mode.
cursor.execute("PRAGMA table_info(chats)")
_chats_columns = {_row[1]: _row for _row in cursor.fetchall()}
if "delete_join_leave" not in _chats_columns or _chats_columns["mode"][3]:
    _copied = "id, purpose, mode" + (
        ", delete_join_leave" if "delete_join_leave" in _chats_columns else ""
    )
    cursor.execute(_CHATS_SCHEMA.format(table="chats_new"))
    cursor.execute(f"INSERT INTO chats_new ({_copied}) SELECT {_copied} FROM chats")
    cursor.execute("DROP TABLE chats")
    cursor.execute("ALTER TABLE chats_new RENAME TO chats")
    db.commit()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS admins (
        id BIGINT PRIMARY KEY
    )
    """)


@dataclass
class ChatSettings:
    """Per-chat settings stored in the chats table. None means the config default."""

    purpose: str | None = None
    mode: Mode | None = None
    delete_join_leave: bool | None = None


db_chats: dict[int, ChatSettings] = {}
cursor.execute("SELECT id, purpose, mode, delete_join_leave FROM chats")
for _row in cursor.fetchall():
    db_chats[_row[0]] = ChatSettings(
        purpose=_row[1],
        mode=_row[2],
        delete_join_leave=None if _row[3] is None else bool(_row[3]),
    )

db_admins: set[int] = set()
cursor.execute("SELECT id FROM admins")
for _row in cursor.fetchall():
    db_admins.add(_row[0])


def effective_admins() -> frozenset[int]:
    return frozenset(cfg.admins) | db_admins


def effective_moderated_chats() -> frozenset[int]:
    return cfg.moderated_chats | frozenset(db_chats.keys())


def effective_spam_checked_chats() -> frozenset[int]:
    return effective_moderated_chats() | effective_admins()


def effective_member_chats() -> frozenset[int]:
    return effective_moderated_chats() | {cfg.chats.channel}


def chat_settings(chat_id: int) -> ChatSettings:
    """Return the chat's setting overrides from the database (all None if there are none)."""
    return db_chats.get(chat_id) or ChatSettings()


def chat_purpose(chat_id: int) -> str:
    """Return the chat's purpose for the LLM, falling back to the config."""
    return chat_settings(chat_id).purpose or cfg.llm.chat_purposes.get(
        str(chat_id), cfg.llm.chat_purposes["default"]
    )


def chat_mode(chat_id: int) -> Mode:
    """Return the chat's moderation mode, falling back to the config."""
    return chat_settings(chat_id).mode or cfg.spam.mode


def chat_deletes_join_leave(chat_id: int) -> bool:
    """Return whether join/leave system messages are deleted in the chat."""
    delete_join_leave = chat_settings(chat_id).delete_join_leave
    return cfg.chats.delete_join_leave if delete_join_leave is None else delete_join_leave


def add_chat(chat_id: int) -> None:
    """Add a chat to the chats table with default settings, if not already present."""
    cursor.execute("INSERT OR IGNORE INTO chats (id) VALUES (?)", [chat_id])
    db.commit()
    db_chats.setdefault(chat_id, ChatSettings())


def remove_chat(chat_id: int) -> None:
    """Remove a chat and its settings from the chats table."""
    cursor.execute("DELETE FROM chats WHERE id = ?", [chat_id])
    db.commit()
    db_chats.pop(chat_id, None)


def set_chat_setting(
    chat_id: int, setting: Literal["purpose", "mode", "delete_join_leave"], value: object
) -> None:
    """
    Set a setting (None for the config default) for a moderated chat. Chats moderated via the
    config get a row on their first override.
    """
    if chat_id not in effective_moderated_chats():
        raise ValueError(f"Chat {chat_id} is not moderated")
    cursor.execute(
        f"""
        INSERT INTO chats (id, {setting})
        VALUES (?, ?)
        ON CONFLICT(id) DO UPDATE
        SET {setting} = excluded.{setting}
        """,
        [chat_id, value],
    )
    db.commit()
    setattr(db_chats.setdefault(chat_id, ChatSettings()), setting, value)


def system_prompt_hash(system_prompt: str) -> bytes:
    """Hash the model and system prompt into a 64-bit verdict cache key component."""
    return hashlib.blake2b(f"{cfg.llm.model}\n{system_prompt}".encode(), digest_size=8).digest()


def content_hash(llm_user_message: str) -> bytes:
    """Hash the user message fed to the LLM into a 128-bit key."""
    return hashlib.blake2b(llm_user_message.encode(), digest_size=16).digest()


@dataclass
class MessageRecord:
    """Information about a message to store in the spam_messages table."""

    chat_id: int
    message_id: int
    user_id: int
    timestamp: int
    system: bool
    spam: bool
    mentions: list[str]
    system_prompt_hash: bytes | None = None
    content_hash: bytes | None = None
    llm_samples: list[tuple[float, str | None]] | None = None


def record_message(record: MessageRecord) -> None:
    """Store the message, with one row per LLM sample. Edits add new rows."""
    samples: list[tuple[int | None, tuple[float | None, str | None]]] = list(
        enumerate(record.llm_samples or [])
    ) or [(None, (None, None))]
    cursor.executemany(
        "INSERT INTO spam_messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            [
                record.chat_id,
                record.message_id,
                record.user_id,
                record.timestamp,
                record.system,
                record.spam,
                " ".join(record.mentions) or None,
                record.system_prompt_hash,
                record.content_hash,
                index,
                prob,
                reasoning,
            ]
            for index, (prob, reasoning) in samples
        ],
    )
    db.commit()


def cached_verdict(sp_hash: bytes, c_hash: bytes) -> list[float]:
    """
    Return all recorded LLM spam probabilities for the same LLM user message and system prompt,
    sorted ascending.
    """
    cursor.execute(
        """
        SELECT llm_prob FROM spam_messages
        WHERE content_hash = ? AND system_prompt_hash = ? AND llm_prob IS NOT NULL
        ORDER BY llm_prob
        """,
        [c_hash, sp_hash],
    )
    return [row[0] for row in cursor.fetchall()]


def is_established_user(user_id: int) -> bool:
    """
    Return whether the user has a recorded message older than ``min_days`` and at least
    ``min_nonspam_messages`` recorded messages not deemed spam, across all known chats.
    """
    cursor.execute(
        """
        SELECT MIN(first), COALESCE(SUM(NOT spam), 0) FROM (
            SELECT MIN(timestamp) AS first, MAX(spam) AS spam FROM spam_messages
            WHERE user_id = ?
            GROUP BY chat_id, message_id
        )
        """,
        [user_id],
    )
    first, nonspam = cursor.fetchone()
    cutoff = time.time() - cfg.established_user.min_days * 86400
    return (
        first is not None
        and first <= cutoff
        and nonspam >= cfg.established_user.min_nonspam_messages
    )


def recent_spam_count(user_id: int) -> int:
    """Return the number of spam messages the user has sent within ``ban_threshold_days``."""
    cursor.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT 1 FROM spam_messages
            WHERE user_id = ? AND timestamp >= ?
            GROUP BY chat_id, message_id
            HAVING MAX(spam)
        )
        """,
        [user_id, int(time.time() - cfg.established_user.ban_threshold_days * 86400)],
    )
    return cursor.fetchone()[0]


def join_message_ids(chat_id: int, user_id: int) -> list[int]:
    """Return the IDs of the recorded join messages of the user in the chat."""
    cursor.execute(
        """
        SELECT DISTINCT message_id FROM spam_messages
        WHERE user_id = ? AND chat_id = ? AND system
        """,
        [user_id, chat_id],
    )
    return [row[0] for row in cursor.fetchall()]


def mentioning_message_ids(chat_id: int, user_id: int, mention: str) -> list[int]:
    """
    Return the IDs of the recorded non-system messages of the user in the chat that mention the
    given lowercase username.
    """
    cursor.execute(
        """
        SELECT DISTINCT message_id FROM spam_messages
        WHERE user_id = ? AND chat_id = ? AND NOT system
        AND instr(' ' || mentions || ' ', ?) > 0
        """,
        [user_id, chat_id, f" {mention} "],
    )
    return [row[0] for row in cursor.fetchall()]


def purge_old_reasonings() -> None:
    """Delete LLM reasonings older than the retention period, as they often contain the content."""
    cutoff = int(time.time()) - cfg.llm.reasoning_retention_days * 86400
    cursor.execute(
        "UPDATE spam_messages SET llm_reasoning = NULL WHERE llm_reasoning IS NOT NULL AND timestamp < ?",
        [cutoff],
    )
    db.commit()


purge_old_reasonings()
