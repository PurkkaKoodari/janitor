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

from janitorbot.config import cfg

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
    CREATE INDEX IF NOT EXISTS spam_messages_reasoning_timestamp
    ON spam_messages (timestamp) WHERE llm_reasoning IS NOT NULL
    """)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id BIGINT PRIMARY KEY,
        purpose TEXT,
        mode TEXT NOT NULL DEFAULT 'ban'
    )
    """)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS admins (
        id BIGINT PRIMARY KEY
    )
    """)

db_chat_purposes: dict[int, str | None] = {}
db_chat_mode: dict[int, str] = {}
cursor.execute("SELECT id, purpose, mode FROM chats")
for _row in cursor.fetchall():
    db_chat_purposes[_row[0]] = _row[1]
    db_chat_mode[_row[0]] = _row[2]

db_admins: set[int] = set()
cursor.execute("SELECT id FROM admins")
for _row in cursor.fetchall():
    db_admins.add(_row[0])


def effective_admins() -> frozenset[int]:
    return frozenset(cfg.admins) | db_admins


def effective_moderated_chats() -> frozenset[int]:
    return cfg.moderated_chats | frozenset(db_chat_purposes.keys())


def effective_spam_checked_chats() -> frozenset[int]:
    return effective_moderated_chats() | effective_admins()


def effective_member_chats() -> frozenset[int]:
    return effective_moderated_chats() | {cfg.chats.channel}


def chat_purpose(chat_id: int) -> str:
    return db_chat_purposes.get(chat_id) or cfg.llm.chat_purposes.get(
        str(chat_id), cfg.llm.chat_purposes["default"]
    )


def chat_mode(chat_id: int) -> str:
    return db_chat_mode.get(chat_id, "ban")


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


def purge_old_reasonings() -> None:
    """Delete LLM reasonings older than the retention period, as they often contain the content."""
    cutoff = int(time.time()) - cfg.llm.reasoning_retention_days * 86400
    cursor.execute(
        "UPDATE spam_messages SET llm_reasoning = NULL WHERE llm_reasoning IS NOT NULL AND timestamp < ?",
        [cutoff],
    )
    db.commit()


purge_old_reasonings()
