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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

from janitorbot.config import Mode, cfg

CACHED_REASONING = "cached"
PURGED_REASONING = "purged"

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
    DROP INDEX IF EXISTS spam_messages_reasoning_timestamp
    """)
cursor.execute(f"""
    CREATE INDEX IF NOT EXISTS spam_messages_purgeable
    ON spam_messages (timestamp)
    WHERE llm_reasoning IS NOT NULL AND llm_reasoning NOT IN ('{CACHED_REASONING}', '{PURGED_REASONING}')
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
    llm_samples: Sequence[tuple[float, str | None]] | None = None
    reasoning: str | None = None


def record_message(record: MessageRecord) -> None:
    """
    Store the message, with one row per LLM sample, or a single row with the non-LLM reasoning.
    Edits add new rows.
    """
    samples: Sequence[tuple[int | None, tuple[float | None, str | None]]] = list(
        enumerate(record.llm_samples or [])
    ) or [(None, (None, record.reasoning))]
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


@dataclass
class RecordedCheck:
    """
    A single recorded spam check of a message, with its LLM samples, or the non-LLM reasoning if
    there are none.
    """

    rowid: int
    chat_id: int
    message_id: int
    timestamp: int
    spam: bool
    system_prompt_hash: bytes | None
    content_hash: bytes | None
    llm_samples: list[tuple[float, str | None]]
    reasoning: str | None = None

    @property
    def cached(self) -> bool:
        """Whether the verdict was reused from earlier checks."""
        return len(self.llm_samples) == 1 and self.llm_samples[0][1] == CACHED_REASONING

    @property
    def purged(self) -> bool:
        """Whether the reasoning was purged due to age."""
        return len(self.llm_samples) > 0 and self.llm_samples[0][1] == PURGED_REASONING


class _CheckRow(NamedTuple):
    """A spam_messages row as selected for grouping into checks."""

    rowid: int
    chat_id: int
    message_id: int
    timestamp: int
    spam: int
    system_prompt_hash: bytes | None
    content_hash: bytes | None
    llm_sample: int | None
    llm_prob: float | None
    llm_reasoning: str | None


_CHECK_COLUMNS = ", ".join(_CheckRow._fields)


def _group_checks(rows: Iterable[_CheckRow]) -> list[RecordedCheck]:
    """Group spam_messages rows (in rowid order) into checks; each check starts at sample 0/NULL."""
    checks: list[RecordedCheck] = []
    for row in rows:
        if not row.llm_sample or not checks:
            checks.append(
                RecordedCheck(
                    row.rowid,
                    row.chat_id,
                    row.message_id,
                    row.timestamp,
                    bool(row.spam),
                    row.system_prompt_hash,
                    row.content_hash,
                    [],
                )
            )
        if row.llm_prob is not None:
            checks[-1].llm_samples.append((row.llm_prob, row.llm_reasoning))
        else:
            checks[-1].reasoning = row.llm_reasoning
    return checks


def find_checks(user_id: int | None, timestamp: int, c_hash: bytes | None) -> list[RecordedCheck]:
    """
    Return the recorded checks of non-system messages sent at the timestamp, optionally filtered
    by sender and LLM user message hash.
    """
    conditions = ["timestamp = ?", "NOT system"]
    params: list[object] = [timestamp]
    if user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)
    if c_hash is not None:
        conditions.append("content_hash = ?")
        params.append(c_hash)
    cursor.execute(
        f"SELECT {_CHECK_COLUMNS} FROM spam_messages WHERE {" AND ".join(conditions)} ORDER BY rowid",
        params,
    )
    return _group_checks(map(_CheckRow._make, cursor))


def latest_uncached_check(
    *, c_hash: bytes, sp_hash: bytes | None = None, before: int | None = None
) -> RecordedCheck | None:
    """
    Return the latest LLM check with reasoning (i.e. not a cached verdict) of the LLM user
    message hash, optionally also matching the system prompt hash and preceding the given rowid.
    """
    conditions = [
        "content_hash = ?",
        "llm_prob IS NOT NULL",
        f"llm_reasoning != '{CACHED_REASONING}'",
    ]
    params: list[object] = [c_hash]
    if sp_hash is not None:
        conditions.append("system_prompt_hash = ?")
        params.append(sp_hash)
    if before is not None:
        conditions.append("rowid < ?")
        params.append(before)
    cursor.execute(f"SELECT MAX(rowid) FROM spam_messages WHERE {" AND ".join(conditions)}", params)
    (last,) = cursor.fetchone()
    if last is None:
        return None
    # The found row is the last sample of its check; collect rows backwards to its first sample.
    cursor.execute(
        f"""
        SELECT {_CHECK_COLUMNS} FROM spam_messages
        WHERE content_hash = ? AND rowid <= ?
        ORDER BY rowid DESC
        """,
        [c_hash, last],
    )
    rows: list[_CheckRow] = []
    for row in map(_CheckRow._make, cursor):
        rows.append(row)
        if not row.llm_sample:
            break
    return _group_checks(rows[::-1])[-1]


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


def is_new_user(chat_id: int, user_id: int) -> bool:
    """
    Return whether the user's first recorded join to the chat is at most ``max_days`` old, and
    they have sent at most ``max_nonspam_messages`` non-system messages not deemed spam in the
    chat. Users never seen joining are not new.
    """
    cursor.execute(
        """
        SELECT MIN(joined), COALESCE(SUM(NOT spam AND NOT system), 0) FROM (
            SELECT
                MIN(CASE WHEN system THEN timestamp END) AS joined,
                MAX(spam) AS spam,
                MAX(system) AS system
            FROM spam_messages
            WHERE user_id = ? AND chat_id = ?
            GROUP BY message_id
        )
        """,
        [user_id, chat_id],
    )
    joined, nonspam = cursor.fetchone()
    cutoff = time.time() - cfg.new_user.max_days * 86400
    return joined is not None and joined >= cutoff and nonspam <= cfg.new_user.max_nonspam_messages


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


def user_message_ids(chat_id: int, user_id: int, *, joins_only: bool = False) -> list[int]:
    """Return the IDs of the recorded messages (or only join messages) of the user in the chat."""
    cursor.execute(
        f"""
        SELECT DISTINCT message_id FROM spam_messages
        WHERE user_id = ? AND chat_id = ? {"AND system" if joins_only else ""}
        """,
        [user_id, chat_id],
    )
    return [row[0] for row in cursor.fetchall()]


def pop_mirrored_message_ids(orig_ids: list[int]) -> list[int]:
    """Remove the channel mirrors of the given source message IDs and return their IDs."""
    placeholders = ", ".join("?" * len(orig_ids))
    cursor.execute(f"SELECT id FROM messages WHERE orig_id IN ({placeholders})", orig_ids)
    ids = [row[0] for row in cursor.fetchall()]
    cursor.execute(f"DELETE FROM messages WHERE orig_id IN ({placeholders})", orig_ids)
    db.commit()
    return ids


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
        f"""
        UPDATE spam_messages SET llm_reasoning = '{PURGED_REASONING}'
        WHERE llm_reasoning IS NOT NULL AND llm_reasoning NOT IN ('{CACHED_REASONING}', '{PURGED_REASONING}') AND timestamp < ?
        """,
        [cutoff],
    )
    db.commit()
