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

from janitorbot.config import LLMConfig, Mode, cfg
from janitorbot.models import Verdict

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
# Old spam_messages tables have a row per sample. If found, rename it to migrate the samples
# (grouped into checks) to spam_checks after creating the new tables. The sqlite3 module doesn't
# open transactions for DDL, so open one explicitly to make the whole migration atomic.
cursor.execute("BEGIN")
cursor.execute("PRAGMA table_info(spam_messages)")
_spam_messages_columns = {_row[1] for _row in cursor.fetchall()}
_migrate_spam_messages = "llm_sample" in _spam_messages_columns
if _migrate_spam_messages:
    cursor.execute("ALTER TABLE spam_messages RENAME TO spam_messages_old")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS spam_messages (
        chat_id BIGINT NOT NULL,
        message_id BIGINT NOT NULL,
        user_id BIGINT NOT NULL,
        timestamp BIGINT NOT NULL,
        system BOOLEAN NOT NULL,
        spam BOOLEAN NOT NULL,
        mentions TEXT,
        check_id INTEGER
    )
    """)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS spam_checks (
        check_id INTEGER NOT NULL,
        sample INTEGER NOT NULL,
        chat_id BIGINT NOT NULL,
        message_id BIGINT NOT NULL,
        timestamp BIGINT NOT NULL,
        spam BOOLEAN,
        prob REAL,
        reasoning TEXT,
        system_prompt_hash BLOB,
        content_hash BLOB,
        text_hash BLOB,
        PRIMARY KEY (check_id, sample)
    )
    """)
if _migrate_spam_messages:
    _text_hash = "text_hash" if "text_hash" in _spam_messages_columns else "NULL"
    cursor.execute(f"""
        SELECT chat_id, message_id, user_id, timestamp, system, spam, mentions,
            system_prompt_hash, content_hash, {_text_hash}, llm_sample, llm_prob, llm_reasoning
        FROM spam_messages_old ORDER BY rowid
        """)
    _messages: list[tuple[object, ...]] = []
    _checks: list[tuple[object, ...]] = []
    _check_id = 0
    for *_message, _sp_hash, _c_hash, _t_hash, _sample, _prob, _reasoning in cursor.fetchall():
        _hashes = (_sp_hash, _c_hash, _t_hash)
        _chat_id, _message_id, _, _timestamp, _system, _spam = _message[:6]
        _check = (_chat_id, _message_id, _timestamp, _spam)
        if _sample:
            # Further sample of the previous check.
            _checks.append((_check_id, _sample, *_check, _prob, _reasoning, *_hashes))
        elif _system:
            # Join without a check.
            _messages.append((*_message, None))
        else:
            _check_id += 1
            _messages.append((*_message, _check_id))
            _checks.append((_check_id, 0, *_check, _prob, _reasoning, *_hashes))
    cursor.executemany("INSERT INTO spam_messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)", _messages)
    cursor.executemany("INSERT INTO spam_checks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", _checks)
    # Also drops the old indexes, which keep their names when renamed.
    cursor.execute("DROP TABLE spam_messages_old")
cursor.execute("DROP TABLE IF EXISTS a_b_test_samples")
db.commit()
cursor.execute("""
    CREATE INDEX IF NOT EXISTS spam_messages_user
    ON spam_messages (user_id, timestamp)
    """)
cursor.execute("""
    CREATE INDEX IF NOT EXISTS spam_messages_user_chat
    ON spam_messages (user_id, chat_id)
    """)
cursor.execute("""
    CREATE INDEX IF NOT EXISTS spam_messages_timestamp
    ON spam_messages (timestamp)
    """)
cursor.execute("""
    CREATE INDEX IF NOT EXISTS spam_checks_hashes
    ON spam_checks (content_hash, system_prompt_hash)
    """)
cursor.execute("""
    CREATE INDEX IF NOT EXISTS spam_checks_text_hash
    ON spam_checks (text_hash)
    WHERE text_hash IS NOT NULL
    """)
cursor.execute(f"""
    CREATE INDEX IF NOT EXISTS spam_checks_purgeable
    ON spam_checks (timestamp)
    WHERE reasoning IS NOT NULL AND reasoning NOT IN ('{CACHED_REASONING}', '{PURGED_REASONING}')
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
    return effective_moderated_chats() | {cfg.channel_forwarding.channel}


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


def system_prompt_hash(system_prompt: str, llm_cfg: LLMConfig) -> bytes:
    """
    Hash the system prompt and the LLM settings affecting the verdict into a 64-bit verdict cache
    key component.
    """
    key = f"{llm_cfg.model}\n{llm_cfg.temperature}\n{llm_cfg.reasoning_effort}\n{system_prompt}"
    return hashlib.blake2b(key.encode(), digest_size=8).digest()


def content_hash(llm_user_message: str) -> bytes:
    """Hash the user message fed to the LLM into a 128-bit key."""
    return hashlib.blake2b(llm_user_message.encode(), digest_size=16).digest()


def text_hash(text: str) -> bytes:
    """Hash the message text into a 128-bit key."""
    return content_hash(text.strip())


def _insert_check(
    chat_id: int,
    message_id: int,
    timestamp: int,
    spam: bool | None,
    samples: Sequence[tuple[float | None, str | None]],
    sp_hash: bytes | None,
    c_hash: bytes | None,
    t_hash: bytes | None,
) -> int:
    """Store a check with its samples (each a spam probability and reasoning); return its ID."""
    cursor.execute("SELECT COALESCE(MAX(check_id), 0) + 1 FROM spam_checks")
    (check_id,) = cursor.fetchone()
    cursor.executemany(
        """
        INSERT INTO spam_checks (
            check_id, sample, chat_id, message_id, timestamp, spam, prob, reasoning,
            system_prompt_hash, content_hash, text_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            [
                check_id,
                index,
                chat_id,
                message_id,
                timestamp,
                spam,
                prob,
                reasoning,
                sp_hash,
                c_hash,
                t_hash,
            ]
            for index, (prob, reasoning) in enumerate(samples)
        ],
    )
    return check_id


def _insert_spam_message(
    chat_id: int,
    message_id: int,
    user_id: int,
    timestamp: int,
    system: bool,
    spam: bool,
    mentions: Sequence[str],
    check_id: int | None,
) -> None:
    """Store a spam_messages row."""
    cursor.execute(
        """
        INSERT INTO spam_messages (
            chat_id, message_id, user_id, timestamp, system, spam, mentions, check_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            chat_id,
            message_id,
            user_id,
            timestamp,
            system,
            spam,
            " ".join(mentions) or None,
            check_id,
        ],
    )


def record_message_check(
    chat_id: int,
    message_id: int,
    user_id: int,
    timestamp: int,
    mentions: Sequence[str],
    verdict: Verdict,
    t_hash: bytes,
    sp_hash: bytes | None = None,
    c_hash: bytes | None = None,
    primary: bool = True,
) -> None:
    """
    Store a spam check of a message, with either the LLM samples of the verdict or its non-LLM
    reasoning. Edits add new rows.

    Non-primary checks (such as A/B tests) are stored without a verdict or spam_messages row, so
    they only affect verdict caching and cached reasoning lookups.
    """
    samples = verdict.samples_to_store or [(None, verdict.reasoning)]
    check_id = _insert_check(
        chat_id=chat_id,
        message_id=message_id,
        timestamp=timestamp,
        spam=verdict.spam if primary else None,
        samples=samples,
        sp_hash=sp_hash,
        c_hash=c_hash,
        t_hash=t_hash,
    )
    if primary:
        _insert_spam_message(
            chat_id=chat_id,
            message_id=message_id,
            user_id=user_id,
            timestamp=timestamp,
            system=False,
            spam=verdict.spam,
            mentions=mentions,
            check_id=check_id,
        )
    db.commit()


def record_join(chat_id: int, message_id: int, timestamp: int, user_ids: Iterable[int]) -> None:
    """Store a join message as a system message, with no check, from each joined user."""
    for user_id in user_ids:
        _insert_spam_message(
            chat_id=chat_id,
            message_id=message_id,
            user_id=user_id,
            timestamp=timestamp,
            system=True,
            spam=False,
            mentions=[],
            check_id=None,
        )
    db.commit()


def cached_verdict(sp_hash: bytes, c_hash: bytes, threshold: float) -> Verdict | None:
    """
    Return the median of all recorded LLM spam probabilities for the same LLM user message and
    system prompt, including non-primary checks, as a cached verdict.
    """
    cursor.execute(
        """
        SELECT prob FROM spam_checks
        WHERE content_hash = ? AND system_prompt_hash = ? AND prob IS NOT NULL
        ORDER BY prob
        """,
        [c_hash, sp_hash],
    )
    sample_probs = [row[0] for row in cursor.fetchall()]
    if not sample_probs:
        return None
    prob = sample_probs[(len(sample_probs) - 1) // 2]
    return Verdict(
        cached=True,
        prob=prob,
        spam=prob >= threshold,
        reasoning="Cached LLM verdict from previous message",
        sample_probs=sample_probs,
        # Only record the reused median, not the reasoning or the individual samples.
        samples_to_store=[(prob, CACHED_REASONING)],
    )


@dataclass
class RecordedCheck:
    """
    A single recorded spam check of a message, with its LLM samples, or the non-LLM reasoning if
    there are none. Non-primary checks have no verdict.
    """

    check_id: int
    chat_id: int
    message_id: int
    timestamp: int
    spam: bool | None
    system_prompt_hash: bytes | None
    content_hash: bytes | None
    llm_samples: list[tuple[float, str | None]]
    reasoning: str | None = None

    @property
    def primary(self) -> bool:
        """Whether this check determined the verdict of its message."""
        return self.spam is not None

    @property
    def cached(self) -> bool:
        """Whether the verdict was reused from earlier checks."""
        return len(self.llm_samples) == 1 and self.llm_samples[0][1] == CACHED_REASONING

    @property
    def purged(self) -> bool:
        """Whether the reasoning was purged due to age."""
        return len(self.llm_samples) > 0 and self.llm_samples[0][1] == PURGED_REASONING


class _CheckRow(NamedTuple):
    """
    A spam_checks row as selected for grouping into checks.
    """

    check_id: int
    chat_id: int
    message_id: int
    timestamp: int
    spam: int | None
    system_prompt_hash: bytes | None
    content_hash: bytes | None
    prob: float | None
    reasoning: str | None


def _select_checks(conditions: str, *, join_messages: bool = False) -> str:
    """
    Return a query selecting the rows of the matching checks (as c) in order, for _group_checks.
    If join_messages, the conditions can also filter on the spam_messages row (as m) referencing
    each check, which excludes non-primary checks.
    """
    join = "JOIN spam_messages m ON m.check_id = c.check_id" if join_messages else ""
    return f"""
        SELECT {", ".join(f"c.{field}" for field in _CheckRow._fields)} FROM spam_checks c {join}
        WHERE {conditions}
        ORDER BY c.check_id, c.sample
        """


def _group_checks(rows: Iterable[_CheckRow]) -> list[RecordedCheck]:
    """Group spam_checks rows (in check ID and sample order) into checks."""
    checks: list[RecordedCheck] = []
    for row in rows:
        if not checks or checks[-1].check_id != row.check_id:
            checks.append(
                RecordedCheck(
                    row.check_id,
                    row.chat_id,
                    row.message_id,
                    row.timestamp,
                    None if row.spam is None else bool(row.spam),
                    row.system_prompt_hash,
                    row.content_hash,
                    [],
                )
            )
        if row.prob is not None:
            checks[-1].llm_samples.append((row.prob, row.reasoning))
        else:
            checks[-1].reasoning = row.reasoning
    return checks


def find_checks(
    user_id: int | None, timestamp: int, c_hash: bytes | None, text_only: bool = False
) -> list[RecordedCheck]:
    """
    Return the recorded checks of non-system messages sent at the timestamp, optionally filtered
    by sender and LLM user message hash.
    """
    # Filter by spam_messages for its timestamp and sender indexes. System messages have no checks.
    conditions = ["m.timestamp = ?"]
    params: list[object] = [timestamp]
    if user_id is not None:
        conditions.append("m.user_id = ?")
        params.append(user_id)
    if c_hash is not None:
        hash_column = "text_hash" if text_only else "content_hash"
        conditions.append(f"c.{hash_column} = ?")
        params.append(c_hash)
    cursor.execute(_select_checks(" AND ".join(conditions), join_messages=True), params)
    return _group_checks(map(_CheckRow._make, cursor))


def find_mirrored_checks(channel_msg_id: int) -> list[RecordedCheck]:
    """Return the recorded checks of the source message mirrored as the channel message."""
    cursor.execute("SELECT orig_id, author FROM messages WHERE id = ?", [channel_msg_id])
    row = cursor.fetchone()
    if row is None:
        return []
    orig_id, author = row
    # The messages table has no chat ID, so match on the sender in any source chat.
    placeholders = ", ".join("?" * len(cfg.channel_forwarding.source))
    cursor.execute(
        _select_checks(
            f"c.message_id = ? AND m.user_id = ? AND c.chat_id IN ({placeholders})",
            join_messages=True,
        ),
        [orig_id, author, *cfg.channel_forwarding.source],
    )
    return _group_checks(map(_CheckRow._make, cursor))


def latest_uncached_check(
    *,
    c_hash: bytes,
    text_only: bool = False,
    llm_only: bool = True,
    sp_hash: bytes | None = None,
    before: int | None = None,
    primary_only: bool = True,
) -> RecordedCheck | None:
    """
    Return the latest check that is not a cached verdict, of either the LLM user message hash or
    the text hash, optionally also matching the system prompt hash and preceding the given check
    ID. Unless llm_only, checks with a non-LLM verdict or an LLM error also match. Unless
    primary_only, non-primary checks also match.
    """
    hash_column = "text_hash" if text_only else "content_hash"
    conditions = [f"{hash_column} = ?", f"reasoning IS NOT '{CACHED_REASONING}'"]
    params: list[object] = [c_hash]
    if llm_only:
        conditions.append("prob IS NOT NULL")
    if sp_hash is not None:
        conditions.append("system_prompt_hash = ?")
        params.append(sp_hash)
    if before is not None:
        conditions.append("check_id < ?")
        params.append(before)
    if primary_only:
        conditions.append("spam IS NOT NULL")
    cursor.execute(
        f"SELECT MAX(check_id) FROM spam_checks WHERE {" AND ".join(conditions)}", params
    )
    (check_id,) = cursor.fetchone()
    if check_id is None:
        return None
    cursor.execute(_select_checks("c.check_id = ?"), [check_id])
    return _group_checks(map(_CheckRow._make, cursor))[0]


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


def mirrored_message_ids(orig_ids: list[int]) -> list[int]:
    """Return the IDs of the channel mirrors of the given source message IDs."""
    placeholders = ", ".join("?" * len(orig_ids))
    cursor.execute(f"SELECT id FROM messages WHERE orig_id IN ({placeholders})", orig_ids)
    return [row[0] for row in cursor.fetchall()]


def delete_mirror_records(ids: list[int]) -> None:
    """Remove the records of the given channel message IDs."""
    placeholders = ", ".join("?" * len(ids))
    cursor.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", ids)
    db.commit()


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
    """
    Delete reasonings of messages older than the retention period, as they often contain the
    content.
    """
    cutoff = int(time.time()) - cfg.llm.reasoning_retention_days * 86400
    cursor.execute(
        f"""
        UPDATE spam_checks SET reasoning = '{PURGED_REASONING}'
        WHERE reasoning IS NOT NULL AND reasoning NOT IN ('{CACHED_REASONING}', '{PURGED_REASONING}')
        AND timestamp < ?
        """,
        [cutoff],
    )
    db.commit()
