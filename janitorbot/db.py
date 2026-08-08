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

import sqlite3

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
