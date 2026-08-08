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

import re, tomllib
from functools import cached_property

from pydantic import BaseModel, computed_field, field_validator

from janitorbot.log import logger


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
    forwarded_story: bool = True
    channel_forwarded_with_keyboard: bool = True
    channel_forwarded_cyrillic_and_links: bool = True
    suspicious_link_regex: str = r"@\w+|https?://t\.me|https?://vk\."


class LLMConfig(BaseModel):
    openrouter_key: str
    model: str
    temperature: float = 0.5
    timeout: float = 15.0
    retry_delay: float = 1.0
    retry_count: int = 2
    min_length: int = 20
    system_prompt: str
    threshold: float
    multisample_threshold: float = 0.25
    multisample_count: int = 0
    outlier_threshold: float = 0.2
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
    debug: bool = False
    quiet: bool = False
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
