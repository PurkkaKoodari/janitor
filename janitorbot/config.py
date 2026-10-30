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
from typing import Annotated, Literal

from openrouter.components import Effort
from pydantic import (
    BaseModel,
    ConfigDict,
    PlainValidator,
    computed_field,
    field_validator,
)

from janitorbot.log import logger


class StrictModel(BaseModel):
    """Base for config models that rejects unknown keys and type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Rule(StrictModel):
    regex: str
    # None means "don't care", True means "must have media", False means "must not have media"
    media: bool | None = None
    # if True, messages matching this rule will be deleted if they are edited to match, even if they previously didn't match
    delete: bool = False


Mode = Literal["test", "delete", "ban"]


class ChatsConfig(StrictModel):
    moderated: list[int]
    monitor: int
    delete_join_leave: bool = False


class SpamConfig(StrictModel):
    regex: list[str]
    mode: Mode = "ban"
    forwarded_story: bool = True
    channel_forwarded_with_keyboard: bool = True
    channel_forwarded_cyrillic_and_links: bool = True
    suspicious_link_regex: str = r"@\w+|https?://t\.me|https?://vk\."


class EstablishedUserConfig(StrictModel):
    min_days: float = 7
    min_nonspam_messages: int = 10
    ban_threshold_days: float = 1
    ban_threshold_messages: int = 3


class NewUserConfig(StrictModel):
    max_days: float = 2
    max_nonspam_messages: int = 2


def _require_false(v: object) -> Literal[False]:
    """Accept only the literal False, unlike Literal[False] which accepts 0 even in strict mode."""
    if v is not False:
        raise ValueError("must be false")
    return v


# False in the A/B test config, meaning "same as A".
SameAsA = Annotated[Literal[False], PlainValidator(_require_false)]


class ABTestConfig(StrictModel):
    model: SameAsA | str = False
    temperature: SameAsA | float = False
    reasoning_effort: SameAsA | Effort = False
    system_prompt: SameAsA | str = False
    threshold: SameAsA | float = False
    multisample_threshold: SameAsA | float = False
    multisample_count: SameAsA | int = False


class LLMConfig(StrictModel):
    openrouter_key: str
    model: str
    temperature: float = 0.5
    reasoning_effort: Effort = "medium"
    timeout: float = 15.0
    retry_delay: float = 1.0
    retry_count: int = 2
    min_length: int = 20
    system_prompt: str
    threshold: float
    multisample_threshold: float = 0.25
    multisample_count: int = 0
    report_provider_issues: bool = True
    report_no_reasoning: bool = True
    outlier_threshold: float = 0.2
    chat_purposes: dict[str, str]
    reasoning_retention_days: int = 90
    a_b_test: ABTestConfig = ABTestConfig()

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

    def a_b_test_config(self) -> "LLMConfig | None":
        """
        Return this config with the A/B test overrides applied, or None if no override differs
        from this config (i.e. A/B testing is disabled).
        """
        overrides = {
            key: value
            for key, value in self.a_b_test.model_dump().items()
            if value is not False and value != getattr(self, key)
        }
        if not overrides:
            return None
        return self.model_copy(update=overrides | {"a_b_test": ABTestConfig()})


class ChannelForwardingConfig(StrictModel):
    source: list[int] = []
    channel: int = 0
    ignore_recent_seconds: int = 900
    ignore_length_min_from_silent: int = 15
    ignore_length_min_from_ignored: int = 20
    ignore_rules: list[Rule] = []

    @field_validator("ignore_rules", mode="after")
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


class Config(StrictModel):
    bot_token: str
    db_name: str
    admins: list[int]
    debug: bool = False
    quiet: bool = False
    admin_name: str
    spam: SpamConfig
    established_user: EstablishedUserConfig = EstablishedUserConfig()
    new_user: NewUserConfig = NewUserConfig()
    chats: ChatsConfig
    llm: LLMConfig
    channel_forwarding: ChannelForwardingConfig = ChannelForwardingConfig()

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
        return frozenset(self.chats.moderated) | frozenset(self.channel_forwarding.source)

    @computed_field
    @cached_property
    def spam_checked_chats(self) -> frozenset[int]:
        """Moderated chats plus admin DM IDs — chats the bot checks for spam."""
        return self.moderated_chats | frozenset(self.admins)

    @computed_field
    @cached_property
    def member_chats(self) -> frozenset[int]:
        """Moderated chats plus the channel — chats the bot is intentionally a member of."""
        return self.moderated_chats | {self.channel_forwarding.channel}


with open("config.toml", "rb") as file:
    cfg = Config.model_validate(tomllib.load(file))
