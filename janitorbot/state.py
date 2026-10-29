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

import time
from typing import Any

from janitorbot.messages import ParsedMessage

# How long evaluation commands or /search without a message wait for the next message.
PENDING_TIMEOUT_MINUTES = 10

_PENDING_KEY = "pending_command"


def get_pending(user_data: dict[Any, Any]) -> ParsedMessage | None:
    """Return the admin's unexpired commands waiting for their next message, if any."""
    pending: tuple[ParsedMessage, float] | None = user_data.get(_PENDING_KEY)
    if pending is None or pending[1] < time.monotonic():
        return None
    return pending[0]


def set_pending(user_data: dict[Any, Any], parsed: ParsedMessage) -> None:
    """Store commands sent without a message to apply to the admin's next message."""
    expires = time.monotonic() + PENDING_TIMEOUT_MINUTES * 60
    user_data[_PENDING_KEY] = (parsed, expires)


def clear_pending(user_data: dict[Any, Any] | None) -> None:
    """Drop the admin's pending command."""
    if user_data is not None:
        user_data.pop(_PENDING_KEY, None)
