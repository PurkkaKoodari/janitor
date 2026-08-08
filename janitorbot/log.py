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

import logging, sys
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

format = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format=format)
logging.getLogger("httpx").setLevel(logging.WARNING)
base_logger = logging.getLogger("JanitorBot")


class _PrefixLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    """Logger adapter that prepends a fixed prefix to every log message."""

    def __init__(self, logger: logging.Logger, prefix: str) -> None:
        super().__init__(logger)
        self.prefix = prefix

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        return f"{self.prefix} {msg}", kwargs


# Holds the logger for the message currently being handled.
_message_logger: ContextVar[_PrefixLoggerAdapter | logging.Logger] = ContextVar(
    "message_logger", default=base_logger
)


@contextmanager
def message_logging(chat_id: int, msg_id: int) -> Iterator[None]:
    """Set a per-message logger that prepends ``[chat:msg]`` to every message logged via `logger`."""
    adapter = _PrefixLoggerAdapter(base_logger, f"[{chat_id}:{msg_id}]")
    token = _message_logger.set(adapter)
    try:
        yield
    finally:
        _message_logger.reset(token)


class _LoggerProxy:
    """Proxy forwarding to the logger set in the current context."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_message_logger.get(), name)


# Drop-in replacement for a Logger. Inside a `message_logging` context it
# prepends the `[chat:msg]` prefix; elsewhere it logs via the base logger.
logger = cast(logging.Logger, _LoggerProxy())
