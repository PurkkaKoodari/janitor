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

import asyncio
from collections.abc import Coroutine
from typing import Any

from janitorbot.log import logger

# Strong references to running background tasks, so they aren't garbage collected.
_background_tasks = set[asyncio.Task[Any]]()


def spawn_task[T](coro: Coroutine[Any, Any, T], kind: str) -> asyncio.Task[T]:
    """
    Run the coroutine as a fire-and-forget background task, logging any unexpected error with the
    given kind of task.
    """

    def done(task: asyncio.Task[T]) -> None:
        _background_tasks.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            logger.error("Unexpected error in background task: %s", kind, exc_info=exc)

    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(done)
    return task
