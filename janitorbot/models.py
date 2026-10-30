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

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class Verdict:
    """A verdict for a user message and system prompt, with its samples."""

    spam: bool
    """Whether the message is considered spam."""
    reasoning: str
    """The reasoning behind the verdict."""
    prob: float = 0.0
    """The spam probability of the message, between 0.0 and 1.0."""
    cached: bool = False
    """Whether the verdict was reused from earlier checks."""
    samples_to_store: Sequence[tuple[float, str]] = ()
    """The samples to store in the database."""
    sample_probs: Sequence[float] = ()
    """The spam probabilities of the samples, sorted ascending."""

    @property
    def spam_str(self) -> str:
        """Return a string representation of the spam verdict."""
        return "spam" if self.spam else "not spam"

    @property
    def sample_range(self) -> str:
        """Return a string representation of the range of sample probabilities, or only the median if there are less than 2 samples."""
        if len(self.sample_probs) < 2:
            return f"{self.prob:.2f}"
        return f"{self.prob:.2f} ({min(self.sample_probs):.2f}\u2026{max(self.sample_probs):.2f})"
