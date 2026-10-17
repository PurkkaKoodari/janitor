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

import re
from collections.abc import Sequence
from html.parser import HTMLParser
from typing import cast

import httpx

from telegram import Message, MessageEntity
from telegram.helpers import escape_markdown

from janitorbot.config import cfg
from janitorbot.log import logger


def expand_links(text: str, entities: Sequence[MessageEntity]) -> str:
    """
    Expand hidden-URL link entities (``text_link``) in ``text`` to Markdown ``[label](url)``.

    Telegram lets spammers hide a URL behind innocuous display text, so the raw message text
    never contains the actual link. Expanding these entities surfaces the URL to both the spam
    regexes and the LLM.
    """
    links = sorted(
        (e for e in entities if e.type == MessageEntity.TEXT_LINK and e.url),
        key=lambda e: e.offset,
    )
    if not links:
        return text
    # Entity offsets/lengths are UTF-16 code units (per the Bot API), so slice the UTF-16 encoding
    # rather than the Python string to stay correct for emoji and other non-BMP characters.
    utf16 = text.encode("utf-16-le")
    parts: list[str] = []
    last = 0
    for entity in links:
        start, end = entity.offset * 2, (entity.offset + entity.length) * 2
        # Leave non-link text verbatim so the existing plain-text regexes keep matching; only
        # escape the structural Markdown characters needed to keep the link well-formed.
        parts.append(utf16[last:start].decode("utf-16-le"))
        label = re.sub(r"([\\\[\]])", r"\\\1", utf16[start:end].decode("utf-16-le"))
        url = escape_markdown(cast(str, entity.url), version=2, entity_type=MessageEntity.TEXT_LINK)
        parts.append(f"[{label}]({url})")
        last = end
    parts.append(utf16[last:].decode("utf-16-le"))
    return "".join(parts)


def parse_message(msg: Message) -> tuple[str, bool, str]:
    """Return (text, has_media, media_info) for the message."""
    # Describe any attachments to the message.
    if msg.contact is not None:  # Contact → "contact: <name>"
        name = f"{msg.contact.first_name or ""} {msg.contact.last_name or ""}".strip()
        media_info = (
            f"The message contains a contact with the name:\n\n{name}"
            if name
            else "The message contains a contact."
        )
    elif msg.photo:  # photo → list[PhotoSize]
        media_info = "The message contains a photo."
    elif msg.video_note:  # VideoNote → "videonote"
        media_info = "The message contains a video note."
    elif (attachment := msg.effective_attachment) is not None:
        media_info = f"The message contains a {type(attachment).__name__.lower()}."
    else:
        media_info = ""

    has_media = bool(media_info)

    # Expand link entities in the message to Markdown.
    if msg.text is not None:
        text = expand_links(msg.text, msg.entities)
    elif msg.caption is not None:
        text = expand_links(msg.caption, msg.caption_entities)
    else:
        text = ""

    # Spammers can hide a URL in the link preview instead of the text, so append it if missing.
    preview = msg.link_preview_options
    if preview and preview.url and str(preview.url) not in text:
        text += f"\n{preview.url}"

    return text, has_media, media_info


def extract_mentions(msg: Message) -> list[str]:
    """
    Return the users and URLs mentioned in the message. Users are lowercase usernames or numeric
    user IDs, URLs include hidden link targets and the link preview URL.
    """
    types: list[str | None] = [
        MessageEntity.MENTION,
        MessageEntity.TEXT_MENTION,
        MessageEntity.URL,
        MessageEntity.TEXT_LINK,
    ]
    entities = msg.parse_entities(types) | msg.parse_caption_entities(types)
    mentions: list[str] = []
    for entity, text in entities.items():
        if entity.type == MessageEntity.TEXT_MENTION and entity.user:
            mentions.append(str(entity.user.id))
        elif entity.type == MessageEntity.MENTION:
            mentions.append(text.removeprefix("@").lower())
        elif entity.type == MessageEntity.TEXT_LINK and entity.url:
            mentions.append(entity.url)
        elif entity.type == MessageEntity.URL:
            mentions.append(text)
    preview = msg.link_preview_options
    if preview and preview.url:
        mentions.append(str(preview.url))
    return sorted(set(mentions))


class _OGDescriptionParser(HTMLParser):
    """Minimal HTML parser that extracts the content of the first og:description meta tag."""

    def __init__(self) -> None:
        super().__init__()
        self.description: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta" or self.description is not None:
            return
        attrs_dict = dict(attrs)
        if attrs_dict.get("property") == "og:description":
            self.description = attrs_dict.get("content") or ""


async def fetch_tme_description(url: str) -> str:
    """Fetch the og:description meta tag of a t.me link, or "" if unavailable."""
    logger.info("Fetching description for link: %s", url)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("Failed to fetch og:description from %s: %s", url, e)
        return ""
    parser = _OGDescriptionParser()
    parser.feed(response.text)
    return (parser.description or "").strip()


CJK_REGEX = re.compile(r"[\u4E00-\u9FFF]")
TME_LINK_REGEX = re.compile(r"https?://t\.me/[^\s)\]]+")


async def prepare_llm_input(text: str, media_info: str, *, force: bool = False) -> str | None:
    """Build the user message content to send to the LLM, embedding attachment info and any
    linked Telegram post description as an <attachments> tag, as described in the system
    prompt's <message_format> section.

    Return None if the message has too little data to be worth checking via LLM, unless forced."""
    # A media with a description always gets forwarded to the LLM.
    # This is checked via formatting, as any media_info usually exceeds cfg.llm.min_length.
    force_llm = force or "\n\n" in media_info or ":" in media_info

    tme_link = TME_LINK_REGEX.search(text)
    if tme_link:
        description = await fetch_tme_description(tme_link.group())
        if description:
            media_info = f"{media_info}\n\nThe message contains a Telegram link with the description:\n\n{description}".strip()
            # A Telegram link with a description always gets forwarded to the LLM.
            force_llm = True

    # CJK spam is often very short, so always check messages containing CJK characters.
    if CJK_REGEX.search(text):
        force_llm = True

    if media_info:
        media_info = f"<attachments>{media_info}</attachments>"

    # Short-circuit the LLM check for short messages, as they are unlikely to be spam
    # and it's not worth the cost and false positives to check them.
    if len(text) < cfg.llm.min_length and not force_llm:
        return None

    return f"{text}\n\n{media_info}".strip()
