#!/usr/bin/env python
"""
Stress test the LLM spam detection system prompt against a Telegram chat dump.

Usage:
    python stress_test.py <chat_purpose> <messages.json> > results.csv
"""

import asyncio
import csv
import json
import logging
import sys
import textwrap
from typing import Any

from janitorbot.config import cfg
from janitorbot.spam import llm_check


def extract_text(text_field: str | list[str | dict[str, Any]]) -> str:
    """Extract plain text from a Telegram export text field.

    The field can be:
    - A plain string
    - A list of strings and/or entity objects {"type": ..., "text": ...}
    """
    if isinstance(text_field, str):
        return text_field
    parts: list[str] = []
    for item in text_field:
        if isinstance(item, str):
            parts.append(item)
        else:
            parts.append(item.get("text", ""))
    return "".join(parts)


def ask_continue(recent_spam: list[dict[str, Any]]) -> bool:
    """Show recent spam messages and ask the user whether to continue."""
    print("\n--- 3 spam messages detected ---", file=sys.stderr)
    for entry in recent_spam:
        preview = textwrap.shorten(entry["text"], width=100, placeholder="...")
        print(f"  [{entry['prob']:.2f}] {entry['sender']}: {preview}", file=sys.stderr)
    print("\nContinue? [Y/n] ", end="", file=sys.stderr, flush=True)
    try:
        answer = sys.stdin.readline().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("", "y", "yes")


async def main() -> None:
    """Run the stress test."""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <chat_purpose> <messages.json>", file=sys.stderr)
        sys.exit(1)

    purpose = sys.argv[1]
    messages_path = sys.argv[2]

    with open(messages_path, encoding="utf-8") as f:
        data = json.load(f)

    messages = [m for m in data["messages"] if m.get("type") == "message"]
    total = len(messages)

    writer = csv.writer(sys.stdout)
    writer.writerow(["probability", "spam", "msg_id", "date", "sender", "text", "reasoning"])
    sys.stdout.flush()

    pending_spam: list[dict[str, Any]] = []
    checked = 0
    skipped = 0
    total_spam = 0

    for i, msg in enumerate(messages):
        text = extract_text(msg.get("text", ""))
        sender = msg.get("from", "?")
        msg_id = msg.get("id", "?")
        date = msg.get("date", "?")

        print(
            f"\r\x1b[K[{i+1}/{total}] checked={checked} skipped={skipped} spam={total_spam}",
            end="",
            file=sys.stderr,
            flush=True,
        )

        if len(text) < 20:
            skipped += 1
            continue

        try:
            prob, reasoning, _ = await llm_check(text, purpose)
        except Exception as e:
            print(f"\nError on msg {msg_id}: {e}", file=sys.stderr)
            continue

        checked += 1
        is_spam = prob >= cfg.llm.threshold
        writer.writerow(
            [f"{prob:.4f}", "1" if is_spam else "0", msg_id, date, sender, text, reasoning]
        )
        sys.stdout.flush()

        print(
            f"\r\x1b[K[{i+1}/{total}] checked={checked} skipped={skipped} prob={prob:.2f} {'SPAM' if is_spam else ''}",
            end="",
            file=sys.stderr,
            flush=True,
        )

        if is_spam:
            total_spam += 1
            pending_spam.append({"prob": prob, "sender": sender, "text": text})
            if len(pending_spam) >= 3:
                if not ask_continue(pending_spam):
                    print("\nAborted by user.", file=sys.stderr)
                    break
                pending_spam.clear()

    print(
        f"\nDone. Checked {checked} messages, skipped {skipped} short messages, {total_spam} spam.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    # Disable JanitorBot logging to avoid cluttering the output CSV with logs.
    logging.getLogger("JanitorBot").setLevel(logging.FATAL)
    asyncio.run(main())
