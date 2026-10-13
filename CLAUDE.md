# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

JanitorBot is a Telegram bot that moderates Telegram group chats using an LLM (via OpenRouter) for spam detection. It also forwards messages from "source" chats to a read-only channel for logging. `janitor.py` is a thin launcher; the bot logic lives in the `janitorbot/` package.

## Commands

Uses `uv` for dependency management (Python 3.13, locked in `uv.lock`).

```bash
# Run the bot
uv run python janitor.py

# Stress test the LLM system prompt against a Telegram chat export
# (The agent should NEVER run this due to cost)
uv run python stress_test.py "<chat_purpose>" messages.json > results.csv

# Format code
uv run black .
```

No test suite exists. `config.toml` must exist (copy from `config.example.toml`) — it is gitignored.

## Code style

- Always use type annotations.
- Docstrings for all functions and classes.
- Use f-strings for formatting.
- Black, line length 100, target Python 3.13.
- Always run Black after finalizing code changes.

## Architecture

The bot logic lives in the `janitorbot/` package, split into focused modules. Config, DB, and OpenRouter client are initialized at **module level** on import.

- `janitorbot/log.py` — logging setup and the per-message `logger` proxy
- `janitorbot/config.py` — Pydantic config models and the `cfg` loaded from `config.toml`
- `janitorbot/db.py` — SQLite connection, table setup, and the `effective_*` / `chat_purpose` / `chat_mode` helpers
- `janitorbot/llm.py` — LLM calls and OpenRouter request/response/error handling
- `janitorbot/spam.py` — message parsing, spam detection, and delete/ban actions
- `janitorbot/channel.py` — `forward_to_channel` / `edit_in_channel`: forwarding source messages (and edits) to the channel, only after spam checks pass
- `janitorbot/commands.py` — Telegram command and callback-query handlers
- `janitorbot/search.py` — admin DM lookups of recorded reasonings: forwarded messages (falling back to simulation if unmatched) and `search:` messages
- `janitorbot/main.py` — message routing/spam-check orchestration (`process_message` / `process_edit`), the message handlers, auto-restart watcher, handler registration, `main()`
- `janitor.py` — thin launcher (`from janitorbot.main import main`)

Import chain (acyclic): `log` ← `config` ← `db` ← `llm` ← `spam` ← `channel` ← `commands` ← `search` ← `main`. Module-level side effects (load `cfg`, open DB, build `or_client`) fire in that order.

### Chat roles

- **source** — chats whose messages are forwarded to the channel after spam filtering
- **moderated** — chats where spam is detected/deleted/banned but not forwarded
- **channel** — read-only channel where source messages are mirrored with inline buttons
- **monitor** — chat where deleted spam is forwarded for admin review

### Two-tier admin system

- **Static admins** — defined in `config.toml`, immutable at runtime; `admins[0]` is the superadmin with extra privileges (`/admin`, `/unadmin`, `/chats`)
- **Dynamic admins** — stored in the SQLite `admins` table, managed via `/admin`/`/unadmin`

`effective_admins()`, `effective_moderated_chats()`, etc. merge both static and dynamic sources.

### Message flow

1. `handle_message` → `process_message` (spawned as asyncio task)
2. **Spam check** (`check_spam`):
   - Regex patterns from `config.spam.regex` checked first (no LLM call)
   - Messages shorter than `llm.min_length` skip LLM
   - LLM call via OpenRouter returns `{"prob": float}` using structured JSON output
   - If `prob >= threshold`, calls `attempt_delete_ban`
3. **Channel forwarding** (`forward_to_channel`, called by `process_message` once spam checks pass): ignore rules → length heuristics → copies to channel with "See message" / "Delete this" buttons
4. The `messages` SQLite table maps channel message IDs → original message IDs + author, enabling reply threading and self-deletion

Every spam-checked message is recorded in the `spam_messages` table. LLM verdicts are cached: a message whose LLM input matches identically reuses the median of all recorded probabilities for it instead of calling the LLM.

### Moderation modes (per chat, default `spam.mode`)

- `ban` — delete message and ban sender, unless the sender is an established user without recent repeated spam
- `delete` — delete message only
- `test` — log only, no action

### Auto-restart

A background daemon thread (`_watch_and_restart`) polls `janitor.py`, `config.toml`, and all `janitorbot/*.py` package files every second and calls `os.execv` to restart when any of them changes.

### SQLite schema

- `messages(id, orig_id, author)` — channel msg id → source msg id + author user id
- `spam_messages(chat_id, message_id, llm_sample, ...)` — one row per LLM sample
  - If not LLM-checked, single row with `llm_sample`/`llm_prob` NULL and the non-LLM reasoning or error in `llm_reasoning`
  - Edits add new rows (but may reuse cached verdicts)
  - No primary key
  - Cached verdicts are a single sample with `llm_reasoning` = `CACHED_REASONING`
  - Joins are recorded as `system` rows for each joined user
  - Reasonings are replaced with `PURGED_REASONING` after `reasoning_retention_days`
- `chats(id, purpose, mode, delete_join_leave)` — dynamically added moderated chats and per-chat setting overrides (NULL = config default); cached in memory as `db_chats`
- `admins(id)` — dynamically added admin user IDs

### Config structure (`config.toml`)

Key sections: `[chats]`, `[spam]`, `[established_user]`, `[new_user]`, `[llm]`, `[llm.chat_purposes]`, `[channel_ignore]`. Validated via Pydantic models at startup. `channel_ignore.rules` are regex-based ignore rules for the channel forwarding step.
