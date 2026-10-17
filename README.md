# JanitorBot

JanitorBot is a Telegram bot that keeps group chats free of spam. It reads every message in the groups it moderates and asks an AI model how likely the message is to be spam.

When a message is very likely spam, the bot deletes it and, depending on settings, bans the sender. Everything it removes can be reported to a monitoring chat so admins can see what happened and catch mistakes.

A few details:

- Each group gets a short description of what it's for, so the AI knows what's on-topic there.
- Regular members who've been around for a while don't get banned for a single mistake. Their message is deleted, but they keep their membership.
- When someone who has only just joined and sent a few messages is banned for spam, those earlier messages are deleted too.
- A group can be set to only report suspected spam without deleting it. This is useful for trying the bot out.
- Optionally, the bot also deletes "X joined the group" and "X left the group" messages.

## Bot setup

You need [uv](https://docs.astral.sh/uv/) installed.

### 1. Create a Telegram bot

1. Message [@BotFather](https://t.me/BotFather) on Telegram and send `/newbot`.
2. Pick a name and username. BotFather gives you a **bot token** like `123456:ABC-DEF...`.

### 2. Get an OpenRouter API key

1. Sign up at [openrouter.ai](https://openrouter.ai/) and add some credits.
2. Create a key at [openrouter.ai/keys](https://openrouter.ai/keys).

The default model (`openai/gpt-oss-20b`) is cheap, on the order of $0.0001 per message. Repeated messages reuse earlier verdicts and messages shorter than `min_length` skip the AI, which keeps costs down further.

### 3. Find your Telegram user ID

You'll be the **superadmin**: you can add other admins and you receive error reports. To find your numeric user ID, message a bot such as [@userinfobot](https://t.me/userinfobot).

### 4. Create the config file

```bash
cp config.example.toml config.toml
```

Then edit `config.toml`. The values you must set:

| Setting | Value |
| --- | --- |
| `bot_token` | Token from BotFather |
| `admins` | `[your_user_id]` |
| `admin_name` | Your name or @username. Strangers who message the bot are told to contact this person. |
| `chats.monitor` | Your user ID for now (see the next step) |
| `llm.openrouter_key` | Your OpenRouter key |

Settings you'll probably want to review or change:

- `llm.chat_purposes.default`: the purpose used for groups you haven't given their own. It completes the sentence "a chat group where …", for example `"students discuss the course Algorithms 101"`.
- `spam.regex`: replace the placeholder with phrases that always mean spam in your groups, or use `[]`. Messages matching these phrases are removed without asking the AI.
- `spam.mode`: consider starting with `"test"` so that nothing is deleted while you check how the bot behaves.
- `llm.system_prompt`: the instructions given to the AI. The default lists of what is and isn't spam fit general Finnish university chats, including marketplace chats. Adjust them to your groups.
- `llm.threshold`: how sure the AI must be (0–1) before a message counts as spam.

Every setting is explained in a comment in `config.example.toml`.

### 5. Run the bot

```bash
uv run python janitor.py
```

The bot restarts on its own whenever `config.toml` (or its code) changes, so you don't need to restart it by hand after editing the config.

### 6. Set up a monitor chat (optional)

The monitor chat receives every deleted message along with the AI's reasoning, plus notes about admin actions. With the default `chats.monitor = 0`, there is no monitor chat and nothing is reported. If you're the only admin, you can set it to your own user ID. For a team or shared bot:

1. Create a private group with your fellow admins.
2. Add the bot to it. Because you're an admin, the bot stays and sends you the group's ID (a negative number like `-1001234567890`).
3. Set `chats.monitor` to that ID.

## Group setup

All admin commands work in a private chat with the bot. Send `/help` to see the commands available to you.

### Adding admins

Admins can add the bot to groups, configure groups and use the debugging tools. Only the superadmin can add or remove admins.

1. Ask the new admin to send `/hello` to the bot. You'll receive their user ID.
2. Send `/admin <user_id>` to the bot. Use `/unadmin <user_id>` to remove an admin.

Admins listed in `config.toml` can't be removed with commands.

### Adding the bot to a group

1. **An admin** must add the bot to the group. If anyone else adds it, the bot leaves.
2. Make the bot a group admin with permission to **delete messages** and **ban users**, if you want it to do those things.
3. The bot posts the group's ID to the monitor chat, or sends it to you if there is none.
4. Send `/moderate <chat_id>` to the bot. The reply shows the group's current settings and warns you if the bot is missing any permissions.

`/unmoderate <chat_id>` stops moderating a group. The superadmin can list all moderated groups with `/chats`.

### Configuring a group

Each setting can be shown by leaving out the value. Use `default` to go back to the value in `config.toml`.

- `/purpose <chat_id> [text]`: what the group is for. It completes "a chat group where …", for example `/purpose -1001234567890 people sell and give away used furniture in Helsinki`. A good purpose matters a lot: it tells the AI what's on-topic.
- `/mode <chat_id> [ban|delete|test]`:
  - `ban`: delete spam and ban the sender, unless they're an established member
  - `delete`: delete spam only
  - `test`: report suspected spam to the monitor chat and take no action
- `/deletejoins <chat_id> [on|off]`: delete "joined" and "left" messages.

Messages from admins are never deleted in any group. When an admin's message looks like spam, the bot only reports what it *would* have done.

## Debugging verdicts

When the bot makes a mistake, admins can find out why by messaging the bot privately. The bot keeps the AI's reasoning for 90 days by default (`llm.reasoning_retention_days`). After that, only the verdicts remain.

### Forward a message

Forward a message from a moderated group to the bot. The bot replies with the verdict it gave that message, including each AI sample's spam probability and reasoning.

If the bot can't find the original check, it says so and checks the message again from scratch. This can happen, for example, when the message was edited.

You can also forward deleted spam from the monitor chat.

### `search:`

To look up a message you can't forward, for example because it was deleted, put `search:` alone on the first line and paste the message text below it:

```
search:
Earn $500 a day from home, DM me!
```

The bot replies with the most recent check of a message with the same text. A `search:` lookup never runs a new check.

### Simulating a message

Any other message you send to the bot privately is checked as if it had been posted in a group, but nothing is deleted. If the bot thinks it's spam, it sends you the verdict and reasoning. If it doesn't, it stays silent. To see the reasoning in that case, send the same text again with `search:` above it.

Messages shorter than `llm.min_length` aren't sent to the AI at all.

Simulations use the default purpose. To test a group's purpose, put it on the first line:

```
purpose: people sell and give away used furniture in Helsinki
Selling my old sofa, 20€, pick up in Kallio
```

### `debug:`

To see exactly what the AI receives without running a check, put `debug:` on the first line:

```
debug:
Selling my old sofa, 20€, pick up in Kallio
```

The bot replies with the full instructions and message it would send to the AI. You can combine this with `purpose:` by putting the `purpose:` line first.

### Debug mode

Setting `debug = true` in `config.toml` sends the superadmin the AI's reasoning for **every** checked message, including ones that weren't spam. Only turn it on for low-traffic setups.

## Channel forwarding (legacy)

> [!NOTE]
> This feature is kept for an existing deployment and has significant limitations. It is not recommended for new setups.

The bot can copy messages from "source" groups into a read-only Telegram channel. Each copied message gets two buttons: **See message**, which links to the original, and **Delete this**. Only messages that pass the spam check are copied. Short chatter is filtered out with simple length rules and the regex rules in `channel_forwarding.ignore_rules`.

Setup is done in the `[channel_forwarding]` section of `config.toml`:

- `source`: groups to copy from. They're also moderated automatically.
- `channel`: the channel to copy to. The bot must be an admin there.

Limitations:

- It can only be configured in `config.toml`. There are no commands for it.
- It's designed for a single source group:
  - Replies in the channel may thread incorrectly when there are several source groups.
  - The **Delete this** button can only be used by the message's author, bot admins, and admins of the *first* source group.
- The rules for filtering out short chatter are reset whenever the bot restarts.
- Edits to the original are copied to the channel only for text and captions.

Forwarding a message from the channel to the bot (see [Debugging verdicts](#debugging-verdicts)) shows the verdict of the original message in the source group.

## License

[GNU Affero General Public License v3.0 or later](LICENSE).
