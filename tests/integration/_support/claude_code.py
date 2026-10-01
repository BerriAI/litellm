"""Shared Claude Code-shaped request builders and upstream stream fixtures for integration contracts."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import Final

from integration._support.wire import Request
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
ANTHROPIC_API_KEY: Final = "synthetic-anthropic-key"
FABLE: Final = "claude-fable-5-1"
OPUS: Final = "claude-opus-5-5"
CLI_BETA: Final = (
    "claude-code-20250219,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,"
    "context-management-2025-06-27,prompt-caching-scope-2026-01-05"
)
FRONTIER_CLI_BETA: Final = (
    f"{CLI_BETA},mid-conversation-system-2026-04-07,per-turn-control-2026-07-01,"
    "mid-conversation-tool-changes-2026-07-01,effort-2025-11-24"
)
CACHE: Final = {"type": "ephemeral"}
CONTEXT_MANAGEMENT: Final = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}
THINKING_BUDGET: Final = {"budget_tokens": 31999, "type": "enabled", "display": "omitted"}
THINKING_ADAPTIVE: Final = {"type": "adaptive", "display": "omitted"}
CLAUDE_CODE_REASONING_BETAS: Final = (
    "effort-2025-11-24",
    "interleaved-thinking-2025-05-14",
    "thinking-token-count-2026-05-13",
)
CLAUDE_CODE_CACHING_BETAS: Final = ("extended-cache-ttl-2025-04-11", "prompt-caching-scope-2026-01-05")
CACHING_CLI_BETA: Final = f"{CLI_BETA},extended-cache-ttl-2025-04-11"
REASONING_FIELDS: Final = ("thinking", "output_config", "reasoning_effort", "temperature")
_OUTPUT_USAGE_KEYS: Final = frozenset({"output_tokens", "output_tokens_details"})
METADATA_USER_ID: Final = json.dumps(
    {
        "device_id": "0" * 64,
        "account_uuid": "",
        "session_id": "00000000-0000-4000-8000-000000000000",
    }
)


def schema(properties: JsonValue, required: tuple[str, ...]) -> dict[str, JsonValue]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def field(description: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"description": description, **extra}


def tools() -> tuple[dict[str, JsonValue], ...]:
    _MAX: Final = 9007199254740991
    return (
        {
            "name": "Agent",
            "description": "Launch a new agent to handle complex, multi-step tasks.",
            "input_schema": schema(
                {
                    "description": field("A short (3-5 word) description of the task", type="string"),
                    "prompt": field("The task for the agent to perform", type="string"),
                    "subagent_type": field("The type of specialized agent to use for this task", type="string"),
                    "model": field(
                        "Optional model override for this agent.",
                        type="string",
                        enum=["sonnet", "opus", "haiku", "fable"],
                    ),
                    "run_in_background": field(
                        "Agents run in the background by default; you will be notified when one completes.",
                        type="boolean",
                    ),
                    "isolation": field("Isolation mode.", type="string", enum=["worktree", "remote"]),
                },
                ("description", "prompt"),
            ),
        },
        {
            "name": "Bash",
            "description": "Executes a given bash command and returns its output.",
            "input_schema": schema(
                {
                    "command": field("The command to execute", type="string"),
                    "timeout": field("Optional timeout in milliseconds (max 600000)", type="number"),
                    "description": field(
                        "Clear, concise description of what this command does in active voice.",
                        type="string",
                    ),
                    "run_in_background": field("Set to true to run this command in the background.", type="boolean"),
                    "dangerouslyDisableSandbox": field(
                        "Set this to true to dangerously override sandbox mode and run commands without sandboxing.",
                        type="boolean",
                    ),
                },
                ("command",),
            ),
        },
        {
            "name": "CronCreate",
            "description": "Schedule a prompt to be enqueued at a future time.",
            "input_schema": schema(
                {
                    "cron": field(
                        'Standard 5-field cron expression in local time: "M H DoM Mon DoW" (e.g.',
                        type="string",
                    ),
                    "prompt": field("The prompt to enqueue at each fire time.", type="string"),
                    "recurring": field(
                        "true (default) = fire on every cron match until deleted or auto-expired after 7 days.",
                        type="boolean",
                    ),
                    "durable": field(
                        "true = persist to .claude/scheduled_tasks.json and survive restarts.",
                        type="boolean",
                    ),
                },
                ("cron", "prompt"),
            ),
        },
        {
            "name": "CronDelete",
            "description": "Cancel a cron job previously scheduled with CronCreate.",
            "input_schema": schema(
                {
                    "id": field("Job ID returned by CronCreate.", type="string"),
                },
                ("id",),
            ),
        },
        {
            "name": "CronList",
            "description": "List all cron jobs scheduled via CronCreate, both durable (.claude/scheduled_tasks.json) and session-only.",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "Edit",
            "description": "Performs exact string replacements in files.",
            "input_schema": schema(
                {
                    "file_path": field("The absolute path to the file to modify", type="string"),
                    "old_string": field("The text to replace", type="string"),
                    "new_string": field(
                        "The text to replace it with (must be different from old_string)", type="string"
                    ),
                    "replace_all": field(
                        "Replace all occurrences of old_string (default false)",
                        default=False,
                        type="boolean",
                    ),
                },
                ("file_path", "old_string", "new_string"),
            ),
        },
        {
            "name": "EnterWorktree",
            "description": "Use this tool ONLY when explicitly instructed to work in a worktree — either by the user directly, or by project instruc",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "name": field("Optional name for a new worktree.", type="string"),
                    "path": field(
                        "Path to an existing worktree to switch into instead of creating a new one.",
                        type="string",
                    ),
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "ExitWorktree",
            "description": "Exit a worktree session created by EnterWorktree and return the session to the original working directory.",
            "input_schema": schema(
                {
                    "action": field(
                        '"keep" leaves the worktree and branch on disk; "remove" deletes both.',
                        type="string",
                        enum=["keep", "remove"],
                    ),
                    "discard_changes": field(
                        'Required true when action is "remove" and the worktree has uncommitted files or unmerged commits.',
                        type="boolean",
                    ),
                },
                ("action",),
            ),
        },
        {
            "name": "ListAgents",
            "description": "Lists agents you can SendMessage to — in-process subagents you spawned, the teammates on your team, other local Claude s",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "channel": field("Not available in this build; leave unset.", type="string", maxLength=256),
                    "q": field("Not available in this build; leave unset.", type="string", maxLength=256),
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "NotebookEdit",
            "description": "Replaces, inserts, or deletes a single cell in a Jupyter notebook (.ipynb file).",
            "input_schema": schema(
                {
                    "notebook_path": field(
                        "The absolute path to the Jupyter notebook file to edit (must be absolute, not relative)",
                        type="string",
                    ),
                    "cell_id": field("The ID of the cell to edit.", type="string"),
                    "new_source": field("The new source for the cell", type="string"),
                    "cell_type": field(
                        "The type of the cell (code or markdown).",
                        type="string",
                        enum=["code", "markdown"],
                    ),
                    "edit_mode": field(
                        "The type of edit to make (replace, insert, delete).",
                        type="string",
                        enum=["replace", "insert", "delete"],
                    ),
                },
                ("notebook_path", "new_source"),
            ),
        },
        {
            "name": "Read",
            "description": "Reads a file from the local filesystem.",
            "input_schema": schema(
                {
                    "file_path": field("The absolute path to the file to read", type="string"),
                    "offset": field("The line number to start reading from.", type="integer", minimum=0, maximum=_MAX),
                    "limit": field(
                        "The number of lines to read.",
                        type="integer",
                        exclusiveMinimum=0,
                        maximum=_MAX,
                    ),
                    "pages": field('Page range for PDF files (e.g., "1-5", "3", "10-20").', type="string"),
                },
                ("file_path",),
            ),
        },
        {
            "name": "ReportFindings",
            "description": "Report code-review findings as a typed list so the host UI can render them.",
            "input_schema": schema(
                {
                    "level": field(
                        "Effort level the review ran at",
                        type="string",
                        enum=["low", "medium", "high", "xhigh", "max"],
                    ),
                    "findings": field(
                        "Verified findings, most-severe first; empty if none survived",
                        maxItems=32,
                        type="array",
                        items={
                            "type": "object",
                            "properties": {
                                "file": field("Repo-relative path of the file the finding is in", type="string"),
                                "line": field(
                                    "1-indexed line the finding anchors to",
                                    type="integer",
                                    minimum=-_MAX,
                                    maximum=_MAX,
                                ),
                                "summary": field("One-sentence statement of the defect", type="string"),
                                "short_summary": field(
                                    "Compressed label for compact UI (≤60 chars): the claim alone, no rationale or consequence clause",
                                    type="string",
                                    maxLength=60,
                                ),
                                "failure_scenario": field("Concrete inputs/state → wrong output/crash", type="string"),
                                "category": field(
                                    "Short kebab-case slug of the finding type, e.g.",
                                    type="string",
                                    maxLength=40,
                                ),
                                "verdict": field(
                                    "Set when a verify pass ran; absent on inline-only reviews",
                                    type="string",
                                    enum=["CONFIRMED", "PLAUSIBLE"],
                                ),
                                "outcome": field(
                                    "Set ONLY when re-reporting after applying fixes: what happened to this finding",
                                    type="string",
                                    enum=["fixed", "skipped", "no_change_needed"],
                                ),
                            },
                            "required": ["file", "summary", "failure_scenario"],
                            "additionalProperties": False,
                        },
                    ),
                },
                ("findings",),
            ),
        },
        {
            "name": "ScheduleWakeup",
            "description": "Schedule when to resume work in /loop dynamic mode — the user invoked /loop without an interval, asking you to self-pace",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "delaySeconds": field("Seconds from now to wake up.", type="number"),
                    "reason": field("One short sentence explaining the chosen delay.", type="string"),
                    "prompt": field("The /loop input to fire on wake-up.", type="string"),
                    "stop": field(
                        "Set to true to end the dynamic loop immediately instead of scheduling another wakeup.",
                        type="boolean",
                    ),
                    "noop": field(
                        "true = nothing changed (you checked and there is nothing to report).",
                        type="boolean",
                    ),
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "SendMessage",
            "description": "# SendMessage\n\nSend a message to another agent.",
            "input_schema": schema(
                {
                    "to": field(
                        'Recipient: a name from ListAgents (append its " [ref]" only when a listing or an error shows one), a teammate name, "mai',
                        type="string",
                        allOf=[{"pattern": "^[^\\n\\r]*$"}, {"pattern": "^[\\s\\S]{0,300}$"}],
                    ),
                    "summary": field(
                        "A 5-10 word label for your own transcript row (not transmitted — the recipient previews the first line of `message`).",
                        type="string",
                        maxLength=200,
                    ),
                    "message": field("Plain text message content.", default="", type="string"),
                    "notify_when_idle": field(
                        "Ask a session ON THIS MACHINE to send you ONE notice when it next goes idle (finishes its turn with nothing queued) or e",
                        type="boolean",
                    ),
                },
                ("to", "message"),
            ),
        },
        {
            "name": "Skill",
            "description": "Invoke a skill.",
            "input_schema": schema(
                {
                    "skill": field("The name of a skill from the available-skills list.", type="string"),
                    "args": field("Optional arguments for the skill", type="string"),
                },
                ("skill",),
            ),
        },
        {
            "name": "TaskCreate",
            "description": "Use this tool to create a structured task list for your current coding session.",
            "input_schema": schema(
                {
                    "subject": field("A brief title for the task", type="string"),
                    "description": field("What needs to be done", type="string"),
                    "activeForm": field(
                        'Present continuous form shown in spinner when in_progress (e.g., "Running tests")',
                        type="string",
                    ),
                    "metadata": field(
                        "Arbitrary metadata to attach to the task",
                        type="object",
                        propertyNames={"type": "string"},
                        additionalProperties={},
                    ),
                },
                ("subject", "description"),
            ),
        },
        {
            "name": "TaskGet",
            "description": "Use this tool to retrieve a task by its ID from the task list.",
            "input_schema": schema(
                {
                    "taskId": field("The ID of the task to retrieve", type="string"),
                },
                ("taskId",),
            ),
        },
        {
            "name": "TaskList",
            "description": "Use this tool to list all tasks in the task list.",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "TaskStop",
            "description": "- Stops a running background task by its ID\n- Takes a task_id parameter identifying the task to stop\n- To stop an agent-",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "task_id": field("The ID of the background task to stop.", type="string"),
                    "shell_id": field("Deprecated: use task_id instead", type="string"),
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "TaskUpdate",
            "description": "Use this tool to update a task in the task list.",
            "input_schema": schema(
                {
                    "taskId": field("The ID of the task to update", type="string"),
                    "subject": field("New subject for the task", type="string"),
                    "description": field("New description for the task", type="string"),
                    "activeForm": field(
                        'Present continuous form shown in spinner when in_progress (e.g., "Running tests")',
                        type="string",
                    ),
                    "status": field(
                        "New status for the task",
                        anyOf=[
                            {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                            {"type": "string", "const": "deleted"},
                        ],
                    ),
                    "addBlocks": field("Task IDs that this task blocks", type="array", items={"type": "string"}),
                    "addBlockedBy": field("Task IDs that block this task", type="array", items={"type": "string"}),
                    "owner": field("New owner for the task", type="string"),
                    "metadata": field(
                        "Metadata keys to merge into the task.",
                        type="object",
                        propertyNames={"type": "string"},
                        additionalProperties={},
                    ),
                },
                ("taskId",),
            ),
        },
        {
            "name": "WebFetch",
            "description": "IMPORTANT: WebFetch WILL FAIL for authenticated or private URLs.",
            "input_schema": schema(
                {
                    "url": field("The URL to fetch content from", type="string", format="uri"),
                    "prompt": field("The prompt to run on the fetched content", type="string"),
                },
                ("url", "prompt"),
            ),
        },
        {
            "name": "WebSearch",
            "description": "- Allows Claude to search the web and use the results to inform responses\n- Provides up-to-date information for current ",
            "input_schema": schema(
                {
                    "query": field("The search query to use", type="string", minLength=2),
                    "allowed_domains": field(
                        "Only include search results from these domains", type="array", items={"type": "string"}
                    ),
                    "blocked_domains": field(
                        "Never include search results from these domains", type="array", items={"type": "string"}
                    ),
                },
                ("query",),
            ),
        },
        {
            "name": "Workflow",
            "description": "Execute a workflow script that orchestrates multiple subagents deterministically.",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "script": field("Self-contained workflow script.", type="string", maxLength=524288),
                    "name": field(
                        "Name of a predefined workflow (built-in or from .claude/workflows/).", type="string"
                    ),
                    "description": field(
                        "Ignored — set the workflow description in the script's `meta` block.", type="string"
                    ),
                    "title": field("Ignored — set the workflow title in the script's `meta` block.", type="string"),
                    "args": field("Optional input value exposed to the script as the global `args`, verbatim."),
                    "scriptPath": field("Path to a workflow script file on disk.", type="string"),
                    "resumeFromRunId": field(
                        "Run ID of a prior Workflow invocation to resume from.",
                        type="string",
                        pattern="^wf_[a-z0-9-]{6,}$",
                    ),
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "Write",
            "description": "Writes a file to the local filesystem.",
            "input_schema": schema(
                {
                    "file_path": field(
                        "The absolute path to the file to write (must be absolute, not relative)", type="string"
                    ),
                    "content": field("The content to write to the file", type="string"),
                },
                ("file_path", "content"),
            ),
        },
    )


def system_blocks() -> tuple[dict[str, JsonValue], ...]:
    return (
        {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.283.00; cc_entrypoint=sdk-cli;"},
        {"type": "text", "text": "Synthetic agent identity system prompt.", "cache_control": CACHE},
        {"type": "text", "text": "Synthetic interactive agent instructions.", "cache_control": CACHE},
    )


def claude_code_request(cache_bust: str) -> dict[str, JsonValue]:
    reminders: Final = (
        f"<system-reminder>\n{cache_bust}\n</system-reminder>",
        "<system-reminder>\nSynthetic model identity reminder.\n</system-reminder>",
        "<system-reminder>\nSynthetic agent types reminder.\n</system-reminder>",
        "<system-reminder>\nSynthetic skills reminder.\n</system-reminder>",
        "<system-reminder>\n<total_tokens>15000000 tokens left</total_tokens>\n</system-reminder>",
        "<system-reminder>\nSynthetic date reminder.\n</system-reminder>",
        "<system-reminder>\nSynthetic attribution reminder.\n</system-reminder>",
    )
    return {
        "model": "",
        "system": list(system_blocks()),
        "messages": [
            {
                "role": "user",
                "content": [
                    *[{"type": "text", "text": reminder} for reminder in reminders],
                    {"type": "text", "text": "Reply with exactly the word PONG", "cache_control": CACHE},
                ],
            }
        ],
        "tools": list(tools()),
        "metadata": {"user_id": METADATA_USER_ID},
        "max_tokens": 32000,
        "thinking": dict(THINKING_BUDGET),
        "context_management": dict(CONTEXT_MANAGEMENT),
        "stream": True,
    }


def frontier_request(
    cache_bust: str,
    effort: str,
    max_tokens: int,
    prompt_text: str = "Reply with exactly the word PONG",
    stream: bool = True,
) -> dict[str, JsonValue]:
    return {
        "model": "",
        "system": list(system_blocks()),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<system-reminder>\n{cache_bust}\n</system-reminder>"},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "# Environment\nSynthetic environment block.",
                        "cache_control": CACHE,
                    }
                ],
            },
        ],
        "tools": list(tools()),
        "metadata": {"user_id": METADATA_USER_ID},
        "max_tokens": max_tokens,
        "thinking": dict(THINKING_ADAPTIVE),
        "context_management": dict(CONTEXT_MANAGEMENT),
        "output_config": {"effort": effort},
        "stream": stream,
    }


def tool_loop_turn2(
    base: dict[str, JsonValue],
    assistant_content: tuple[dict[str, JsonValue], ...],
    tool_results: tuple[tuple[str, JsonValue], ...],
) -> dict[str, JsonValue]:
    return {
        **base,
        "messages": [
            *base["messages"],
            {"role": "assistant", "content": list(assistant_content)},
            {
                "role": "user",
                "content": [
                    {"tool_use_id": tool_use_id, "type": "tool_result", "content": content}
                    for tool_use_id, content in tool_results
                ],
            },
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "<total_tokens>14999970 tokens left</total_tokens>",
                        "cache_control": CACHE,
                    },
                    {
                        "type": "text",
                        "text": "First privately list what you need next; then request every item that doesn't depend on another's result in this one response.",
                    },
                ],
            },
        ],
    }


def cli_headers(key: str, beta: str = CLI_BETA) -> dict[str, str]:
    return {
        "accept": "application/json",
        "content-type": "application/json",
        "user-agent": "claude-cli/2.1.283 (external, sdk-cli)",
        "x-claude-code-session-id": "00000000-0000-4000-8000-000000000000",
        "x-stainless-arch": "x64",
        "x-stainless-lang": "js",
        "x-stainless-os": "Linux",
        "x-stainless-package-version": "0.112.1",
        "x-stainless-retry-count": "0",
        "x-stainless-runtime": "node",
        "x-stainless-runtime-version": "v26.3.0",
        "x-stainless-timeout": "600",
        "anthropic-beta": beta,
        "anthropic-dangerous-direct-browser-access": "true",
        "anthropic-version": "2023-06-01",
        "x-app": "cli",
        "x-api-key": key,
    }


def sse_frame(event: str, data: JsonValue) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def sse_events(text: str) -> tuple[tuple[str, dict[str, object]], ...]:
    frames: Final = tuple(frame for frame in text.split("\n\n") if frame.strip())
    return tuple(
        (
            event,
            json.loads(next(line.removeprefix("data: ") for line in frame.splitlines() if line.startswith("data: "))),
        )
        for frame in frames
        if (event := next(line.removeprefix("event: ") for line in frame.splitlines() if line.startswith("event: ")))
        != "ping"
    )


def reasoning_betas(anthropic_beta: str) -> tuple[str, ...]:
    return tuple(sorted(beta for beta in anthropic_beta.split(",") if beta in CLAUDE_CODE_REASONING_BETAS))


def body_diff(expected: Mapping[str, JsonValue], body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        key: {"expected": expected.get(key), "upstream": body.get(key)}
        for key in expected.keys() | body.keys()
        if expected.get(key) != body.get(key)
    }


@dataclass(frozen=True, slots=True)
class Forwarded:
    model: JsonValue
    reasoning: dict[str, JsonValue]
    assistant_history: tuple[JsonValue, ...]
    other_changes: dict[str, JsonValue]
    reasoning_betas: tuple[str, ...]


def _is_assistant_turn(message: JsonValue) -> bool:
    return isinstance(message, dict) and message.get("role") == "assistant"


def _assistant_history(body: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    messages: Final = body.get("messages")
    if not isinstance(messages, list):
        return ()
    return tuple(
        message["content"] for message in messages if isinstance(message, dict) and _is_assistant_turn(message)
    )


def _without_assistant_turns(messages: JsonValue) -> JsonValue:
    if not isinstance(messages, list):
        return messages
    return [message for message in messages if not _is_assistant_turn(message)]


def _unrelated_fields(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    excluded: Final = frozenset({*REASONING_FIELDS, "model"})
    return {
        key: _without_assistant_turns(value) if key == "messages" else value
        for key, value in body.items()
        if key not in excluded
    }


def forwarded(sent: Mapping[str, JsonValue], request: Request) -> Forwarded:
    body: Final = JSON_OBJECT.validate_json(request.body)
    return Forwarded(
        model=body.get("model"),
        reasoning={field: body[field] for field in REASONING_FIELDS if field in body},
        assistant_history=_assistant_history(body),
        other_changes=body_diff(_unrelated_fields(sent), _unrelated_fields(body)),
        reasoning_betas=reasoning_betas(request.headers.get("anthropic-beta", "")),
    )


@dataclass(frozen=True, slots=True)
class CacheForwarded:
    breakpoints: dict[str, JsonValue]
    other_changes: dict[str, JsonValue]
    caching_betas: tuple[str, ...]


def _marked(prefix: str, blocks: JsonValue) -> tuple[tuple[str, JsonValue], ...]:
    if not isinstance(blocks, list):
        return ()
    return tuple(
        (f"{prefix}[{index}]", block["cache_control"])
        for index, block in enumerate(blocks)
        if isinstance(block, dict) and "cache_control" in block
    )


def _marked_message(index: int, message: JsonValue) -> tuple[tuple[str, JsonValue], ...]:
    return _marked(f"messages[{index}].content", message.get("content")) if isinstance(message, dict) else ()


def _marked_messages(messages: JsonValue) -> tuple[tuple[str, JsonValue], ...]:
    if not isinstance(messages, list):
        return ()
    return tuple(chain.from_iterable(_marked_message(index, message) for index, message in enumerate(messages)))


def cache_breakpoints(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    top_level: Final = (("cache_control", body["cache_control"]),) if "cache_control" in body else ()
    return dict(
        (
            *top_level,
            *_marked("system", body.get("system")),
            *_marked("tools", body.get("tools")),
            *_marked_messages(body.get("messages")),
        )
    )


def _unmarked_block(block: JsonValue) -> JsonValue:
    if not isinstance(block, dict):
        return block
    return {key: value for key, value in block.items() if key != "cache_control"}


def _unmarked_blocks(blocks: JsonValue) -> JsonValue:
    return [_unmarked_block(block) for block in blocks] if isinstance(blocks, list) else blocks


def _unmarked_message(message: JsonValue) -> JsonValue:
    if not isinstance(message, dict) or "content" not in message:
        return message
    return {**message, "content": _unmarked_blocks(message["content"])}


def _unmarked_value(key: str, value: JsonValue) -> JsonValue:
    match key, value:
        case ("system" | "tools", _):
            return _unmarked_blocks(value)
        case ("messages", list()):
            return [_unmarked_message(message) for message in value]
        case _:
            return value


def with_block_breakpoint(
    body: Mapping[str, JsonValue], field: str, index: int, control: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    blocks: Final = body[field]
    assert isinstance(blocks, list), body
    return {
        **body,
        field: [
            {**block, "cache_control": dict(control)} if position == index and isinstance(block, dict) else block
            for position, block in enumerate(blocks)
        ],
    }


def without_cache_breakpoints(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: _unmarked_value(key, value) for key, value in body.items() if key != "cache_control"}


def caching_betas(anthropic_beta: str) -> tuple[str, ...]:
    return tuple(sorted(beta for beta in anthropic_beta.split(",") if beta in CLAUDE_CODE_CACHING_BETAS))


def _without_model(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in without_cache_breakpoints(body).items() if key != "model"}


def cache_forwarded(sent: Mapping[str, JsonValue], request: Request) -> CacheForwarded:
    body: Final = JSON_OBJECT.validate_json(request.body)
    return CacheForwarded(
        breakpoints=cache_breakpoints(body),
        other_changes=body_diff(_without_model(sent), _without_model(body)),
        caching_betas=caching_betas(request.headers.get("anthropic-beta", "")),
    )


def streamed_start_usage(stream: str) -> JsonValue:
    return next(
        JSON_OBJECT.validate_python(data["message"]).get("usage")
        for event, data in _client_events(stream)
        if event == "message_start"
    )


def _appended(block: Mapping[str, JsonValue], key: str, text: str) -> dict[str, JsonValue]:
    return {**block, key: f"{block.get(key) or ''}{text}"}


def _with_delta(block: dict[str, JsonValue], delta: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    match delta:
        case {"type": "thinking_delta", "thinking": str(text)}:
            return _appended(block, "thinking", text)
        case {"type": "signature_delta", "signature": str(text)}:
            return _appended(block, "signature", text)
        case {"type": "text_delta", "text": str(text)}:
            return _appended(block, "text", text)
        case {"type": "input_json_delta", "partial_json": str(text)}:
            return _appended(block, "partial_json", text)
        case _:
            return block


def _with_event(
    blocks: tuple[dict[str, JsonValue], ...], event: tuple[str, dict[str, JsonValue]]
) -> tuple[dict[str, JsonValue], ...]:
    match event:
        case ("content_block_start", {"content_block": dict() as block}):
            return (*blocks, JSON_OBJECT.validate_python(block))
        case ("content_block_delta", {"index": int(index), "delta": dict() as delta}):
            return (
                *blocks[:index],
                _with_delta(blocks[index], JSON_OBJECT.validate_python(delta)),
                *blocks[index + 1 :],
            )
        case _:
            return blocks


def _finished(block: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    partial_json: Final = block.get("partial_json")
    if not isinstance(partial_json, str):
        return dict(block)
    return {**{key: value for key, value in block.items() if key != "partial_json"}, "input": json.loads(partial_json)}


def _client_events(stream: str) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    return tuple((event, JSON_OBJECT.validate_python(data)) for event, data in sse_events(stream))


def _stopped_indices(events: tuple[tuple[str, dict[str, JsonValue]], ...]) -> frozenset[JsonValue]:
    return frozenset(data.get("index") for event, data in events if event == "content_block_stop")


def streamed_content(stream: str) -> list[dict[str, JsonValue]]:
    events: Final = _client_events(stream)
    stopped: Final = _stopped_indices(events)
    blocks: Final = reduce(_with_event, events, ())
    return [_finished(block) for index, block in enumerate(blocks) if index in stopped]


def streamed_usage(stream: str) -> JsonValue:
    return next(data.get("usage") for event, data in reversed(_client_events(stream)) if event == "message_delta")


def _start_usage(usage: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in usage.items() if key not in _OUTPUT_USAGE_KEYS}


def _delta_usage(usage: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in usage.items() if key in _OUTPUT_USAGE_KEYS}


def _block_start(index: int, block: Mapping[str, JsonValue]) -> bytes:
    return sse_frame("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})


def _block_delta(index: int, delta: Mapping[str, JsonValue]) -> bytes:
    return sse_frame("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})


def _block_stop(index: int) -> bytes:
    return sse_frame("content_block_stop", {"type": "content_block_stop", "index": index})


def _block_frames(index: int, block: Mapping[str, JsonValue]) -> tuple[bytes, ...]:
    match block.get("type"):
        case "thinking":
            return (
                _block_start(index, {"type": "thinking", "thinking": "", "signature": ""}),
                _block_delta(index, {"type": "thinking_delta", "thinking": block["thinking"]}),
                _block_delta(index, {"type": "signature_delta", "signature": block["signature"]}),
                _block_stop(index),
            )
        case "text":
            return (
                _block_start(index, {"type": "text", "text": ""}),
                _block_delta(index, {"type": "text_delta", "text": block["text"]}),
                _block_stop(index),
            )
        case _:
            return (_block_start(index, block), _block_stop(index))


def message_reply(
    identity: str, model: str, content: tuple[dict[str, JsonValue], ...], usage: dict[str, JsonValue]
) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": list(content),
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": usage,
        }
    ).encode()


def message_stream(
    identity: str,
    model: str,
    content: tuple[dict[str, JsonValue], ...],
    usage: dict[str, JsonValue],
    final_usage: Mapping[str, JsonValue] | None = None,
) -> tuple[bytes, ...]:
    start: Final = sse_frame(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": identity,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": _start_usage(usage),
            },
        },
    )
    delta: Final = sse_frame(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": _delta_usage(usage) if final_usage is None else dict(final_usage),
        },
    )
    blocks: Final = chain.from_iterable(_block_frames(index, block) for index, block in enumerate(content))
    return (start, *blocks, delta, sse_frame("message_stop", {"type": "message_stop"}))


def text_stream(identity: str, model: str, text: str, usage: dict[str, int]) -> tuple[bytes, ...]:
    return (
        sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": _start_usage(usage),
                },
            },
        ),
        sse_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        ),
        sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        ),
        sse_frame("message_stop", {"type": "message_stop"}),
    )


def _tool_use_frames(index: int, tool_id: str, name: str, tool_input: JsonValue) -> tuple[bytes, ...]:
    arguments: Final = json.dumps(tool_input)
    return (
        sse_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
            },
        ),
        sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": arguments[: len(arguments) // 2]},
            },
        ),
        sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": arguments[len(arguments) // 2 :]},
            },
        ),
        sse_frame("content_block_stop", {"type": "content_block_stop", "index": index}),
    )


def tool_use_stream(
    identity: str,
    model: str,
    thinking: str,
    signature: str,
    tool_calls: tuple[tuple[str, str, JsonValue], ...],
    usage: dict[str, int],
) -> tuple[bytes, ...]:
    head: Final = (
        sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": _start_usage(usage),
                },
            },
        ),
        sse_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        ),
        sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": thinking}},
        ),
        sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": signature}},
        ),
        sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
    )
    tail: Final = (
        sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        ),
        sse_frame("message_stop", {"type": "message_stop"}),
    )
    frames: Final = (
        *head,
        *chain.from_iterable(
            _tool_use_frames(index, tool_id, name, tool_input)
            for index, (tool_id, name, tool_input) in enumerate(tool_calls, start=1)
        ),
        *tail,
    )
    return frames
