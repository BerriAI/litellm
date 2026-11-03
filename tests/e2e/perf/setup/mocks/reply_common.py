"""What every generated provider reply is made of: a `Reply` sized from the request (prompt tokens, the
output cap, whether tools were offered), deterministic ids and text drawn from the request marker, and the
chunking that makes a stream arrive the way the provider's TCP writes do. The provider modules render a
`Reply` in their own wire format; nothing here is provider specific."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from itertools import accumulate, groupby
from typing import Final, Literal

from pydantic import TypeAdapter

Provider = Literal["openai", "anthropic", "bedrock"]
CHARS_PER_TOKEN: Final = 4
CACHED_PREFIX_TOKENS: Final = 1024
"""Prompt tokens the provider reports as a cache hit (system prompt and tool set) on the requests that hit."""
CACHE_HIT_SHARE: Final = 0.7
REASONING_SHARE: Final = 0.6
"""Share of the output spent on reasoning when the request offered tools, as the captured replies did."""
_BASE62: Final = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_DIRS: Final = (
    "management",
    "guardrails",
    "llm_translation",
    "router",
    "quota_management",
    "mcp",
    "logging",
    "security",
)
_WORDS: Final = (
    "spend", "budget", "team", "key", "fallback", "cooldown", "retry", "stream", "cache", "tag",
    "alias", "quota", "webhook", "secret", "model", "org", "audit", "rerank", "batch", "realtime",
)  # fmt: skip
_THOUGHT: Final = "Checking {path}: the except clause around the client call logs and returns None, so the caller never sees the error. "


def unit(seed: str) -> float:
    """A float in [0, 1) that is the same for the same seed on every mock pod."""
    digest: Final = hashlib.sha1(seed.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def pick(seed: str, index: int, size: int) -> int:
    return int(hashlib.sha1(f"{seed}:{index}".encode()).hexdigest()[:8], 16) % size


def ident(seed: str, prefix: str, length: int, *, hex_digits: bool = False) -> str:
    """A provider-shaped id (`chatcmpl-...`, `msg_...`, `toolu_...`) that is a pure function of the seed."""
    digest: Final = hashlib.sha256(f"{seed}:{prefix}".encode())
    if hex_digits:
        return prefix + digest.hexdigest()[:length]
    return prefix + "".join(_BASE62[byte % len(_BASE62)] for byte in digest.digest()[:length])


def file_paths(seed: str, count: int) -> tuple[str, ...]:
    return tuple(
        f"tests/e2e/{_DIRS[pick(seed, i, len(_DIRS))]}/test_{_WORDS[pick(seed, i + 1000, len(_WORDS))]}"
        f"_{_WORDS[pick(seed, i + 2000, len(_WORDS))]}_e2e.py"
        for i in range(count)
    )


def answer_text(seed: str, tokens: int) -> str:
    """The assistant's answer to the corpus question (which files swallow exceptions): paths, one per line."""
    return "\n".join(file_paths(seed, max(1, tokens * CHARS_PER_TOKEN // 56)))


def thinking_text(seed: str, tokens: int) -> str:
    if tokens == 0:
        return ""
    paths: Final = file_paths(f"{seed}:thinking", max(1, tokens * CHARS_PER_TOKEN // len(_THOUGHT)))
    return "".join(_THOUGHT.format(path=path) for path in paths).rstrip()


def grep_arguments(seed: str, tokens: int) -> str:
    """A `Grep` call over the files the session read, its glob sized to the output budget."""
    paths: Final = file_paths(f"{seed}:grep", max(1, tokens * CHARS_PER_TOKEN // 56))
    arguments: Final = {
        "pattern": r"\braise\s+\w+",
        "path": "/workspace/litellm/tests/e2e",
        "glob": "{" + ",".join(paths) + "}",
    }
    return json.dumps(arguments, separators=(",", ":"))


_ARGUMENTS: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: str

    @property
    def parsed_arguments(self) -> dict[str, object]:
        return _ARGUMENTS.validate_json(self.arguments)


@dataclass(frozen=True, slots=True)
class Reply:
    """One generated reply, every number the gateway will read already decided."""

    provider: Provider
    seed: str
    model: str
    prompt_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    output_tokens: int
    reasoning: str
    text: str
    tool_call: ToolCall | None

    @property
    def visible_tokens(self) -> int:
        return self.output_tokens - self.reasoning_tokens

    @property
    def uncached_tokens(self) -> int:
        return self.prompt_tokens - self.cached_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


def plan_reply(
    *, provider: Provider, seed: str, model: str, prompt_tokens: int, max_tokens: int, tool_names: Sequence[str]
) -> Reply:
    """Size the reply from the request: a quarter to three quarters of the output cap, a tool call (with
    reasoning first) when tools were offered, otherwise a text answer, a prefix cache hit on most requests."""
    output_tokens: Final = max(32, int(max_tokens * (0.25 + 0.5 * unit(f"{seed}:length"))))
    tool: Final = "Grep" if "Grep" in tool_names else next(iter(tool_names), None)
    reasoning_tokens: Final = int(output_tokens * REASONING_SHARE) if tool is not None else 0
    visible: Final = output_tokens - reasoning_tokens
    cached: Final = (
        CACHED_PREFIX_TOKENS if unit(f"{seed}:cache") < CACHE_HIT_SHARE and prompt_tokens > CACHED_PREFIX_TOKENS else 0
    )
    return Reply(
        provider=provider,
        seed=seed,
        model=model,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached,
        reasoning_tokens=reasoning_tokens,
        output_tokens=output_tokens,
        reasoning=thinking_text(seed, reasoning_tokens),
        text="" if tool is not None else answer_text(seed, visible),
        tool_call=ToolCall(tool, grep_arguments(seed, visible)) if tool is not None else None,
    )


def pieces(text: str, size: int) -> tuple[str, ...]:
    """`text` cut into the deltas a provider streams it as, `size` characters each."""
    return tuple(text[i : i + size] for i in range(0, len(text), size))


def sse(payload: dict[str, object], event: str | None = None) -> bytes:
    data: Final = json.dumps(payload, separators=(",", ":"))
    prefix: Final = f"event: {event}\n" if event is not None else ""
    return f"{prefix}data: {data}\n\n".encode()


def grouped(events: Sequence[bytes], target_bytes: int) -> tuple[bytes, ...]:
    """Consecutive events joined until a chunk reaches `target_bytes`, the size the provider's writes arrive in."""
    cumulative: Final = tuple(accumulate(len(event) for event in events))
    keyed: Final = tuple(zip((total // target_bytes for total in cumulative), events, strict=True))
    return tuple(b"".join(event for _, event in group) for _, group in groupby(keyed, key=lambda pair: pair[0]))


async def chunks_after(chunks: Sequence[bytes], delay_s: float) -> AsyncIterator[bytes]:
    """The chunks back to back once `delay_s` has passed."""
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    for chunk in chunks:
        yield chunk
