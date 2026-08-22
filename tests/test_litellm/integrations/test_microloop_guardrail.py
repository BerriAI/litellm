"""Tests for MicroloopGuardrail — deterministic loop detection for LiteLLM."""

import sys
from unittest.mock import MagicMock

import pytest

# ── Mock the Rust engine for CI coverage ──────────────────────────────────────
# The CI environment does not have the `microloop` Rust binary installed.
# We mock it here so the guardrail code executes and Codecov registers coverage.
#
# The mock implements a faithful Python port of the Rust engine's stateful
# loop-detection logic so the integration tests are meaningful even when
# the native binary is absent.

import json
from typing import Any


class _MockMicroloopEngine:
    """Stateful mock that replicates the Microloop Rust engine's loop detection."""

    def __init__(self, config_yaml: str) -> None:
        self._history: list[tuple[str, str]] = []
        self.max_repeats = 3
        self._window: int | None = None
        self.volatile_fields: list[str] = []

        for line in config_yaml.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key == "max_repeats":
                    self.max_repeats = int(val)
                elif key == "history_window":
                    self._window = int(val)
                elif key == "volatile_fields":
                    self.volatile_fields = json.loads(val)

    def _canonicalize(self, raw_args: str) -> str:
        try:
            d: dict[str, Any] = json.loads(raw_args)
            for f in self.volatile_fields:
                d.pop(f, None)
            return json.dumps(d, sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            return raw_args

    def verify(self, tool_name: str, raw_args: str) -> int:
        args = self._canonicalize(raw_args)

        count = 1
        for past_tool, past_args in self._history:
            if past_tool == tool_name and past_args == args:
                count += 1

        self._history.append((tool_name, args))
        if self._window is not None and len(self._history) > self._window:
            self._history.pop(0)

        return 0 if count < self.max_repeats else 1


_mock_microloop_module = MagicMock()
_mock_microloop_module.Microloop.side_effect = _MockMicroloopEngine
# ──────────────────────────────────────────────────────────────────────────────

# The conftest eagerly imports `litellm` (and thus all integrations via
# integrations/__init__.py's `from . import *`), so the module-level
# `from microloop import Microloop` already ran with MICROLOOP_AVAILABLE = False.
# Re-inject both the binding and the flag so the guardrail code path is exercised.
import litellm.integrations.microloop_guardrail as _mg_module

_mg_module.Microloop = _MockMicroloopEngine
_mg_module.MICROLOOP_AVAILABLE = True

from litellm.integrations.microloop_guardrail import MicroloopGuardrail
from litellm.exceptions import GuardrailRaisedException


@pytest.fixture
def call_type() -> MagicMock:
    ct = MagicMock()
    ct.value = "completion"
    return ct


@pytest.fixture
def user() -> MagicMock:
    return MagicMock(api_key="microloop-test-key")


class TestMicroloopGuardrail:
    """Test suite for MicroloopGuardrail deterministic loop detection."""

    @pytest.mark.asyncio
    async def test_blocks_identical_tool_calls(self, call_type, user):
        guardrail = MicroloopGuardrail(max_repeats=3)
        data = {
            "metadata": {"session_id": "blocks-identical"},
            "messages": [
                {"role": "user", "content": "search for X"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "1",
                            "function": {
                                "name": "search",
                                "arguments": '{"q":"X"}',
                            },
                        }
                    ],
                },
            ]
        }
        await guardrail.async_pre_call_hook(
            user_api_key_dict=user,
            cache=MagicMock(),
            data=data,
            call_type=call_type,
        )
        await guardrail.async_pre_call_hook(
            user_api_key_dict=user,
            cache=MagicMock(),
            data=data,
            call_type=call_type,
        )
        with pytest.raises(GuardrailRaisedException):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user,
                cache=MagicMock(),
                data=data,
                call_type=call_type,
            )

    @pytest.mark.asyncio
    async def test_allows_unique_tool_calls(self, call_type, user):
        guardrail = MicroloopGuardrail(max_repeats=3)
        for q in ["X", "Y", "Z"]:
            data = {
                "messages": [
                    {"role": "user", "content": f"search for {q}"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "1",
                                "function": {
                                    "name": "search",
                                    "arguments": f'{{"q":"{q}"}}',
                                },
                            }
                        ],
                    },
                ]
            }
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user,
                cache=MagicMock(),
                data=data,
                call_type=call_type,
            )

    @pytest.mark.asyncio
    async def test_volatile_field_stripping(self, call_type, user):
        guardrail = MicroloopGuardrail(max_repeats=3, volatile_fields=["req_id"])
        for rid in [1, 2]:
            data = {
                "metadata": {"session_id": "volatile-test"},
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "1",
                                "function": {
                                    "name": "search",
                                    "arguments": (f'{{"q":"X","req_id":{rid}}}'),
                                },
                            }
                        ],
                    },
                ]
            }
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user,
                cache=MagicMock(),
                data=data,
                call_type=call_type,
            )
        with pytest.raises(GuardrailRaisedException):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user,
                cache=MagicMock(),
                data={
                    "metadata": {"session_id": "volatile-test"},
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "1",
                                    "function": {
                                        "name": "search",
                                        "arguments": ('{"q":"X","req_id":4}'),
                                    },
                                }
                            ],
                        },
                    ]
                },
                call_type=call_type,
            )

    @pytest.mark.asyncio
    async def test_auto_inference_does_not_create_false_positives(self, call_type, user):
        guardrail = MicroloopGuardrail(max_repeats=3, auto_infer_volatile=True)
        for q, rid in [("A", 1), ("B", 2), ("C", 3)]:
            data = {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "1",
                                "function": {
                                    "name": "search",
                                    "arguments": (f'{{"q":"{q}","rid":{rid}}}'),
                                },
                            }
                        ],
                    },
                ]
            }
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user,
                cache=MagicMock(),
                data=data,
                call_type=call_type,
            )

    @pytest.mark.asyncio
    async def test_session_isolation(self, call_type, user):
        guardrail = MicroloopGuardrail(max_repeats=2)
        base_messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {
                            "name": "search",
                            "arguments": '{"q":"X"}',
                        },
                    }
                ],
            },
        ]
        await guardrail.async_pre_call_hook(
            user_api_key_dict=user,
            cache=MagicMock(),
            data={
                "metadata": {"session_id": "s1"},
                "messages": base_messages,
            },
            call_type=call_type,
        )
        await guardrail.async_pre_call_hook(
            user_api_key_dict=user,
            cache=MagicMock(),
            data={
                "metadata": {"session_id": "s2"},
                "messages": base_messages,
            },
            call_type=call_type,
        )

    @pytest.mark.asyncio
    async def test_anthropic_tool_use_format(self, call_type, user):
        guardrail = MicroloopGuardrail(max_repeats=2)
        data = {
            "metadata": {"session_id": "anthropic-test"},
            "messages": [
                {"role": "user", "content": "search"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "search", "input": {"q": "X"}}],
                },
            ]
        }
        await guardrail.async_pre_call_hook(
            user_api_key_dict=user,
            cache=MagicMock(),
            data=data,
            call_type=call_type,
        )
        with pytest.raises(GuardrailRaisedException):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user,
                cache=MagicMock(),
                data=data,
                call_type=call_type,
            )

    @pytest.mark.asyncio
    async def test_hook_blocks_repeated_loop(self, call_type, user):
        guardrail = MicroloopGuardrail(max_repeats=2)
        data = {
            "metadata": {"session_id": "hook-loop"},
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "1",
                            "function": {
                                "name": "search",
                                "arguments": '{"q":"X"}',
                            },
                        }
                    ],
                },
            ]
        }
        await guardrail.async_pre_call_hook(
            user_api_key_dict=user,
            cache=MagicMock(),
            data=data,
            call_type=call_type,
        )
        with pytest.raises(GuardrailRaisedException):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user,
                cache=MagicMock(),
                data=data,
                call_type=call_type,
            )

    @pytest.mark.asyncio
    async def test_no_false_positive_when_query_changes(self, call_type, user):
        guardrail = MicroloopGuardrail(max_repeats=3)
        for q in ["A", "B", "C"]:
            data = {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "1",
                                "function": {
                                    "name": "search",
                                    "arguments": f'{{"q":"{q}"}}',
                                },
                            }
                        ],
                    },
                ]
            }
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user,
                cache=MagicMock(),
                data=data,
                call_type=call_type,
            )

    @pytest.mark.asyncio
    async def test_respects_history_window(self, call_type, user):
        guardrail = MicroloopGuardrail(max_repeats=3, history_window=2)
        await guardrail.async_pre_call_hook(
            user_api_key_dict=user,
            cache=MagicMock(),
            data={
                "metadata": {"session_id": "history-window-test"},
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "1",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":"X"}',
                                },
                            }
                        ],
                    },
                ]
            },
            call_type=call_type,
        )
        await guardrail.async_pre_call_hook(
            user_api_key_dict=user,
            cache=MagicMock(),
            data={
                "metadata": {"session_id": "history-window-test"},
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "1",
                                "function": {
                                    "name": "read",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                ]
            },
            call_type=call_type,
        )
        await guardrail.async_pre_call_hook(
            user_api_key_dict=user,
            cache=MagicMock(),
            data={
                "metadata": {"session_id": "history-window-test"},
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "1",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":"X"}',
                                },
                            }
                        ],
                    },
                ]
            },
            call_type=call_type,
        )
        await guardrail.async_pre_call_hook(
            user_api_key_dict=user,
            cache=MagicMock(),
            data={
                "metadata": {"session_id": "history-window-test"},
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "1",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":"X"}',
                                },
                            }
                        ],
                    },
                ]
            },
            call_type=call_type,
        )
        with pytest.raises(GuardrailRaisedException):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user,
                cache=MagicMock(),
                data={
                    "metadata": {"session_id": "history-window-test"},
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "1",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"q":"X"}',
                                    },
                                }
                            ],
                        },
                    ]
                },
                call_type=call_type,
            )

    @pytest.mark.asyncio
    async def test_rejects_max_repeats_one(self, call_type, user):
        with pytest.raises(ValueError, match="max_repeats must be at least 2"):
            MicroloopGuardrail(max_repeats=1)

    @pytest.mark.asyncio
    async def test_canonicalization_blocks_reordered_json(self, call_type, user):
        guardrail = MicroloopGuardrail(max_repeats=2)
        data_a = {
            "metadata": {"session_id": "canon-test"},
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "1",
                            "function": {
                                "name": "search",
                                "arguments": '{"a":1,"b":2}',
                            },
                        }
                    ],
                },
            ]
        }
        data_b = {
            "metadata": {"session_id": "canon-test"},
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "2",
                            "function": {
                                "name": "search",
                                "arguments": '{"b":2,"a":1}',
                            },
                        }
                    ],
                },
            ]
        }
        await guardrail.async_pre_call_hook(
            user_api_key_dict=user,
            cache=MagicMock(),
            data=data_a,
            call_type=call_type,
        )
        with pytest.raises(GuardrailRaisedException):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user,
                cache=MagicMock(),
                data=data_b,
                call_type=call_type,
            )
