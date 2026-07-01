"""Tests for MicroloopGuardrail — deterministic loop detection for LiteLLM."""

from unittest.mock import MagicMock

import pytest

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
