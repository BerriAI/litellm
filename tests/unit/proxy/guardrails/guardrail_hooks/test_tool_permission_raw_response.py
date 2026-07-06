import copy
import json
from typing import Final, Literal

import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.tool_permission import ToolPermissionGuardrail
from litellm.types.guardrails import GuardrailEventHooks

_Format = Literal["openai", "legacy", "gemini", "bedrock"]


def _response(format: _Format, names: tuple[str, ...], with_ids: bool = True) -> dict[str, object]:
    arguments: Final = {"path": "/private/secret"}
    match format:
        case "openai":
            calls: Final = [
                {
                    **({"id": "duplicate-id"} if with_ids else {}),
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                    "provider_extra": name,
                }
                for name in names
            ]
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "working", "tool_calls": calls},
                        "finish_reason": "tool_calls",
                        "logprobs": {"retained": True},
                    }
                ],
                "usage": {"total_tokens": 17},
                "provider_extra": {"request_id": "keep"},
            }
        case "legacy":
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "working",
                            "function_call": {"name": names[0], "arguments": json.dumps(arguments)},
                        },
                        "finish_reason": "function_call",
                        "logprobs": {"retained": True},
                    }
                ],
                "usage": {"total_tokens": 17},
                "provider_extra": {"request_id": "keep"},
            }
        case "gemini":
            parts: Final = [
                {"functionCall": {"name": name, "args": arguments}, "thoughtSignature": name} for name in names
            ]
            return {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "working"}, *parts]},
                        "finishReason": "STOP",
                        "safetyRatings": [{"blocked": False}],
                    }
                ],
                "usageMetadata": {"totalTokenCount": 17},
                "provider_extra": {"request_id": "keep"},
            }
        case "bedrock":
            blocks: Final = [
                {"toolUse": {**({"toolUseId": "duplicate-id"} if with_ids else {}), "name": name, "input": arguments}}
                for name in names
            ]
            return {
                "output": {"message": {"role": "assistant", "content": [{"text": "working"}, *blocks]}},
                "stopReason": "tool_use",
                "usage": {"totalTokens": 17},
                "provider_extra": {"request_id": "keep"},
            }


def _guardrail(action: Literal["block", "rewrite"]) -> ToolPermissionGuardrail:
    return ToolPermissionGuardrail(
        guardrail_name="raw-tools",
        event_hook=GuardrailEventHooks.post_call,
        default_on=True,
        default_action="allow",
        on_disallowed_action=action,
        rules=[{"id": "deny-read", "tool_name": "^Read$", "decision": "deny"}],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("format", ["openai", "legacy", "gemini", "bedrock"])
async def test_raw_response_blocks_denied_tool_without_mutating_response(format: _Format) -> None:
    response: Final = _response(format, ("Read",))
    before: Final = copy.deepcopy(response)

    with pytest.raises(GuardrailRaisedException, match="Read"):
        await _guardrail("block").async_post_call_success_hook(
            data={}, user_api_key_dict=UserAPIKeyAuth(), response=response
        )

    assert response == before


@pytest.mark.asyncio
@pytest.mark.parametrize("format", ["openai", "legacy", "gemini", "bedrock"])
async def test_raw_response_allowed_tool_is_identity_passthrough(format: _Format) -> None:
    response: Final = _response(format, ("Bash",))
    before: Final = copy.deepcopy(response)

    result: Final = await _guardrail("rewrite").async_post_call_success_hook(
        data={}, user_api_key_dict=UserAPIKeyAuth(), response=response
    )

    assert result is response
    assert response == before


@pytest.mark.asyncio
@pytest.mark.parametrize("format", ["openai", "legacy", "gemini", "bedrock"])
@pytest.mark.parametrize("with_ids", [True, False])
async def test_raw_response_rewrite_removes_denied_call_and_preserves_metadata(format: _Format, with_ids: bool) -> None:
    response: Final = _response(format, ("Read", "Bash"), with_ids)
    before: Final = copy.deepcopy(response)

    result: Final = await _guardrail("rewrite").async_post_call_success_hook(
        data={}, user_api_key_dict=UserAPIKeyAuth(), response=response
    )

    assert result is response
    assert response["provider_extra"] == before["provider_extra"]
    assert "Permission denied" in json.dumps(response)
    match response:
        case {"choices": [{"message": message, "finish_reason": finish, "logprobs": logprobs}]}:
            assert logprobs == {"retained": True}
            assert response["usage"] == before["usage"]
            assert message["content"].startswith("working\n\nPermission denied")
            if format == "legacy":
                assert message["function_call"] is None
                assert finish == "stop"
            else:
                assert [call["function"]["name"] for call in message["tool_calls"]] == ["Bash"]
                assert message["tool_calls"][0]["provider_extra"] == "Bash"
                assert finish == "tool_calls"
        case {"candidates": [{"content": {"parts": parts}, "safetyRatings": ratings, "finishReason": finish}]}:
            assert parts[0] == {"text": "working"}
            assert parts[1] == {
                "functionCall": {"name": "Bash", "args": {"path": "/private/secret"}},
                "thoughtSignature": "Bash",
            }
            assert len(parts) == 3
            assert ratings == [{"blocked": False}]
            assert finish == "STOP"
            assert response["usageMetadata"] == before["usageMetadata"]
        case {"output": {"message": {"content": blocks}}, "stopReason": stop}:
            assert blocks[0] == {"text": "working"}
            assert blocks[1]["toolUse"]["name"] == "Bash"
            assert len(blocks) == 3
            assert stop == "tool_use"
            assert response["usage"] == before["usage"]
        case _:
            pytest.fail("response shape changed")


@pytest.mark.asyncio
@pytest.mark.parametrize("format", ["openai", "legacy", "gemini", "bedrock"])
async def test_raw_response_rewrite_ends_turn_when_all_tools_denied(format: _Format) -> None:
    response: Final = _response(format, ("Read",))

    await _guardrail("rewrite").async_post_call_success_hook(
        data={}, user_api_key_dict=UserAPIKeyAuth(), response=response
    )

    assert "/private/secret" not in json.dumps(response)
    match response:
        case {"choices": [{"message": message, "finish_reason": finish}]}:
            assert not message.get("tool_calls")
            assert not message.get("function_call")
            assert finish == "stop"
        case {"candidates": [{"content": {"parts": parts}}]}:
            assert all("functionCall" not in part for part in parts)
        case {"output": {"message": {"content": blocks}}, "stopReason": stop}:
            assert all("toolUse" not in block for block in blocks)
            assert stop == "end_turn"
        case _:
            pytest.fail("response shape changed")


@pytest.mark.asyncio
@pytest.mark.parametrize("format", ["openai", "legacy", "gemini", "bedrock"])
async def test_raw_response_checks_argument_rules(format: _Format) -> None:
    guardrail: Final = ToolPermissionGuardrail(
        guardrail_name="raw-paths",
        event_hook=GuardrailEventHooks.post_call,
        default_on=True,
        default_action="deny",
        on_disallowed_action="block",
        rules=[
            {
                "id": "public-read",
                "tool_name": "^Read$",
                "decision": "allow",
                "allowed_param_patterns": {"path": "^/public/.*$"},
            }
        ],
    )

    with pytest.raises(GuardrailRaisedException, match="path"):
        await guardrail.async_post_call_success_hook(
            data={}, user_api_key_dict=UserAPIKeyAuth(), response=_response(format, ("Read",))
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("format", ["openai", "gemini"])
async def test_raw_response_rewrite_isolates_candidates(format: _Format) -> None:
    key: Final = "choices" if format == "openai" else "candidates"
    allowed: Final = _response(format, ("Bash",))[key]
    denied: Final = _response(format, ("Read",))[key]
    assert isinstance(allowed, list)
    assert isinstance(denied, list)
    response: Final = {key: [*allowed, *denied]}
    before: Final = copy.deepcopy(allowed)

    result: Final = await _guardrail("rewrite").async_post_call_success_hook(
        data={}, user_api_key_dict=UserAPIKeyAuth(), response=response
    )

    assert result[key][0] == before[0]
    assert "Permission denied" in json.dumps(result[key][1])
    assert "/private/secret" not in json.dumps(result[key][1])


@pytest.mark.asyncio
async def test_raw_openai_rewrite_retains_content_parts() -> None:
    content: Final = [{"type": "text", "text": "working", "provider_extra": 17}]
    response: Final = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [{"type": "function", "function": {"name": "Read", "arguments": "{}"}}],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    result: Final = await _guardrail("rewrite").async_post_call_success_hook(
        data={}, user_api_key_dict=UserAPIKeyAuth(), response=response
    )

    parts: Final = result["choices"][0]["message"]["content"]
    assert parts[0] is content[0]
    assert len(parts) == 2
    assert parts[1]["type"] == "text"
    assert "Permission denied" in parts[1]["text"]
