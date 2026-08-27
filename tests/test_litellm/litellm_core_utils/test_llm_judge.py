"""Unit tests for the shared LLM-judge primitives: verdict parsing, router resolution, dispatch."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.litellm_core_utils.llm_judge import (
    extract_text_from_content,
    judge_acompletion,
    parse_json_verdict,
    router_resolves_model,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"preference": "A", "confidence": 0.9}', "A"),
        ('Here it is:\n```json\n{"preference": "B"}\n```\nDone.', "B"),
        ('```\n{"preference": "tie"}\n```', "tie"),
        ('Verdict: {"preference": "A", "confidence": 0.5} final.', "A"),
    ],
)
def test_parse_json_verdict_tolerates_fences_and_prose(raw, expected):
    assert parse_json_verdict(raw)["preference"] == expected


def test_parse_json_verdict_rejects_non_object():
    with pytest.raises(ValueError, match='judge response is not a JSON object'):
        parse_json_verdict('["not", "an", "object"]')
    with pytest.raises((json.JSONDecodeError, ValueError)):
        parse_json_verdict("no json here at all")


@pytest.mark.parametrize(
    "content,expected",
    [
        ("hello", "hello"),
        ([{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}, {"type": "text", "text": "b"}], "a b"),
        (42, ""),
        (None, ""),
    ],
)
def test_extract_text_from_content(content, expected):
    assert extract_text_from_content(content) == expected


def _router(alias=(), deployments=False) -> MagicMock:
    router = MagicMock()
    router.model_group_alias = dict.fromkeys(alias, "x")
    router.get_model_list = MagicMock(
        return_value=[{"litellm_params": {"model": "openai/gpt-4o"}}] if deployments else None
    )
    router.acompletion = AsyncMock(return_value={"choices": [{"message": {"content": "router answer"}}]})
    return router


def test_router_resolves_model_matrix():
    assert router_resolves_model(None, "gpt-4o") is False
    assert router_resolves_model(_router(), "gpt-4o") is False
    assert router_resolves_model(_router(alias=("gpt-4o",)), "gpt-4o") is True
    assert router_resolves_model(_router(deployments=True), "gpt-4o") is True


@pytest.mark.asyncio
async def test_judge_acompletion_prefers_router_and_disables_retries():
    router = _router(deployments=True)
    response = await judge_acompletion(router, "judge-model", [{"role": "user", "content": "hi"}], temperature=0)
    assert response == {"choices": [{"message": {"content": "router answer"}}]}
    _, kwargs = router.acompletion.call_args
    assert kwargs["num_retries"] == 0
    assert kwargs["fallbacks"] == []
    assert kwargs["temperature"] == 0
    assert kwargs["drop_params"] is True


@pytest.mark.asyncio
async def test_judge_acompletion_falls_back_to_sdk_for_unconfigured_model(monkeypatch: pytest.MonkeyPatch):
    import litellm as litellm_module

    sdk = AsyncMock(return_value={"choices": [{"message": {"content": "sdk answer"}}]})
    monkeypatch.setattr(litellm_module, "acompletion", sdk)
    router = _router()

    response = await judge_acompletion(router, "anthropic/claude-sonnet-5", [{"role": "user", "content": "hi"}])

    assert response == {"choices": [{"message": {"content": "sdk answer"}}]}
    router.acompletion.assert_not_called()
    assert sdk.call_args.kwargs["model"] == "anthropic/claude-sonnet-5"
    assert sdk.call_args.kwargs["num_retries"] == 0
    assert sdk.call_args.kwargs["drop_params"] is True
