"""Unit tests for the shared LLM-judge primitives: verdict parsing, router resolution, dispatch."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.litellm_core_utils.llm_judge import (
    extract_text_from_content,
    judge_acompletion,
    judge_target,
    parse_json_verdict,
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


def _router(alias: tuple[str, ...] = (), deployments: bool = False) -> litellm.Router:
    """A real Router, so name resolution is the product's own.

    Only the network call is faked: a resolution fake has to be kept in step with every
    channel the real one composes, and the one that was here answered a stubbed
    `get_model_list` while the code under test asked a different method, so every arm-choice
    assertion passed on a truthy Mock.
    """
    router = litellm.Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"}}
            for name in (("gpt-4o",) if deployments else ()) + (("alias-target",) if alias else ())
        ],
        model_group_alias=dict.fromkeys(alias, "alias-target"),
    )
    router.acompletion = AsyncMock(  # pyright: ignore[reportAttributeAccessIssue]  # fake only the call, not the resolution
        return_value={"choices": [{"message": {"content": "router answer"}}]}
    )
    return router


def test_judge_target_matrix() -> None:
    """Every name lands in exactly one of the three outcomes the dispatch branches on."""
    assert judge_target(None, "gpt-4o").via == "sdk"
    assert judge_target(_router(), "gpt-4o").via == "sdk"
    assert judge_target(_router(alias=("gpt-4o",)), "gpt-4o").via == "router"
    assert judge_target(_router(deployments=True), "gpt-4o").via == "router"
    assert judge_target(_router(), "not/a real model!").via == "nothing"


@pytest.mark.asyncio
async def test_judge_acompletion_prefers_router_and_disables_retries():
    router = _router(deployments=True)
    response = await judge_acompletion(router, "gpt-4o", [{"role": "user", "content": "hi"}], temperature=0)
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


@pytest.mark.parametrize(
    "model,expected",
    [
        ("named-deployment", frozenset({"anthropic/claude-sonnet-5"})),
        ("alias-for-it", frozenset({"anthropic/claude-sonnet-5"})),
        ("anthropic/claude-sonnet-5", frozenset({"anthropic/claude-sonnet-5"})),
        ("anthropic/claude-opus-4-5", frozenset({"anthropic/claude-opus-4-5"})),
    ],
    ids=["deployment", "alias", "the-public-name-the-deployment-serves", "nothing-configured"],
)
def test_judge_target_identifies_a_name_by_what_would_serve_it(model: str, expected: frozenset[str]) -> None:
    """Three spellings of one model must come back as one identity, or a caller comparing
    two names by their answering models would call the same model two different ones.

    The last case is the fallback: nothing on the proxy serves it, so the SDK gets the name
    verbatim and the name is the identity.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "named-deployment",
                "litellm_params": {"model": "anthropic/claude-sonnet-5", "api_key": "fake"},
            }
        ],
        model_group_alias={"alias-for-it": "named-deployment"},
    )

    assert judge_target(router, model).models == expected


def test_judge_target_without_a_router_is_the_public_name_the_sdk_would_call() -> None:
    target = judge_target(None, "anthropic/claude-sonnet-5")
    assert (target.via, target.models) == ("sdk", frozenset({"anthropic/claude-sonnet-5"}))


def test_judge_target_gives_one_identity_to_a_bare_public_name_and_a_prefixed_deployment() -> None:
    """`gpt-4o` and a deployment serving `openai/gpt-4o` are one model, so a judge named the
    first must collide with a tier named the second. Comparing the spellings finds nothing
    and the job runs with the judge grading itself."""
    router = litellm.Router(
        model_list=[{"model_name": "fast-tier", "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"}}]
    )

    assert judge_target(router, "gpt-4o").models == judge_target(router, "fast-tier").models
