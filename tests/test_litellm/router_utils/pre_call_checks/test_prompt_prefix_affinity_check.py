import os
import sys

import hashlib
import re

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm.router_utils.pre_call_checks.prompt_prefix_affinity_check as affinity_module

from litellm.router_utils.pre_call_checks.prompt_prefix_affinity_check import (
    PromptPrefixAffinityCheck,
)


def _fake_encode(*, model: str, text: str):
    """
    Deterministic, offline tokenizer stub.

    Tests in `tests/test_litellm/` must not make network calls. The real
    `litellm.utils.encode()` can trigger a tiktoken download on fresh CI runners.
    We stub it with a pure function that produces stable token IDs derived from
    token content, so prefix routing behavior remains meaningful.
    """

    _ = model  # model selection is irrelevant for this stub
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    token_ids = []
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        token_ids.append(int.from_bytes(digest[:4], "big"))
    return token_ids


@pytest.fixture(autouse=True)
def _stub_encode(monkeypatch):
    monkeypatch.setattr(affinity_module, "encode", _fake_encode)


def _deployments():
    return [
        {"model_info": {"id": "deployment-a"}},
        {"model_info": {"id": "deployment-b"}},
        {"model_info": {"id": "deployment-c"}},
    ]


@pytest.mark.asyncio
async def test_same_prompt_prefix_routes_to_same_deployment_across_suffixes():
    check = PromptPrefixAffinityCheck(
        prefix_tokens=64,
        min_tokens=0,
    )
    shared_prefix = "shared context " * 300

    first_kwargs = {"input": shared_prefix + "question A"}
    second_kwargs = {"input": shared_prefix + "question B"}

    first = await check.async_filter_deployments(
        model="gpt-3.5-turbo",
        healthy_deployments=_deployments(),
        messages=None,
        request_kwargs=first_kwargs,
    )
    second = await check.async_filter_deployments(
        model="gpt-3.5-turbo",
        healthy_deployments=list(reversed(_deployments())),
        messages=None,
        request_kwargs=second_kwargs,
    )

    assert len(first) == 1
    assert len(second) == 1
    assert first[0]["model_info"]["id"] == second[0]["model_info"]["id"]
    assert first_kwargs["_prompt_prefix_affinity_pinned"] is True
    assert second_kwargs["_prompt_prefix_affinity_pinned"] is True


def test_different_prompt_prefixes_get_different_prefix_hashes():
    check = PromptPrefixAffinityCheck(
        prefix_tokens=64,
        min_tokens=0,
    )

    first_prompt = check._build_canonical_prompt(
        messages=None,
        request_kwargs={"input": "alpha " * 300},
    )
    second_prompt = check._build_canonical_prompt(
        messages=None,
        request_kwargs={"input": "beta " * 300},
    )

    assert first_prompt is not None
    assert second_prompt is not None
    assert check._get_prefix_hash(
        "gpt-3.5-turbo", first_prompt
    ) != check._get_prefix_hash("gpt-3.5-turbo", second_prompt)


def test_encrypted_content_is_excluded_from_canonical_prompt_hash():
    check = PromptPrefixAffinityCheck(
        prefix_tokens=64,
        min_tokens=0,
    )

    first_prompt = check._build_canonical_prompt(
        messages=None,
        request_kwargs={
            "input": [
                {
                    "type": "reasoning",
                    "encrypted_content": "encrypted-content-a",
                },
                {"role": "user", "content": "shared context " * 300},
            ]
        },
    )
    second_prompt = check._build_canonical_prompt(
        messages=None,
        request_kwargs={
            "input": [
                {
                    "type": "reasoning",
                    "encrypted_content": "encrypted-content-b",
                },
                {"role": "user", "content": "shared context " * 300},
            ]
        },
    )

    assert first_prompt is not None
    assert second_prompt is not None
    assert check._get_prefix_hash(
        "gpt-3.5-turbo", first_prompt
    ) == check._get_prefix_hash("gpt-3.5-turbo", second_prompt)


@pytest.mark.asyncio
async def test_prompt_prefix_affinity_does_not_filter_below_min_tokens():
    check = PromptPrefixAffinityCheck(
        prefix_tokens=64,
        min_tokens=10_000,
    )
    deployments = _deployments()
    request_kwargs = {"input": "short prompt"}

    result = await check.async_filter_deployments(
        model="gpt-3.5-turbo",
        healthy_deployments=deployments,
        messages=None,
        request_kwargs=request_kwargs,
    )

    assert result == deployments
    assert "_prompt_prefix_affinity_pinned" not in request_kwargs
