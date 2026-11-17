"""anthropic-beta handling and `/v1/messages/count_tokens` per provider behind `/v1/messages`.

Caller headers. The proxy hands the caller's `anthropic-version` and `anthropic-beta` headers only to
providers that speak Anthropic: `anthropic`, `bedrock`, `bedrock_mantle` and `vertex_ai` on every route, and
`azure_ai` and `databricks` on their native Messages route only
(`litellm/proxy/litellm_pre_call_utils.py` `add_provider_specific_headers_to_request`). Chat-only providers
reached through the bridge (Fireworks, Baseten, hosted vLLM, Azure OpenAI) must never receive `anthropic-*`
headers, which strict OpenAI-compatible servers reject or log. Bedrock Converse is an Anthropic-family
provider: for a Claude model the caller's betas travel inside the Converse body as
`additionalModelRequestFields.anthropic_beta` (`converse_transformation.py` `_process_tools_and_beta`), and for
a non-Anthropic model they are not sent at all.

Count tokens. `POST /v1/messages/count_tokens` asks the provider's count API when the provider has one
(Bedrock CountTokens, Anthropic, Azure AI Foundry, Vertex) and answers from a local tokenizer otherwise. The
answer is `{"input_tokens": N, "litellm_count_source": ...}` (`count_source_fields`), where the source is
`provider`, `token_count_model` or `local_estimate`. Chat-only providers are never called and the request
writes no SpendLogs row.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final

import pytest
from integration._support.client import Gateway, held, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import JSON_OBJECT, SPEND_COLUMNS, Rig
from integration.messages_endpoint.dec1_msgprov2_support import HOSTED_VLLM_BACKEND, HOSTED_VLLM_KEY
from integration.messages_endpoint.dec1_msgprov_support import (
    AZURE_API_VERSION,
    AZURE_DEPLOYMENT,
    AZURE_KEY,
    BASETEN_BACKEND,
    BASETEN_KEY,
    BEDROCK_BASE_MODEL_ID,
    BEDROCK_MODEL_ID,
    BEDROCK_TOKEN,
    FIREWORKS_BACKEND,
    FIREWORKS_KEY,
    TOOL,
    at,
    body_of,
    chat_echo,
    converse_echo,
    path_of,
    provider_deployment,
    unique_text,
)
from pydantic import JsonValue

_CALLER_BETA: Final = "interleaved-thinking-2025-05-14"
_CALLER_HEADERS: Final = {"anthropic-beta": _CALLER_BETA, "anthropic-version": "2023-06-01"}
_NOVA: Final = "us.amazon.nova-pro-v1:0"


def _fireworks(gateway: Gateway, respond: Callable[[Request], Reply] | None = None) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        respond or chat_echo(FIREWORKS_BACKEND),
        model=f"fireworks_ai/{FIREWORKS_BACKEND}",
        api_key=FIREWORKS_KEY,
    )


def _baseten(gateway: Gateway, respond: Callable[[Request], Reply] | None = None) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway, respond or chat_echo(BASETEN_BACKEND), model=f"baseten/{BASETEN_BACKEND}", api_key=BASETEN_KEY
    )


def _vllm(gateway: Gateway, respond: Callable[[Request], Reply] | None = None) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        respond or chat_echo(HOSTED_VLLM_BACKEND),
        model=f"hosted_vllm/{HOSTED_VLLM_BACKEND}",
        base_suffix="/v1",
        api_key=HOSTED_VLLM_KEY,
    )


def _azure(gateway: Gateway) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        chat_echo(AZURE_DEPLOYMENT),
        model=f"azure/{AZURE_DEPLOYMENT}",
        api_key=AZURE_KEY,
        api_version=AZURE_API_VERSION,
    )


def _converse(gateway: Gateway, model_id: str = BEDROCK_MODEL_ID) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway, converse_echo, model=f"bedrock/converse/{model_id}", api_key=BEDROCK_TOKEN, aws_region_name="us-east-1"
    )


_CHAT_ONLY: Final[dict[str, Callable[[Gateway], AbstractContextManager[Rig]]]] = {
    "fireworks_ai": _fireworks,
    "baseten": _baseten,
    "hosted_vllm": _vllm,
    "azure": _azure,
}


# ---------------------------------------------------------------- anthropic-* headers per provider


@pytest.mark.parametrize("provider", sorted(_CHAT_ONLY))
def test_a_chat_only_provider_never_receives_anthropic_headers(gateway: Gateway, provider: str) -> None:
    """The caller sends `anthropic-beta: interleaved-thinking-2025-05-14` and `anthropic-version: 2023-06-01`. The
    bridged request to Fireworks, Baseten, hosted vLLM or Azure OpenAI carries no header whose name starts with
    `anthropic-`, because none of these providers is in the Anthropic header scopes. A port that forwards
    `anthropic-*` to every upstream would break strict OpenAI-compatible servers."""
    with _CHAT_ONLY[provider](gateway) as rig:
        response: Final = rig.post(rig.body(), headers=_CALLER_HEADERS)
        assert response.status_code == 200, response.text
        sent: Final = [name for name in rig.one_call().headers if name.lower().startswith("anthropic-")]
        assert sent == [], sent


def test_a_chat_only_provider_never_receives_a_beta_in_the_body(gateway: Gateway) -> None:
    """The same caller beta must not move into the chat body either: the Fireworks request has no
    `anthropic_beta` or `betas` member and no `extra_headers` member."""
    with _fireworks(gateway) as rig:
        assert rig.post(rig.body(), headers=_CALLER_HEADERS).status_code == 200
        body: Final = rig.one_body()
        assert not {"anthropic_beta", "betas", "extra_headers"} & set(body), sorted(body)
        assert _CALLER_BETA not in json.dumps(body)


def test_a_claude_converse_call_carries_the_caller_beta_in_the_body(gateway: Gateway) -> None:
    """For `bedrock/converse/us.anthropic.claude-sonnet-4-5-...` with a tool, the caller's
    `anthropic-beta: interleaved-thinking-2025-05-14` arrives in the Converse body as
    `additionalModelRequestFields.anthropic_beta: ["interleaved-thinking-2025-05-14"]`, which is how Bedrock
    takes Anthropic betas. UNVERIFIED: that the beta list is built when the request has tools only; the tool is
    sent to be safe."""
    with _converse(gateway) as rig:
        assert rig.post(rig.body(tools=[TOOL]), headers=_CALLER_HEADERS).status_code == 200
        assert at(body_of(rig.one_call()), "additionalModelRequestFields", "anthropic_beta") == [_CALLER_BETA]


def test_a_non_anthropic_converse_model_gets_no_anthropic_beta(gateway: Gateway) -> None:
    """For a Nova model on Converse the same caller beta is not put in `additionalModelRequestFields`
    (`anthropic_beta` is only set when the base model starts with `anthropic`), since Nova rejects unknown
    fields there."""
    with _converse(gateway, _NOVA) as rig:
        assert rig.post(rig.body(tools=[TOOL]), headers=_CALLER_HEADERS).status_code == 200
        fields: Final = body_of(rig.one_call()).get("additionalModelRequestFields")
        assert fields is None or "anthropic_beta" not in object_value(fields), fields


def test_a_nova_converse_call_posts_to_the_nova_model_path(gateway: Gateway) -> None:
    """A `bedrock/converse/us.amazon.nova-pro-v1:0` deployment receives `POST /model/us.amazon.nova-pro-v1:0/converse`
    on `/v1/messages`, so a non-Claude Bedrock model is reachable through the bridge at all."""
    with _converse(gateway, _NOVA) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert path_of(rig.one_call()) == f"/model/{_NOVA}/converse"


# ---------------------------------------------------------------- count_tokens per provider


def _count(rig: Rig, **fields: JsonValue) -> dict[str, JsonValue]:
    body: Final[dict[str, JsonValue]] = {
        "model": rig.model,
        "messages": [{"role": "user", "content": unique_text()}],
        **fields,
    }
    response: Final = rig.post(body, path="/v1/messages/count_tokens")
    assert response.status_code == 200, response.text
    return JSON_OBJECT.validate_json(response.content)


@pytest.mark.parametrize("provider", ["fireworks_ai", "baseten", "hosted_vllm"])
def test_count_tokens_on_a_chat_only_provider_is_a_local_estimate(gateway: Gateway, provider: str) -> None:
    """For Fireworks, Baseten and hosted vLLM, `POST /v1/messages/count_tokens` answers HTTP 200 with a positive
    integer `input_tokens` and `litellm_count_source: "local_estimate"`, and the provider is never called:
    none of them has a provider token counter."""
    with _CHAT_ONLY[provider](gateway) as rig:
        payload: Final = _count(rig)
        assert isinstance(payload["input_tokens"], int) and payload["input_tokens"] > 0
        assert payload["litellm_count_source"] == "local_estimate"
        assert rig.calls() == ()


def test_a_local_count_grows_with_tools(gateway: Gateway) -> None:
    """On Baseten the same prompt counted with one tool definition gives a larger `input_tokens` than without
    it: the local estimate includes the tool schema, as inference would bill it."""
    prompt: Final = unique_text()
    with _baseten(gateway) as rig:
        plain: Final = _count(rig, messages=[{"role": "user", "content": prompt}])
        with_tool: Final = _count(rig, messages=[{"role": "user", "content": prompt}], tools=[TOOL])
        assert int(str(with_tool["input_tokens"])) > int(str(plain["input_tokens"]))


def test_count_tokens_on_a_chat_only_provider_writes_no_spend_row(gateway: Gateway) -> None:
    """A Fireworks `count_tokens` call is never billed: no SpendLogs row appears for the model group within
    five seconds. A port that logs count requests as inference would charge keys for counting."""
    with _fireworks(gateway) as rig:
        _count(rig)
        held(
            lambda: read_rows(SPEND_COLUMNS + "WHERE model_group=%s", (rig.model,)),
            lambda rows: rows == [],
            holding=5,
        )


def test_count_tokens_on_a_claude_converse_deployment_calls_bedrock_count_tokens(gateway: Gateway) -> None:
    """A `bedrock/converse/us.anthropic.claude-sonnet-4-5-...` deployment counts with Bedrock: one
    `POST /model/anthropic.claude-sonnet-4-5-20250929-v1:0/count-tokens` (the base model id) and the answer's
    `input_tokens` is Bedrock's `inputTokens: 42`. UNVERIFIED: `litellm_count_source` (`provider` or
    `token_count_model`, depending on the cost map row), so it is not pinned."""

    def respond(request: Request) -> Reply:
        if path_of(request).endswith("/count-tokens"):
            return Reply(body=json.dumps({"inputTokens": 42}).encode())
        return converse_echo(request)

    with provider_deployment(
        gateway,
        respond,
        model=f"bedrock/converse/{BEDROCK_MODEL_ID}",
        api_key=BEDROCK_TOKEN,
        aws_region_name="us-east-1",
    ) as rig:
        assert _count(rig)["input_tokens"] == 42
        assert path_of(rig.one_call()) == f"/model/{BEDROCK_BASE_MODEL_ID}/count-tokens"


def test_count_tokens_without_messages_is_refused_before_any_provider(gateway: Gateway) -> None:
    """`count_tokens` with `messages: []` on a Baseten deployment is HTTP 400 in the Anthropic envelope with a
    message containing `messages parameter is required`, and Baseten is not called."""
    with _baseten(gateway) as rig:
        response: Final = rig.post({"model": rig.model, "messages": []}, path="/v1/messages/count_tokens")
        assert response.status_code == 400, response.text
        error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
        assert "messages parameter is required" in str(error["message"])
        assert rig.calls() == ()
