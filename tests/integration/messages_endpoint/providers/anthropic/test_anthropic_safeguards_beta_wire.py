import itertools
import json
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

BETA: Final = "dangerous-tool-use-2026-09-03"
SAFEGUARDS: Final[list[JsonValue]] = [
    {"type": "dangerous_tool_use", "classifier_context": {"v": 1, "permission_mode": "auto"}}
]
ANTHROPIC_MODEL: Final = "claude-sonnet-4-5-20250929"
ANTHROPIC_KEY: Final = "synthetic-anthropic-key"
_CLAUDE_CODE_BETAS: Final = (
    "claude-code-20250219",
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
    "context-management-2025-06-27",
)
_CLIENT_BETAS: Final = ",".join(_CLAUDE_CODE_BETAS)
_CLIENT_BETAS_WITH_DANGEROUS: Final = ",".join((*_CLAUDE_CODE_BETAS, BETA))
_MERGED_BETAS: Final = ",".join(sorted((*_CLAUDE_CODE_BETAS, BETA)))
_FOUNDRY_MODEL: Final = "azure_ai/claude-sonnet-4-6"
_FOUNDRY_KEY: Final = "synthetic-foundry-key"
_FOUNDRY_TARGET: Final = "/anthropic/v1/messages"
_FOUNDRY_ARRAY_ERROR: Final = "safeguards: Input should be a valid array"
_FOUNDRY_EXTRA_INPUT_ERROR: Final = "safeguards: Extra inputs are not permitted"
_ANTHROPIC_VERSION: Final = MappingProxyType({"anthropic-version": "2023-06-01"})
_MESSAGES_KEYS: Final = frozenset({"max_tokens", "messages", "model", "stream"})
_RESPONSES_KEYS: Final = frozenset({"include", "input", "max_output_tokens", "model"})
_SDK_EXTRA: Final[dict[str, JsonValue]] = {"safeguards": SAFEGUARDS, "cache": {"no-cache": True}, "num_retries": 0}
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class _Omitted:
    pass


_OMITTED: Final = _Omitted()


@dataclass(frozen=True, slots=True)
class _Observed:
    method: str
    target: str
    marker: str
    headers: Mapping[str, str]
    body: Mapping[str, JsonValue]

    @property
    def beta(self) -> str | None:
        return self.headers.get("anthropic-beta")

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self.body)

    @property
    def header_names(self) -> frozenset[str]:
        return frozenset(self.headers)


def marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode())
    assert found is not None, request.body
    return found.group(1)


def _message(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{marker}",
        "type": "message",
        "role": "assistant",
        "model": ANTHROPIC_MODEL,
        "content": [{"type": "text", "text": f"answer marker-{marker}"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 5},
    }


def stream_frames(marker: str) -> tuple[bytes, ...]:
    opening: Final[dict[str, JsonValue]] = {
        **_message(marker),
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 12, "output_tokens": 0},
    }
    events: Final[tuple[tuple[str, dict[str, JsonValue]], ...]] = (
        ("message_start", {"message": opening}),
        ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": f"answer marker-{marker}"}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 5}}),
        ("message_stop", {}),
    )
    return tuple(
        f"event: {kind}\ndata: {json.dumps({'type': kind, **payload})}\n\n".encode() for kind, payload in events
    )


def anthropic_peer(request: Request) -> Reply:
    marker: Final = marker_of(request)
    if _JSON_OBJECT.validate_json(request.body).get("stream") is True:
        return Reply(content_type="text/event-stream", chunks=stream_frames(marker))
    return Reply(body=json.dumps(_message(marker)).encode())


def _responses_peer(request: Request) -> Reply:
    marker: Final = marker_of(request)
    return Reply(
        body=json.dumps(
            {
                "id": f"resp_{marker}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "type": "message",
                        "id": f"msg_{marker}",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": f"answer marker-{marker}", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
            }
        ).encode()
    )


def _error_peer(status: int, kind: str, message: str) -> Callable[[Request], Reply]:
    body: Final = json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode()
    return lambda request: Reply(status=status, body=body)


def messages_body(
    model: str,
    marker: str,
    *,
    safeguards: JsonValue | _Omitted = SAFEGUARDS,
    stream: bool = False,
    no_cache: bool = True,
) -> dict[str, JsonValue]:
    base: Final[dict[str, JsonValue]] = {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": f"Question marker-{marker}"}],
        "num_retries": 0,
        **({"stream": True} if stream else {}),
        **({"cache": {"no-cache": True}} if no_cache else {}),
    }
    if isinstance(safeguards, _Omitted):
        return base
    return {**base, "safeguards": safeguards}


def _observe(request: Request) -> _Observed:
    return _Observed(
        request.method, request.target, marker_of(request), request.headers, _JSON_OBJECT.validate_json(request.body)
    )


def _only(wire: Wire) -> _Observed:
    (request,) = wire.drain()
    return _observe(request)


def _post(gateway: Gateway, body: Mapping[str, JsonValue], *, beta: str | None = None) -> httpx.Response:
    headers: Final = {**_ANTHROPIC_VERSION, **({"anthropic-beta": beta} if beta else {})}
    return gateway.request("POST", "/v1/messages", body, headers=headers)


def _assert_answered(response: httpx.Response, marker: str) -> None:
    assert response.status_code == 200, response.text
    assert response.json()["id"] == f"msg_{marker}", response.text


def _assert_streamed(response: httpx.Response, marker: str) -> None:
    assert response.status_code == 200, response.text
    assert f"msg_{marker}" in response.text and f"answer marker-{marker}" in response.text, response.text
    assert "event: message_stop" in response.text, response.text


def _spend_row(identity: str) -> None:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (identity,)),
        lambda found: len(found) == 1,
        seconds=70,
    )
    assert rows[0]["status"] == "success", rows


def _cache_hit_spend_row(identity: str) -> None:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, status, cache_hit FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s',
            (f"{identity}\\_cache\\_hit%",),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    assert (rows[0]["status"], str(rows[0]["cache_hit"])) == ("success", "True"), rows


def _responses_spend_row(model: str) -> None:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group = %s AND call_type = %s',
            (model, "aresponses"),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    assert rows[0]["status"] == "success", rows


def _expect_messages(
    observed: _Observed,
    *,
    target: str,
    auth: tuple[str, str] | None,
    beta: str | None,
    safeguards: JsonValue | _Omitted = SAFEGUARDS,
    keys: frozenset[str] = _MESSAGES_KEYS,
) -> None:
    assert (observed.method, observed.target) == ("POST", target)
    if auth is not None:
        credential_matches: Final = observed.headers.get(auth[0]) == auth[1]
        assert credential_matches, f"{auth[0]} differs from the deployment credential; headers {observed.header_names}"
    assert observed.beta == beta, observed.header_names
    if isinstance(safeguards, _Omitted):
        assert observed.keys == keys
        return
    assert observed.keys == keys | {"safeguards"}
    assert observed.body["safeguards"] == safeguards


def _expect_anthropic_version(observed: _Observed) -> None:
    assert observed.headers.get("anthropic-version") == "2023-06-01", observed.header_names


@dataclass(frozen=True, slots=True)
class _ShapeCase:
    sent: JsonValue | _Omitted
    forwarded: JsonValue | _Omitted
    client_beta: str | None
    expected_beta: str | None
    stream: bool = False


_ANTHROPIC_SHAPES: Final = MappingProxyType(
    {
        "D1_safeguards_without_client_header": _ShapeCase(SAFEGUARDS, SAFEGUARDS, None, BETA),
        "D2_safeguards_with_claude_code_header": _ShapeCase(
            SAFEGUARDS, SAFEGUARDS, _CLIENT_BETAS_WITH_DANGEROUS, _MERGED_BETAS
        ),
        "D3_safeguards_with_claude_code_header_lacking_the_beta": _ShapeCase(
            SAFEGUARDS, SAFEGUARDS, _CLIENT_BETAS, _MERGED_BETAS
        ),
        "D4_empty_safeguards": _ShapeCase([], [], None, BETA),
        "D5_null_safeguards": _ShapeCase(None, _OMITTED, None, None),
        "D6_missing_safeguards": _ShapeCase(_OMITTED, _OMITTED, None, None),
    }
)
_FOUNDRY_SHAPES: Final = MappingProxyType(
    {
        "D9_streamed_safeguards_with_claude_code_header": _ShapeCase(
            SAFEGUARDS, SAFEGUARDS, _CLIENT_BETAS_WITH_DANGEROUS, BETA, stream=True
        ),
        "D11_empty_safeguards": _ShapeCase([], [], None, BETA),
        "D12_null_safeguards": _ShapeCase(None, _OMITTED, None, None),
        "D13_missing_safeguards_with_claude_code_header_lacking_the_beta": _ShapeCase(
            _OMITTED, _OMITTED, _CLIENT_BETAS, None
        ),
    }
)


@pytest.mark.parametrize("row", tuple(_ANTHROPIC_SHAPES))
def test_anthropic_messages_derive_the_dangerous_tool_use_beta_from_safeguards(gateway: Gateway, row: str) -> None:
    case: Final = _ANTHROPIC_SHAPES[row]
    marker: Final = uuid.uuid4().hex
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=ANTHROPIC_KEY)
        response: Final = _post(gateway, messages_body(model, marker, safeguards=case.sent), beta=case.client_beta)
        _assert_answered(response, marker)
        observed: Final = _only(wire)
        _expect_messages(
            observed,
            target="/v1/messages",
            auth=("x-api-key", ANTHROPIC_KEY),
            beta=case.expected_beta,
            safeguards=case.forwarded,
        )
        _expect_anthropic_version(observed)
        _spend_row(f"msg_{marker}")


@pytest.mark.parametrize("row", tuple(_FOUNDRY_SHAPES))
def test_azure_ai_messages_derive_the_beta_instead_of_forwarding_the_client_header(gateway: Gateway, row: str) -> None:
    case: Final = _FOUNDRY_SHAPES[row]
    marker: Final = uuid.uuid4().hex
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_FOUNDRY_MODEL, api_base=wire.url, api_key=_FOUNDRY_KEY)
        response: Final = _post(
            gateway, messages_body(model, marker, safeguards=case.sent, stream=case.stream), beta=case.client_beta
        )
        if case.stream:
            _assert_streamed(response, marker)
        else:
            _assert_answered(response, marker)
        observed: Final = _only(wire)
        _expect_messages(
            observed,
            target=_FOUNDRY_TARGET,
            auth=("x-api-key", _FOUNDRY_KEY),
            beta=case.expected_beta,
            safeguards=case.forwarded,
        )
        _expect_anthropic_version(observed)
        _spend_row(f"msg_{marker}")


def test_anthropic_sdk_sync_safeguards_derive_the_beta(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=ANTHROPIC_KEY)
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
        reply: Final = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": f"Question marker-{marker}"}],
            extra_body=_SDK_EXTRA,
        )
        assert reply.id == f"msg_{marker}", reply
        _expect_messages(_only(wire), target="/v1/messages", auth=("x-api-key", ANTHROPIC_KEY), beta=BETA)
        _spend_row(f"msg_{marker}")


async def test_anthropic_sdk_async_stream_safeguards_derive_the_beta_and_log_spend(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=ANTHROPIC_KEY)
        client: Final = anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0
        )
        async with client.messages.with_streaming_response.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": f"Question marker-{marker}"}],
            stream=True,
            extra_body=_SDK_EXTRA,
        ) as streamed:
            pieces: Final = [
                event.delta.text
                async for event in await streamed.parse()
                if event.type == "content_block_delta" and event.delta.type == "text_delta"
            ]
        assert "".join(pieces) == f"answer marker-{marker}", pieces
        _expect_messages(_only(wire), target="/v1/messages", auth=("x-api-key", ANTHROPIC_KEY), beta=BETA)
        _spend_row(f"msg_{marker}")


def test_azure_ai_sdk_sync_safeguards_derive_the_beta(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_FOUNDRY_MODEL, api_base=wire.url, api_key=_FOUNDRY_KEY)
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
        reply: Final = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": f"Question marker-{marker}"}],
            extra_body=_SDK_EXTRA,
        )
        assert reply.id == f"msg_{marker}", reply
        _expect_messages(_only(wire), target=_FOUNDRY_TARGET, auth=("x-api-key", _FOUNDRY_KEY), beta=BETA)
        _spend_row(f"msg_{marker}")


@dataclass(frozen=True, slots=True, kw_only=True)
class _ProviderCase:
    parameters: Mapping[str, JsonValue]
    auth: tuple[str, str] | None
    expected_beta: str | None
    target: str = "/v1/messages"
    api_base_suffix: str = ""
    model_info: Mapping[str, JsonValue] | None = None
    client_beta: str | None = None
    keys: frozenset[str] = _MESSAGES_KEYS
    extra_headers: Mapping[str, str] = MappingProxyType({})
    body_beta: list[str] | None = None


_PROVIDERS: Final = MappingProxyType(
    {
        "D14_deepseek_with_claude_code_header": _ProviderCase(
            parameters={"model": "deepseek/deepseek-chat", "api_key": "synthetic-deepseek-key"},
            client_beta=_CLIENT_BETAS_WITH_DANGEROUS,
            target="/anthropic/v1/messages",
            auth=("x-api-key", "synthetic-deepseek-key"),
            expected_beta=None,
        ),
        "D15_tencent_with_claude_code_header": _ProviderCase(
            parameters={"model": "tencent/deepseek-v4-flash", "api_key": "synthetic-tencent-key"},
            client_beta=_CLIENT_BETAS_WITH_DANGEROUS,
            auth=("x-api-key", "synthetic-tencent-key"),
            expected_beta=None,
        ),
        "D16_minimax_with_claude_code_header": _ProviderCase(
            parameters={"model": "minimax/MiniMax-M2", "api_key": "synthetic-minimax-key"},
            client_beta=_CLIENT_BETAS_WITH_DANGEROUS,
            auth=("x-api-key", "synthetic-minimax-key"),
            expected_beta=None,
        ),
        "D17_openai_compatible_native_passthrough": _ProviderCase(
            parameters={"model": "openai/claude-sonnet-4-5", "api_key": "synthetic-openai-like-key"},
            api_base_suffix="/v1",
            model_info={"supported_endpoints": ["/v1/messages"]},
            auth=("authorization", "Bearer synthetic-openai-like-key"),
            expected_beta=BETA,
        ),
        "D18_sail_with_claude_code_header": _ProviderCase(
            parameters={"model": "sail/sail-claude", "api_key": "synthetic-sail-key"},
            client_beta=_CLIENT_BETAS_WITH_DANGEROUS,
            auth=("authorization", "Bearer synthetic-sail-key"),
            expected_beta=BETA,
        ),
        "D19_edenai": _ProviderCase(
            parameters={"model": "edenai/claude-sonnet-4-5", "api_key": "synthetic-edenai-key"},
            auth=("authorization", "Bearer synthetic-edenai-key"),
            expected_beta=BETA,
        ),
        "D21_bedrock_claude_platform": _ProviderCase(
            parameters={
                "model": f"bedrock/claude_platform/{ANTHROPIC_MODEL}",
                "api_key": "synthetic-platform-key",
                "workspace_id": "ws-synthetic",
                "aws_region_name": "us-east-1",
            },
            auth=("x-api-key", "synthetic-platform-key"),
            expected_beta=BETA,
            extra_headers={"anthropic-workspace-id": "ws-synthetic"},
        ),
        "D22_bedrock_claude_platform_with_claude_code_header": _ProviderCase(
            parameters={
                "model": f"bedrock/claude_platform/{ANTHROPIC_MODEL}",
                "api_key": "synthetic-platform-key",
                "workspace_id": "ws-synthetic",
                "aws_region_name": "us-east-1",
            },
            client_beta=_CLIENT_BETAS_WITH_DANGEROUS,
            auth=("x-api-key", "synthetic-platform-key"),
            expected_beta=_MERGED_BETAS,
            extra_headers={"anthropic-workspace-id": "ws-synthetic"},
        ),
        "D23_bedrock_invoke": _ProviderCase(
            parameters={
                "model": "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0",
                "api_key": "synthetic-bedrock-bearer",
                "aws_region_name": "us-east-1",
            },
            target="/model/anthropic.claude-sonnet-4-5-20250929-v1:0/invoke",
            auth=("authorization", "Bearer synthetic-bedrock-bearer"),
            expected_beta=None,
            keys=frozenset({"anthropic_beta", "anthropic_version", "max_tokens", "messages"}),
            body_beta=[BETA],
        ),
        "D24_bedrock_mantle": _ProviderCase(
            parameters={
                "model": "bedrock_mantle/anthropic.claude-sonnet-5-v1:0",
                "api_key": "synthetic-mantle-bearer",
                "aws_region_name": "us-east-1",
            },
            target="/anthropic/v1/messages",
            auth=("authorization", "Bearer synthetic-mantle-bearer"),
            expected_beta=BETA,
            keys=frozenset({"max_tokens", "messages", "model"}),
        ),
    }
)


@pytest.mark.parametrize("row", tuple(_PROVIDERS))
def test_anthropic_compatible_providers_receive_the_beta_only_where_the_proxy_derives_it(
    gateway: Gateway, row: str
) -> None:
    case: Final = _PROVIDERS[row]
    marker: Final = uuid.uuid4().hex
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model_info=case.model_info, api_base=wire.url + case.api_base_suffix, **case.parameters
        )
        response: Final = _post(gateway, messages_body(model, marker), beta=case.client_beta)
        _assert_answered(response, marker)
        observed: Final = _only(wire)
        _expect_messages(observed, target=case.target, auth=case.auth, beta=case.expected_beta, keys=case.keys)
        for name, value in case.extra_headers.items():
            assert observed.headers.get(name) == value, observed.header_names
        assert observed.body.get("anthropic_beta") == case.body_beta, observed.body
        _spend_row(f"msg_{marker}")


def test_github_copilot_messages_receive_the_derived_beta_without_the_client_header(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex
    token_dir: Final = tmp_path / "copilot"
    token_dir.mkdir()
    with wire_server(anthropic_peer) as wire:
        (token_dir / "api-key.json").write_text(
            json.dumps({"token": "synthetic-copilot-token", "expires_at": 4102444800, "endpoints": {"api": wire.url}})
        )
        with (
            owned_proxy(gateway, tmp_path, {"GITHUB_COPILOT_TOKEN_DIR": str(token_dir)}) as owned,
            owned.scenario() as scenario,
        ):
            model: Final = scenario.model(model="github_copilot/claude-sonnet-4-5", api_base=wire.url)
            response: Final = _post(owned, messages_body(model, marker), beta=_CLIENT_BETAS_WITH_DANGEROUS)
            _assert_answered(response, marker)
            observed: Final = _only(wire)
            _expect_messages(
                observed, target="/v1/messages", auth=("authorization", "Bearer synthetic-copilot-token"), beta=BETA
            )
            assert observed.headers.get("openai-intent") == "messages-proxy", observed.header_names
            _spend_row(f"msg_{marker}")


def test_non_native_deployments_bridge_to_the_responses_api_without_safeguards_or_the_beta(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_responses_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key="synthetic-openai-key"
        )
        response: Final = _post(gateway, messages_body(model, marker), beta=_CLIENT_BETAS_WITH_DANGEROUS)
        assert response.status_code == 200 and f"answer marker-{marker}" in response.text, response.text
        observed: Final = _only(wire)
        assert (observed.method, observed.target) == ("POST", "/v1/responses")
        assert observed.headers.get("authorization") == "Bearer synthetic-openai-key", observed.header_names
        assert observed.beta is None, observed.header_names
        assert observed.keys == _RESPONSES_KEYS, observed.keys
        _spend_row(str(_JSON_OBJECT.validate_json(response.content)["id"]))


def test_chat_completions_forward_safeguards_verbatim_without_deriving_the_beta(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=ANTHROPIC_KEY)
        client: Final = openai.OpenAI(
            base_url=str(gateway.client.base_url).rstrip("/") + "/v1", api_key=gateway.key, max_retries=0
        )
        completion: Final = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": f"Question marker-{marker}"}], extra_body=_SDK_EXTRA
        )
        assert completion.choices[0].message.content == f"answer marker-{marker}", completion
        _expect_messages(
            _only(wire),
            target="/v1/messages",
            auth=("x-api-key", ANTHROPIC_KEY),
            beta=None,
            keys=frozenset({"max_tokens", "messages", "model"}),
        )
        _spend_row(completion.id)


async def test_responses_stream_forwards_safeguards_verbatim_without_deriving_the_beta(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=ANTHROPIC_KEY)
        client: Final = openai.AsyncOpenAI(
            base_url=str(gateway.client.base_url).rstrip("/") + "/v1", api_key=gateway.key, max_retries=0
        )
        async with client.responses.with_streaming_response.create(
            model=model, input=f"Question marker-{marker}", stream=True, extra_body=_SDK_EXTRA
        ) as streamed:
            events: Final = [event async for event in await streamed.parse()]
        deltas: Final = [event.delta for event in events if event.type == "response.output_text.delta"]
        assert "".join(deltas) == f"answer marker-{marker}", events
        assert any(event.type == "response.completed" for event in events), events
        _expect_messages(_only(wire), target="/v1/messages", auth=("x-api-key", ANTHROPIC_KEY), beta=None)
        _responses_spend_row(model)


_FOUNDRY_REJECTED_SHAPES: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"S1_int": 1, "S2_empty_string": "", "S3_five_kb_string": "x" * 5000}
)


@pytest.mark.parametrize("row", tuple(_FOUNDRY_REJECTED_SHAPES))
def test_foundry_rejecting_a_non_array_safeguards_value_reaches_the_caller_after_one_attempt(
    gateway: Gateway, row: str
) -> None:
    value: Final = _FOUNDRY_REJECTED_SHAPES[row]
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(_error_peer(400, "invalid_request_error", _FOUNDRY_ARRAY_ERROR)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=_FOUNDRY_MODEL, api_base=wire.url, api_key=_FOUNDRY_KEY)
        response: Final = _post(gateway, messages_body(model, marker, safeguards=value))
        assert response.status_code == 400 and _FOUNDRY_ARRAY_ERROR in response.text, response.text
        _expect_messages(
            _only(wire), target=_FOUNDRY_TARGET, auth=("x-api-key", _FOUNDRY_KEY), beta=BETA, safeguards=value
        )


def test_duplicate_safeguards_keys_resolve_to_the_last_value(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_FOUNDRY_MODEL, api_base=wire.url, api_key=_FOUNDRY_KEY)
        raw: Final = (
            f'{{"model": "{model}", "max_tokens": 16, "num_retries": 0, "cache": {{"no-cache": true}}, '
            f'"messages": [{{"role": "user", "content": "Question marker-{marker}"}}], '
            f'"safeguards": {json.dumps(SAFEGUARDS)}, "safeguards": null}}'
        )
        response: Final = gateway.client.post(
            "/v1/messages",
            content=raw.encode(),
            headers={
                "Authorization": f"Bearer {gateway.key}",
                "content-type": "application/json",
                **_ANTHROPIC_VERSION,
            },
        )
        _assert_answered(response, marker)
        _expect_messages(
            _only(wire), target=_FOUNDRY_TARGET, auth=("x-api-key", _FOUNDRY_KEY), beta=None, safeguards=_OMITTED
        )
        _spend_row(f"msg_{marker}")


def test_the_same_safeguards_sent_twice_are_two_upstream_calls_with_two_spend_rows(gateway: Gateway) -> None:
    markers: Final = (uuid.uuid4().hex, uuid.uuid4().hex)
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_FOUNDRY_MODEL, api_base=wire.url, api_key=_FOUNDRY_KEY)
        responses: Final = tuple(_post(gateway, messages_body(model, marker)) for marker in markers)
        for marker, response in zip(markers, responses, strict=True):
            _assert_answered(response, marker)
        received: Final = tuple(_observe(request) for request in wire.drain())
        assert len(received) == 2, received
        for observed in received:
            _expect_messages(observed, target=_FOUNDRY_TARGET, auth=("x-api-key", _FOUNDRY_KEY), beta=BETA)
        assert sorted(item.marker for item in received) == sorted(markers)
        for marker in markers:
            _spend_row(f"msg_{marker}")


_UPSTREAM_ERRORS: Final = MappingProxyType(
    {
        "S6_unauthorized_deployment": (401, "authentication_error", "invalid x-api-key"),
        "S7_foundry_extra_inputs_rejection": (400, "invalid_request_error", _FOUNDRY_EXTRA_INPUT_ERROR),
    }
)


@pytest.mark.parametrize("row", tuple(_UPSTREAM_ERRORS))
def test_upstream_errors_reach_the_caller_with_the_beta_sent_once(gateway: Gateway, row: str) -> None:
    status, kind, message = _UPSTREAM_ERRORS[row]
    marker: Final = uuid.uuid4().hex
    with wire_server(_error_peer(status, kind, message)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_FOUNDRY_MODEL, api_base=wire.url, api_key=_FOUNDRY_KEY)
        response: Final = _post(gateway, messages_body(model, marker))
        assert response.status_code == status and message in response.text, response.text
        _expect_messages(_only(wire), target=_FOUNDRY_TARGET, auth=("x-api-key", _FOUNDRY_KEY), beta=BETA)


def _post_noting_whether_upstream_was_called(
    gateway: Gateway, body: Mapping[str, JsonValue], wire: Wire
) -> tuple[httpx.Response, bool]:
    before: Final = wire.received.qsize()
    response: Final = _post(gateway, body)
    assert response.status_code == 200, response.text
    return response, wire.received.qsize() > before


def _until_served_from_cache(gateway: Gateway, body: Mapping[str, JsonValue], wire: Wire) -> tuple[httpx.Response, int]:
    deadline: Final = time.monotonic() + 20
    for attempt in itertools.count(1):
        response, called_upstream = _post_noting_whether_upstream_was_called(gateway, body, wire)
        if not called_upstream:
            return response, attempt
        assert time.monotonic() < deadline, "the response cache never served the repeated request"
    raise AssertionError("unreachable")


def test_the_response_cache_serves_the_repeated_request_without_an_upstream_call(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_FOUNDRY_MODEL, api_base=wire.url, api_key=_FOUNDRY_KEY)
        body: Final = messages_body(model, marker, no_cache=False)
        first: Final = _post(gateway, body)
        _assert_answered(first, marker)
        _expect_messages(_only(wire), target=_FOUNDRY_TARGET, auth=("x-api-key", _FOUNDRY_KEY), beta=BETA)
        _spend_row(f"msg_{marker}")
        hit, attempts = _until_served_from_cache(gateway, body, wire)
        assert hit.json()["id"] == first.json()["id"], hit.text
        assert hit.json()["content"] == first.json()["content"], hit.text
        _cache_hit_spend_row(f"msg_{marker}")
        misses: Final = tuple(_observe(request) for request in wire.drain())
        assert len(misses) == attempts - 1, (attempts, misses)
        for observed in misses:
            _expect_messages(observed, target=_FOUNDRY_TARGET, auth=("x-api-key", _FOUNDRY_KEY), beta=BETA)
