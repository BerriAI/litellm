import asyncio
import json
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, get_args
from uuid import uuid4

import anthropic
import httpx
import pytest
from integration._support.client import Gateway, Scenario, object_value, string_value
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter
from typing_extensions import assert_never

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_MODEL: Final = "gemini-4-argon"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-east5"
_COUNT: Final = 777
_UNIT: Final = " red blue green yellow orange purple black white silver gold"
_USER: Final = f"<filler>{_UNIT * 100}</filler>\nReturn exactly OK."
_MESSAGES: Final[JsonValue] = [{"role": "user", "content": _USER}]
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_LIST: Final = TypeAdapter(list[JsonValue])
_REPLY: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "OK"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1, "totalTokenCount": 11},
}


class _GeminiPeer:
    def __init__(self) -> None:
        self.status: int = 200
        self.requests: tuple[Request, ...] = ()
        self._lock: Final = threading.Lock()

    def set_status(self, status: int) -> None:
        self.status = status

    def __call__(self, request: Request) -> Reply:
        with self._lock:
            self.requests = (*self.requests, request)
            status: Final = self.status
        if request.target.endswith(":countTokens"):
            return Reply(status=status, body=json.dumps({"totalTokens": _COUNT}).encode())
        if request.target.endswith(":generateContent"):
            return Reply(body=json.dumps(_REPLY).encode())
        return Reply(status=404, body=json.dumps({"error": request.target}).encode())

    def drain(self) -> tuple[Request, ...]:
        with self._lock:
            requests: Final = self.requests
            self.requests = ()
        return requests


def _deployment(gateway: Gateway, scenario: Scenario, api_base: str, **parameters: JsonValue) -> str:
    return scenario.model(
        model=f"vertex_ai/{_MODEL}",
        api_base=api_base,
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
        **parameters,
    )


def _messages(model: str, system: JsonValue | None, tools: JsonValue | None) -> Mapping[str, JsonValue]:
    return {
        "model": model,
        "messages": _MESSAGES,
        **({"system": system} if system is not None else {}),
        **({"tools": tools} if tools is not None else {}),
    }


def _json_body(request: Request) -> Mapping[str, JsonValue]:
    return _JSON_OBJECT.validate_json(request.body)


def _count_requests(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if request.target.endswith(":countTokens"))


def _count_tokens_tools(generated_tools: JsonValue) -> JsonValue:
    return _JSON_LIST.validate_python(
        [
            {
                {
                    "function_declarations": "functionDeclarations",
                    "url_context": "urlContext",
                    "code_execution": "codeExecution",
                }.get(key, key): value
                for key, value in tool.items()
            }
            for tool in TypeAdapter(list[Mapping[str, JsonValue]]).validate_python(generated_tools)
        ]
    )


def _count_tokens_request_from_generate_content(
    generated_body: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    system_instruction: Final = generated_body.get("systemInstruction", generated_body.get("system_instruction"))
    generated_tools: Final = generated_body.get("tools")
    return MappingProxyType(
        {
            "contents": generated_body["contents"],
            **({"systemInstruction": system_instruction} if system_instruction is not None else {}),
            **({"tools": _count_tokens_tools(generated_tools)} if generated_tools is not None else {}),
        }
    )


def _tools(description: str) -> JsonValue:
    return _JSON_LIST.validate_python(
        [
            {
                "name": "noop",
                "description": description,
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
    )


@pytest.mark.parametrize(
    ("large_system", "large_tool"),
    ((False, False), (True, False), (False, True), (True, True)),
    ids=("baseline", "large-system", "large-tool", "large-system-and-tool"),
)
def test_messages_and_count_tokens_send_matching_gemini_context(
    gateway: Gateway, large_system: bool, large_tool: bool
) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        system: Final = (
            f"<cache-buster>{uuid4().hex}</cache-buster>\n"
            "This is a synthetic context-budget diagnostic. Return exactly OK." + (_UNIT * 100 if large_system else "")
            if large_system
            else None
        )
        tools: Final = _tools(_UNIT * 100) if large_tool else None
        body: Final = _messages(model, system, tools)
        generated: Final = gateway.request("POST", "/v1/messages", {**body, "max_tokens": 16})
        counted: Final = gateway.request("POST", "/v1/messages/count_tokens", body)
        assert generated.status_code == 200, generated.text
        assert counted.status_code == 200, counted.text
        assert _JSON_OBJECT.validate_json(counted.content)["input_tokens"] == _COUNT, counted.text
        requests: Final = peer.drain()
        generated_body: Final = _json_body(
            next(request for request in requests if request.target.endswith(":generateContent"))
        )
        count_body: Final = _json_body(next(request for request in requests if request.target.endswith(":countTokens")))
        assert count_body == _count_tokens_request_from_generate_content(generated_body)


_LOOKUP_TOOL: Final[JsonValue] = {
    "name": "lookup",
    "description": "Look up a value",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
}
_CONTENT_BLOCK_BODIES: Final[Mapping[str, Mapping[str, JsonValue]]] = MappingProxyType(
    {
        "base64-pdf": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize the document."},
                        {
                            "type": "document",
                            "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQK"},
                        },
                    ],
                }
            ],
        },
        "base64-image": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}},
                        {"type": "text", "text": "What is this?"},
                    ],
                }
            ],
        },
        "tool-turns": {
            "tools": [_LOOKUP_TOOL],
            "messages": [
                {"role": "user", "content": "Look up the value."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Looking it up."},
                        {"type": "tool_use", "id": "toolu_01", "name": "lookup", "input": {"query": "value"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "42"}],
                },
            ],
        },
    }
)


def _ai_studio_deployment(scenario: Scenario, api_base: str, **parameters: JsonValue) -> str:
    return scenario.model(model=f"gemini/{_MODEL}", api_base=api_base, api_key="scripted-gemini-key", **parameters)


def _counted_contents(count_body: Mapping[str, JsonValue]) -> JsonValue:
    if "contents" in count_body:
        return count_body["contents"]
    return _JSON_OBJECT.validate_python(count_body["generateContentRequest"])["contents"]


@pytest.mark.parametrize("provider", ("vertex_ai", "gemini"))
@pytest.mark.parametrize("case", sorted(_CONTENT_BLOCK_BODIES))
def test_count_tokens_sends_the_contents_messages_inference_sends(gateway: Gateway, provider: str, case: str) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = (
            _deployment(gateway, scenario, wire.url)
            if provider == "vertex_ai"
            else _ai_studio_deployment(scenario, wire.url)
        )
        body: Final = {"model": model, **_CONTENT_BLOCK_BODIES[case]}
        generated: Final = gateway.request("POST", "/v1/messages", {**body, "max_tokens": 16})
        counted: Final = gateway.request("POST", "/v1/messages/count_tokens", body)
        assert generated.status_code == 200, generated.text
        assert counted.status_code == 200, counted.text
        assert _JSON_OBJECT.validate_json(counted.content)["input_tokens"] == _COUNT, counted.text
        requests: Final = peer.drain()
        generated_body: Final = _json_body(
            next(request for request in requests if request.target.endswith(":generateContent"))
        )
        count_body: Final = _json_body(next(request for request in requests if request.target.endswith(":countTokens")))
        assert _counted_contents(count_body) == generated_body["contents"]


_UNCONVERTIBLE_MESSAGES: Final[JsonValue] = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Summarize the notes."},
            {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "notes"}},
            {"type": "search_result", "source": "https://example.com/notes", "title": "Notes", "content": []},
        ],
    },
    {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "search", "input": {}}]},
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01",
                "content": [
                    {
                        "type": "search_result",
                        "source": "https://example.com/hits",
                        "title": "Hits",
                        "content": [{"type": "text", "text": "hit"}],
                    }
                ],
            }
        ],
    },
]


@pytest.mark.parametrize("provider", ("vertex_ai", "gemini"))
def test_count_tokens_rejects_blocks_it_cannot_convert(gateway: Gateway, provider: str) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = (
            _deployment(gateway, scenario, wire.url)
            if provider == "vertex_ai"
            else _ai_studio_deployment(scenario, wire.url)
        )
        response: Final = gateway.request(
            "POST", "/v1/messages/count_tokens", {"model": model, "messages": _UNCONVERTIBLE_MESSAGES}
        )
        assert response.status_code == 400, response.text
        assert "document with a text source, search_result, search_result inside a tool_result" in response.text
        assert peer.drain() == ()


_FILE_TEXT: Final = "line of a long file that the agent read earlier. " * 180
_FILES: Final = 3
_READ_FILE_TOOL: Final[JsonValue] = {
    "name": "read_file",
    "description": "Read a file",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
}
_CLEAR_OLDER_TOOL_RESULTS: Final[JsonValue] = {
    "edits": [
        {
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": 500},
            "keep": {"type": "tool_uses", "value": 1},
        }
    ]
}
_COMPACT_UNDER_THE_MINIMUM_TRIGGER: Final[JsonValue] = {
    "edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 1000}}]
}


def _read_file_round(marker: str, index: int) -> tuple[JsonValue, ...]:
    return (
        {"role": "user", "content": f"Read file {index}."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": f"toolu_{index}", "name": "read_file", "input": {"path": f"{index}.txt"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": f"toolu_{index}",
                    "content": f"<file {marker}-{index}>{_FILE_TEXT}",
                }
            ],
        },
    )


def _agent_body(model: str, marker: str, options: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    rounds: Final = tuple(chain.from_iterable(_read_file_round(marker, index) for index in range(_FILES)))
    return {
        "model": model,
        "tools": [_READ_FILE_TOOL],
        "messages": [*rounds, {"role": "user", "content": "Summarize what you read."}],
        "context_management": _CLEAR_OLDER_TOOL_RESULTS,
        **options,
    }


def _files_kept(contents: JsonValue, marker: str) -> tuple[bool, ...]:
    sent: Final = json.dumps(contents)
    return tuple(f"<file {marker}-{index}>" in sent for index in range(_FILES))


def _contents_sent_upstream(
    gateway: Gateway, peer: _GeminiPeer, body: Mapping[str, JsonValue], headers: Mapping[str, str] | None = None
) -> tuple[JsonValue, JsonValue]:
    generated: Final = gateway.request("POST", "/v1/messages", {**body, "max_tokens": 16}, headers=headers)
    assert generated.status_code == 200, generated.text
    generate_requests: Final = tuple(request for request in peer.drain() if request.target.endswith(":generateContent"))
    counted: Final = gateway.request("POST", "/v1/messages/count_tokens", body, headers=headers)
    assert counted.status_code == 200, counted.text
    assert _JSON_OBJECT.validate_json(counted.content)["input_tokens"] == _COUNT, counted.text
    count_requests: Final = peer.drain()
    assert len(generate_requests) == 1
    assert [request.target.endswith(":countTokens") for request in count_requests] == [True]
    return _json_body(generate_requests[0])["contents"], _counted_contents(_json_body(count_requests[0]))


@pytest.mark.parametrize("provider", ("vertex_ai", "gemini"))
def test_count_tokens_counts_the_history_messages_sends_after_context_edits(gateway: Gateway, provider: str) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _provider_deployment(gateway, scenario, provider, wire.url)
        marker: Final = uuid4().hex
        generated, counted = _contents_sent_upstream(gateway, peer, _agent_body(model, marker, {}))
        assert _files_kept(generated, marker) == (False, False, True)
        assert counted == generated


@pytest.mark.parametrize(
    ("deployment_options", "request_options", "files_kept"),
    (
        ({"additional_drop_params": ["context_management"]}, {}, (True, True, True)),
        ({}, {"additional_drop_params": ["context_management"]}, (True, True, True)),
        ({"additional_drop_params": ["context_management"]}, {"additional_drop_params": []}, (False, False, True)),
    ),
    ids=("deployment-drops", "request-drops", "request-overrides-the-deployment-drop"),
)
def test_count_tokens_follows_the_context_management_opt_out_messages_follows(
    gateway: Gateway,
    deployment_options: Mapping[str, JsonValue],
    request_options: Mapping[str, JsonValue],
    files_kept: tuple[bool, ...],
) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _ai_studio_deployment(scenario, wire.url, **deployment_options)
        marker: Final = uuid4().hex
        generated, counted = _contents_sent_upstream(gateway, peer, _agent_body(model, marker, request_options))
        assert _files_kept(generated, marker) == files_kept
        assert counted == generated


_RoutedCase: TypeAlias = Literal[
    "router-default-drops",
    "fallback-deployment-drops",
    "fallback-router-default-drops",
    "fallback-request-drops",
    "alias-to-a-litellm-model-drops",
    "completion-model-drops",
]
_DROPS_CONTEXT_MANAGEMENT: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"additional_drop_params": ["context_management"]}
)


@dataclass(frozen=True, slots=True)
class _RoutedProxy:
    settings: Mapping[str, JsonValue]
    requested: str
    request_options: Mapping[str, JsonValue]


def _scripted_gemini(model_name: str, model: str, api_base: str, options: Mapping[str, JsonValue]) -> JsonValue:
    return {
        "model_name": model_name,
        "litellm_params": {"model": model, "api_base": api_base, "api_key": "scripted-gemini-key", **options},
    }


def _default_fallback_settings(
    prefix: str, api_base: str, deployment_options: Mapping[str, JsonValue], router_defaults: Mapping[str, JsonValue]
) -> Mapping[str, JsonValue]:
    return {
        "model_list": [
            _scripted_gemini(f"{prefix}-fallback", f"gemini/{_MODEL}", api_base, deployment_options),
            _scripted_gemini(f"{prefix}-wild/*", "gemini/*", api_base, {}),
        ],
        "router_settings": {
            "default_fallbacks": [f"{prefix}-fallback"],
            **({"default_litellm_params": dict(router_defaults)} if router_defaults else {}),
        },
    }


def _routed_proxy(case: _RoutedCase, prefix: str, api_base: str) -> _RoutedProxy:
    match case:
        case "router-default-drops":
            return _RoutedProxy(
                {
                    "model_list": [_scripted_gemini(prefix, f"gemini/{_MODEL}", api_base, {})],
                    "router_settings": {"default_litellm_params": dict(_DROPS_CONTEXT_MANAGEMENT)},
                },
                prefix,
                {},
            )
        case "fallback-deployment-drops":
            return _RoutedProxy(
                _default_fallback_settings(prefix, api_base, _DROPS_CONTEXT_MANAGEMENT, {}), f"{prefix}-unrouted", {}
            )
        case "fallback-router-default-drops":
            return _RoutedProxy(
                _default_fallback_settings(prefix, api_base, {}, _DROPS_CONTEXT_MANAGEMENT), f"{prefix}-unrouted", {}
            )
        case "fallback-request-drops":
            return _RoutedProxy(
                _default_fallback_settings(prefix, api_base, {}, {}), f"{prefix}-unrouted", _DROPS_CONTEXT_MANAGEMENT
            )
        case "alias-to-a-litellm-model-drops":
            return _RoutedProxy(
                {
                    "model_list": [_scripted_gemini(prefix, f"gemini/{_MODEL}", api_base, _DROPS_CONTEXT_MANAGEMENT)],
                    "router_settings": {"model_group_alias": {f"{prefix}-alias": f"gemini/{_MODEL}"}},
                },
                f"{prefix}-alias",
                {},
            )
        case "completion-model-drops":
            return _RoutedProxy(
                {
                    "model_list": [
                        _scripted_gemini(f"{prefix}-keeps", f"gemini/{_MODEL}", api_base, {}),
                        _scripted_gemini(f"{prefix}-drops", f"gemini/{_MODEL}", api_base, _DROPS_CONTEXT_MANAGEMENT),
                    ],
                    "general_settings": {"completion_model": f"{prefix}-drops"},
                },
                f"{prefix}-keeps",
                {},
            )
        case _:
            assert_never(case)


@pytest.mark.parametrize("case", get_args(_RoutedCase))
def test_count_tokens_follows_the_opt_out_of_the_deployment_proxy_settings_send_to(
    gateway: Gateway, tmp_path: Path, case: _RoutedCase
) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire:
        routed: Final = _routed_proxy(case, f"routed-{uuid4().hex}", wire.url)
        config: Final = tmp_path / "routed.yaml"
        config.write_text(json.dumps(routed.settings), encoding="utf-8")
        with owned_proxy_process(gateway, tmp_path, {}, config=config) as owned:
            marker: Final = uuid4().hex
            generated, counted = _contents_sent_upstream(
                owned.gateway, peer, _agent_body(routed.requested, marker, routed.request_options)
            )
            assert _files_kept(generated, marker) == (True, True, True)
            assert counted == generated


def _named_deployment(
    scenario: Scenario,
    model_name: str,
    litellm_params: Mapping[str, JsonValue],
    model_info: Mapping[str, JsonValue],
) -> str:
    created: Final = scenario.gateway.post(
        "/model/new",
        {"model_name": model_name, "litellm_params": dict(litellm_params), "model_info": dict(model_info)},
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return identity


def test_count_tokens_reads_the_opt_out_of_a_deployment_id_beside_a_wildcard_route(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        prefix: Final = f"integration-{uuid4().hex}"
        _named_deployment(
            scenario,
            f"{prefix}-*",
            {"model": "gemini/*", "api_base": wire.url, "api_key": "scripted-gemini-key"},
            {},
        )
        deployment_id: Final = _named_deployment(
            scenario,
            prefix,
            {
                "model": f"gemini/{_MODEL}",
                "api_base": wire.url,
                "api_key": "scripted-gemini-key",
                "additional_drop_params": ["context_management"],
            },
            {"id": f"{prefix}-dropping"},
        )
        assert deployment_id == f"{prefix}-dropping"
        marker: Final = uuid4().hex
        generated, counted = _contents_sent_upstream(gateway, peer, _agent_body(deployment_id, marker, {}))
        assert _files_kept(generated, marker) == (True, True, True)
        assert counted == generated


@pytest.mark.parametrize(
    ("requested", "wildcards", "files_kept"),
    (
        ("dropping", (), (True, True, True)),
        ("editing", (), (False, False, True)),
        ("dropping", ("*",), (False, False, True)),
    ),
    ids=("deployment-drops", "sibling-deployment-drops", "wildcard-route-matches-first"),
)
def test_count_tokens_reads_the_opt_out_of_the_deployment_its_litellm_model_name_routes_to(
    gateway: Gateway, requested: str, wildcards: tuple[str, ...], files_kept: tuple[bool, ...]
) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        prefix: Final = f"gemini/{_MODEL}-{uuid4().hex}"
        scenario.model(
            model=f"{prefix}-dropping",
            api_base=wire.url,
            api_key="scripted-gemini-key",
            additional_drop_params=["context_management"],
        )
        scenario.model(model=f"{prefix}-editing", api_base=wire.url, api_key="scripted-gemini-key")
        for wildcard in wildcards:
            _named_deployment(
                scenario,
                f"{prefix}-{wildcard}",
                {"model": f"{prefix}-{wildcard}", "api_base": wire.url, "api_key": "scripted-gemini-key"},
                {},
            )
        marker: Final = uuid4().hex
        generated, counted = _contents_sent_upstream(gateway, peer, _agent_body(f"{prefix}-{requested}", marker, {}))
        assert _files_kept(generated, marker) == files_kept
        assert counted == generated


def _error_type(response: httpx.Response) -> JsonValue:
    return object_value(_JSON_OBJECT.validate_json(response.content)["error"])["type"]


def _error_message(response: httpx.Response) -> JsonValue:
    return object_value(_JSON_OBJECT.validate_json(response.content)["error"])["message"]


@pytest.mark.parametrize("provider", ("vertex_ai", "gemini"))
def test_count_tokens_rejects_a_context_edit_messages_rejects_without_an_upstream_call(
    gateway: Gateway, provider: str
) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _provider_deployment(gateway, scenario, provider, wire.url)
        body: Final = _agent_body(model, uuid4().hex, {"context_management": _COMPACT_UNDER_THE_MINIMUM_TRIGGER})
        generated: Final = gateway.request("POST", "/v1/messages", {**body, "max_tokens": 16})
        counted: Final = gateway.request("POST", "/v1/messages/count_tokens", body)
        assert generated.status_code == 400, generated.text
        assert counted.status_code == 400, counted.text
        assert _error_type(counted) == _error_type(generated)
        assert peer.drain() == ()


def test_count_tokens_rejects_an_invalid_edit_from_a_coding_agent_that_turns_drop_params_off(
    gateway: Gateway,
) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _ai_studio_deployment(scenario, wire.url)
        body: Final = _agent_body(
            model, uuid4().hex, {"context_management": _COMPACT_UNDER_THE_MINIMUM_TRIGGER, "drop_params": False}
        )
        claude_code: Final = MappingProxyType({"User-Agent": "claude-cli/2.1.0 (external, cli)"})
        generated: Final = gateway.request("POST", "/v1/messages", {**body, "max_tokens": 16}, headers=claude_code)
        counted: Final = gateway.request("POST", "/v1/messages/count_tokens", body, headers=claude_code)
        assert generated.status_code == 400, generated.text
        assert counted.status_code == 400, counted.text
        assert _error_type(counted) == _error_type(generated)
        assert peer.drain() == ()


@pytest.mark.parametrize(
    ("deployment_options", "request_options", "user_agent"),
    (
        ({"drop_params": True}, {}, "python-httpx"),
        ({}, {"drop_params": True}, "python-httpx"),
        ({}, {}, "claude-cli/2.1.0 (external, cli)"),
        ({}, {}, "codex_cli_rs/0.50.0"),
    ),
    ids=("deployment-drop-params", "request-drop-params", "claude-code-client", "codex-client"),
)
def test_count_tokens_counts_the_unedited_body_messages_sends_when_drop_params_drops_an_invalid_edit(
    gateway: Gateway,
    deployment_options: Mapping[str, JsonValue],
    request_options: Mapping[str, JsonValue],
    user_agent: str,
) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _ai_studio_deployment(scenario, wire.url, **deployment_options)
        marker: Final = uuid4().hex
        generated, counted = _contents_sent_upstream(
            gateway,
            peer,
            _agent_body(model, marker, {"context_management": _COMPACT_UNDER_THE_MINIMUM_TRIGGER, **request_options}),
            {"User-Agent": user_agent},
        )
        assert _files_kept(generated, marker) == (True, True, True)
        assert counted == generated


def test_count_tokens_answers_for_the_unedited_body_when_a_context_edit_fails_inside_the_proxy(
    gateway: Gateway,
) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _ai_studio_deployment(scenario, wire.url)
        unedited: Final = gateway.request(
            "POST", "/v1/messages/count_tokens", {"model": model, "messages": _UNCONVERTIBLE_MESSAGES}
        )
        edited: Final = gateway.request(
            "POST",
            "/v1/messages/count_tokens",
            {"model": model, "messages": _UNCONVERTIBLE_MESSAGES, "context_management": _CLEAR_OLDER_TOOL_RESULTS},
        )
        assert unedited.status_code == 400, unedited.text
        assert (edited.status_code, _error_type(edited), _error_message(edited)) == (
            unedited.status_code,
            _error_type(unedited),
            _error_message(unedited),
        )
        assert peer.drain() == ()


def test_count_tokens_keeps_each_text_block_in_system_instruction(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        system: Final[JsonValue] = [
            {"type": "text", "text": "First instruction"},
            {"type": "text", "text": "Second instruction"},
        ]
        response: Final = gateway.request("POST", "/v1/messages/count_tokens", _messages(model, system, None))
        assert response.status_code == 200, response.text
        requests: Final = _count_requests(wire)
        assert len(requests) == 1
        assert _json_body(requests[0]) == {
            "contents": [{"role": "user", "parts": [{"text": _USER}]}],
            "systemInstruction": {"parts": [{"text": "First instruction"}, {"text": "Second instruction"}]},
        }


def test_count_tokens_without_system_or_tools_sends_contents_only(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request("POST", "/v1/messages/count_tokens", _messages(model, None, None))
        assert response.status_code == 200, response.text
        requests: Final = _count_requests(wire)
        assert len(requests) == 1
        assert _json_body(requests[0]) == {"contents": [{"role": "user", "parts": [{"text": _USER}]}]}


_SDK_COUNT_ARGS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "messages": [{"role": "user", "content": _USER}],
        "system": "Follow the system instruction.",
        "tools": [
            {
                "name": "noop",
                "description": "Return no output",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    }
)


def _sync_sdk_count_tokens(gateway: Gateway, model: str) -> int:
    client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
    return client.messages.count_tokens(model=model, **_SDK_COUNT_ARGS).input_tokens  # pyright: ignore[reportArgumentType]  # SDK TypedDicts reject a JsonValue mapping


async def _async_sdk_count_tokens(gateway: Gateway, model: str) -> int:
    client: Final = anthropic.AsyncAnthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
    return (await client.messages.count_tokens(model=model, **_SDK_COUNT_ARGS)).input_tokens  # pyright: ignore[reportArgumentType]  # SDK TypedDicts reject a JsonValue mapping


@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync-sdk", "async-sdk"))
def test_anthropic_sdk_count_tokens_forwards_system_and_tools(gateway: Gateway, asynchronous: bool) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        input_tokens: Final = (
            asyncio.run(_async_sdk_count_tokens(gateway, model))
            if asynchronous
            else _sync_sdk_count_tokens(gateway, model)
        )
        assert input_tokens == _COUNT
        requests: Final = _count_requests(wire)
        assert len(requests) == 1
        request_body: Final = _json_body(requests[0])
        assert request_body["systemInstruction"] == {"parts": [{"text": "Follow the system instruction."}]}
        assert request_body["tools"] == [
            {
                "functionDeclarations": [
                    {"name": "noop", "description": "Return no output", "parameters": {"type": "object"}}
                ]
            }
        ]


def test_google_count_tokens_forwards_system_and_tools_to_vertex(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        body: Final[Mapping[str, JsonValue]] = {
            "contents": [
                {"role": "user", "parts": [{"text": _USER}]},
                {"role": "model", "parts": [{"functionCall": {"name": "noop", "args": {"query": "value"}}}]},
                {
                    "role": "user",
                    "parts": [{"functionResponse": {"name": "noop", "response": {"result": "ok"}}}],
                },
            ],
            "systemInstruction": {"parts": [{"text": "Follow this instruction."}]},
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "noop",
                            "description": "Return no output",
                            "parameters": {"type": "OBJECT", "properties": {}},
                        }
                    ]
                }
            ],
        }
        response: Final = gateway.request("POST", f"/v1beta/models/{model}:countTokens", body)
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["totalTokens"] == _COUNT, response.text
        requests: Final = _count_requests(wire)
        assert len(requests) == 1
        request_body: Final = _json_body(requests[0])
        assert request_body["contents"] == body["contents"]
        assert request_body["systemInstruction"] == body["systemInstruction"]
        assert request_body["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "noop",
                        "description": "Return no output",
                        "parameters": {"type": "object"},
                    }
                ]
            }
        ]


def test_responses_input_tokens_maps_flat_function_tools_to_vertex(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        body: Final[Mapping[str, JsonValue]] = {
            "model": model,
            "instructions": "Follow the system instruction.",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_lookup",
                    "name": "lookup",
                    "arguments": '{"query":"value"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_lookup",
                    "output": "result",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }
        response: Final = gateway.request("POST", "/v1/responses/input_tokens", body)
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["input_tokens"] == _COUNT, response.text
        requests: Final = _count_requests(wire)
        assert len(requests) == 1
        request_body: Final = _json_body(requests[0])
        assert request_body["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "lookup",
                        "description": "Look up a value",
                        "parameters": {"type": "object"},
                    }
                ]
            }
        ]


def test_utils_token_counter_maps_openai_function_tools_to_vertex(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        tool_schema: Final[JsonValue] = {"type": "object", "properties": {}}
        body: Final[Mapping[str, JsonValue]] = {
            "model": model,
            "messages": _MESSAGES,
            "system": "Follow the system instruction.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "Return no output",
                        "parameters": tool_schema,
                    },
                }
            ],
        }
        response: Final = gateway.request("POST", "/utils/token_counter", body, params={"call_endpoint": "true"})
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["total_tokens"] == _COUNT, response.text
        requests: Final = _count_requests(wire)
        assert len(requests) == 1
        request_body: Final = _json_body(requests[0])
        assert request_body["systemInstruction"] == {"parts": [{"text": "Follow the system instruction."}]}
        assert request_body["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "noop",
                        "description": "Return no output",
                        "parameters": {"type": "object"},
                    }
                ]
            }
        ]


def test_count_tokens_surfaces_peer_errors_and_rejects_non_text_system(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    peer.set_status(400)
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request("POST", "/v1/messages/count_tokens", _messages(model, None, None))
        assert response.status_code == 400, response.text
        assert len(_count_requests(wire)) == 1

    invalid_peer: Final = _GeminiPeer()
    with wire_server(invalid_peer) as invalid_wire, gateway.scenario() as scenario:
        invalid_model: Final = _deployment(gateway, scenario, invalid_wire.url)
        invalid_response: Final = gateway.request(
            "POST",
            "/v1/messages/count_tokens",
            _messages(
                invalid_model,
                [{"type": "image", "source": {"type": "base64", "data": "AAAA"}}],
                None,
            ),
        )
        assert invalid_response.status_code == 400, invalid_response.text
        assert _count_requests(invalid_wire) == ()


@pytest.mark.parametrize(
    "tools",
    (
        [{"name": "noop", "input_schema": {"type": "object", "properties": {"query": "not-a-schema"}}}],
        [{"name": "noop", "input_schema": "not-a-schema"}],
    ),
    ids=("malformed-property", "string-schema"),
)
def test_count_tokens_rejects_unconvertible_tools_without_request(gateway: Gateway, tools: JsonValue) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request("POST", "/v1/messages/count_tokens", _messages(model, None, tools))
        assert response.status_code == 400, response.text
        assert "tools" in response.text
        assert _count_requests(wire) == ()


def _count_wave(clients: Sequence[httpx.Client], body: Mapping[str, JsonValue]) -> tuple[tuple[int, JsonValue], ...]:
    def count(client: httpx.Client) -> tuple[int, JsonValue]:
        response: Final = client.post("/v1/messages/count_tokens", json=dict(body))
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        return response.status_code, payload.get("input_tokens")

    with ThreadPoolExecutor(max_workers=len(clients)) as executor:
        return tuple(executor.map(count, clients))


def test_concurrent_count_waves_recover_after_vertex_count_outage(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario, ExitStack() as stack:
        model: Final = _deployment(gateway, scenario, wire.url)
        clients: Final = tuple(
            stack.enter_context(
                httpx.Client(
                    base_url=str(gateway.client.base_url),
                    headers={"Authorization": f"Bearer {gateway.key}"},
                    timeout=30,
                    trust_env=False,
                )
            )
            for _ in range(3)
        )
        body: Final = _messages(model, None, None)
        assert _count_wave(clients, body) == ((200, _COUNT), (200, _COUNT), (200, _COUNT))
        peer.set_status(503)
        assert _count_wave(clients, body) == ((503, None), (503, None), (503, None))
        peer.set_status(200)
        assert _count_wave(clients, body) == ((200, _COUNT), (200, _COUNT), (200, _COUNT))
        requests: Final = _count_requests(wire)
        assert len(requests) == 9
        assert (
            tuple(_json_body(request) for request in requests)
            == ({"contents": [{"role": "user", "parts": [{"text": _USER}]}]},) * 9
        )


_CONTENTS: Final[JsonValue] = [{"role": "user", "parts": [{"text": _USER}]}]
_OPENAI_TOOLS: Final[JsonValue] = [
    {
        "type": "function",
        "function": {"name": "noop", "description": "Return no output", "parameters": {"type": "object"}},
    }
]


def _provider_deployment(
    gateway: Gateway, scenario: Scenario, provider: str, api_base: str, **parameters: JsonValue
) -> str:
    if provider == "vertex_ai":
        return _deployment(gateway, scenario, api_base, **parameters)
    return _ai_studio_deployment(scenario, api_base, **parameters)


def _count(gateway: Gateway, body: Mapping[str, JsonValue], call_endpoint: bool = True) -> httpx.Response:
    return gateway.request(
        "POST", "/utils/token_counter", body, params={"call_endpoint": "true" if call_endpoint else "false"}
    )


def _total_tokens(response: httpx.Response) -> JsonValue:
    return _JSON_OBJECT.validate_json(response.content)["total_tokens"]


@pytest.mark.parametrize(
    ("provider", "empty_conversation"),
    (("vertex_ai", {}), ("gemini", {}), ("gemini", {"messages": []}), ("gemini", {"contents": []})),
    ids=("vertex-prompt-only", "ai-studio-prompt-only", "ai-studio-empty-messages", "ai-studio-empty-contents"),
)
def test_utils_token_counter_sends_a_prompt_as_the_user_turn_of_its_messages_form(
    gateway: Gateway, provider: str, empty_conversation: Mapping[str, JsonValue]
) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _provider_deployment(gateway, scenario, provider, wire.url)
        prompted: Final = _count(gateway, {"model": model, "prompt": _USER, **empty_conversation})
        prompted_requests: Final = _count_requests(wire)
        as_message: Final = _count(gateway, {"model": model, "messages": _MESSAGES})
        message_requests: Final = _count_requests(wire)
    assert prompted.status_code == 200, prompted.text
    assert as_message.status_code == 200, as_message.text
    assert _total_tokens(prompted) == _COUNT, prompted.text
    assert len(prompted_requests) == 1
    assert len(message_requests) == 1
    assert _counted_contents(_json_body(prompted_requests[0])) == _counted_contents(_json_body(message_requests[0]))


@pytest.mark.parametrize(
    ("conversation", "call_endpoint"),
    (({"messages": _MESSAGES}, True), ({"contents": _CONTENTS}, True), ({"messages": _MESSAGES}, False)),
    ids=("provider-count-messages", "provider-count-contents", "local-count-messages"),
)
def test_utils_token_counter_rejects_a_prompt_next_to_a_conversation_without_calling_the_provider(
    gateway: Gateway, conversation: Mapping[str, JsonValue], call_endpoint: bool
) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _ai_studio_deployment(scenario, wire.url)
        response: Final = _count(gateway, {"model": model, "prompt": "hi", **conversation}, call_endpoint)
        requests: Final = _count_requests(wire)
    assert response.status_code == 400, response.text
    assert _JSON_OBJECT.validate_python(_JSON_OBJECT.validate_json(response.content)["error"])["param"] == "prompt"
    assert requests == ()


@pytest.mark.parametrize("call_endpoint", (True, False), ids=("provider-count", "local-count"))
def test_utils_token_counter_counts_only_the_conversation_when_drop_params_drops_the_prompt(
    gateway: Gateway, call_endpoint: bool
) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _ai_studio_deployment(scenario, wire.url, drop_params=True)
        mixed: Final = _count(gateway, {"model": model, "prompt": "hi", "messages": _MESSAGES}, call_endpoint)
        mixed_requests: Final = _count_requests(wire)
        alone: Final = _count(gateway, {"model": model, "messages": _MESSAGES}, call_endpoint)
        alone_requests: Final = _count_requests(wire)
    assert mixed.status_code == 200, mixed.text
    assert alone.status_code == 200, alone.text
    assert _total_tokens(mixed) == _total_tokens(alone)
    assert [_counted_contents(_json_body(request)) for request in mixed_requests] == [
        _counted_contents(_json_body(request)) for request in alone_requests
    ]


@pytest.mark.parametrize(
    "extra",
    ({"system": "Follow the system instruction."}, {"tools": _OPENAI_TOOLS}),
    ids=("system", "tools"),
)
def test_utils_token_counter_counts_a_local_prompt_with_system_or_tools_like_its_messages_form(
    gateway: Gateway, extra: Mapping[str, JsonValue]
) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _ai_studio_deployment(scenario, wire.url)
        prompted: Final = _count(gateway, {"model": model, "prompt": _USER, **extra}, call_endpoint=False)
        as_message: Final = _count(gateway, {"model": model, "messages": _MESSAGES, **extra}, call_endpoint=False)
        requests: Final = _count_requests(wire)
    assert prompted.status_code == 200, prompted.text
    assert as_message.status_code == 200, as_message.text
    assert _total_tokens(prompted) == _total_tokens(as_message)
    assert requests == ()


_RESPONSES_TOOL_ROUND_TRIP: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "instructions": "Follow the system instruction.",
        "input": [
            {"role": "user", "content": "Look up the value."},
            {"type": "function_call", "call_id": "call_lookup", "name": "lookup", "arguments": '{"query":"value"}'},
            {"type": "function_call_output", "call_id": "call_lookup", "output": "result"},
        ],
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ],
    }
)


@pytest.mark.parametrize("provider", ("vertex_ai", "gemini"))
def test_responses_input_tokens_counts_the_tool_round_trip_contents_responses_inference_sends(
    gateway: Gateway, provider: str
) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = (
            _deployment(gateway, scenario, wire.url)
            if provider == "vertex_ai"
            else _ai_studio_deployment(scenario, wire.url)
        )
        body: Final = {"model": model, **_RESPONSES_TOOL_ROUND_TRIP}
        generated: Final = gateway.request("POST", "/v1/responses", {**body, "max_output_tokens": 16})
        counted: Final = gateway.request("POST", "/v1/responses/input_tokens", body)
        assert generated.status_code == 200, generated.text
        assert counted.status_code == 200, counted.text
        assert _JSON_OBJECT.validate_json(counted.content)["input_tokens"] == _COUNT, counted.text
        requests: Final = peer.drain()
        (generated_request,) = (request for request in requests if request.target.endswith(":generateContent"))
        (count_request,) = (request for request in requests if request.target.endswith(":countTokens"))
        instruction: Final[JsonValue] = {"text": _RESPONSES_TOOL_ROUND_TRIP["instructions"]}
        first_turn, *later_turns = _JSON_LIST.validate_python(_counted_contents(_json_body(count_request)))
        first_parts: Final = _JSON_LIST.validate_python(_JSON_OBJECT.validate_python(first_turn)["parts"])
        assert instruction in first_parts, count_request.body
        counted_turns: Final = [
            {
                **_JSON_OBJECT.validate_python(first_turn),
                "parts": [part for part in first_parts if part != instruction],
            },
            *later_turns,
        ]
        assert counted_turns == _json_body(generated_request)["contents"], count_request.body
