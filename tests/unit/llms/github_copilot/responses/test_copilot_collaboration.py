import copy
import datetime
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.github_copilot.responses.collaboration import CopilotPlaintextCollaboration
from litellm.llms.github_copilot.responses.transformation import GithubCopilotResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams

CLIENT_NAMESPACE: Final = "collaboration"
UPSTREAM_NAMESPACE: Final = "codex_collaboration_plaintext"
JSON_OBJECT: Final = TypeAdapter(dict[str, object])
MESSAGE: Final = 'Keep all content, including unicode λ and "quotes".\nSecond line.'
ARGUMENTS: Final = '{"message":"hello","target":"child"}'


def _tools(namespace: str = CLIENT_NAMESPACE, encrypted: bool = True) -> list[dict[str, object]]:
    return [
        {
            "type": "namespace",
            "name": namespace,
            "description": "Agent communication",
            "tools": [
                {
                    "type": "function",
                    "name": action,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Full message",
                                **({"encrypted": True} if encrypted else {}),
                            },
                            "target": {"type": "string", "encrypted": True},
                        },
                        "required": ["message"],
                    },
                }
                for action in ("spawn_agent", "send_message", "followup_task")
            ]
            + [{"type": "function", "name": "wait_agent", "parameters": {"type": "object"}}],
        }
    ]


def _call(namespace: str = UPSTREAM_NAMESPACE, name: str = "send_message") -> dict[str, object]:
    return {
        "type": "function_call",
        "id": "fc_call",
        "call_id": "call_1",
        "name": name,
        "namespace": namespace,
        "arguments": ARGUMENTS,
        "status": "completed",
        "provider_extension": {"keep": True},
    }


def _response(output: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-6-astra",
        "output": output,
        "usage": {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13},
        "provider_extension": {"keep": True},
    }


def _logging(stream: bool = False) -> Logging:
    return Logging(
        model="gpt-6-astra",
        messages="hello",
        stream=stream,
        call_type="responses",
        start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="call_logging",
        function_id="function_logging",
        supports_correlation_logging=False,
    )


def _activate(config: GithubCopilotResponsesAPIConfig) -> dict[str, object]:
    return config.transform_responses_api_request(
        model="gpt-6-astra",
        input="hello",
        response_api_optional_request_params={"tools": _tools(), "store": False},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )


def test_request_projection_preserves_message_text_ciphertext_and_nonmessage_schema() -> None:
    adapter: Final = CopilotPlaintextCollaboration()
    history: Final = [
        {"role": "user", "content": MESSAGE},
        {**_call(CLIENT_NAMESPACE), "encrypted_function_args": [{"name": "message", "content": "old-ciphertext"}]},
        {"type": "function_call_output", "call_id": "call_1", "output": "unchanged result"},
        {"type": "agent_message", "encrypted_content": "historical-ciphertext"},
        {"type": "reasoning", "encrypted_content": "reasoning-ciphertext", "summary": []},
        {"type": "additional_tools", "tools": _tools()},
        {"type": "tool_search_output", "call_id": "search_1", "tools": _tools()},
    ]
    request: Final = {"model": "gpt-6-astra", "tools": _tools(), "input": history, "store": False}
    original: Final = copy.deepcopy(request)

    projected: Final = adapter.prepare_request(request)

    assert projected == {
        **request,
        "tools": _tools(UPSTREAM_NAMESPACE, encrypted=False),
        "input": [
            history[0],
            {**history[1], "namespace": UPSTREAM_NAMESPACE},
            *history[2:5],
            {"type": "additional_tools", "tools": _tools(UPSTREAM_NAMESPACE, encrypted=False)},
            {"type": "tool_search_output", "call_id": "search_1", "tools": _tools(UPSTREAM_NAMESPACE, encrypted=False)},
        ],
    }
    assert request == original


@pytest.mark.parametrize("inventory_type", ["additional_tools", "tool_search_output"])
def test_historical_tool_inventory_can_activate_projection(inventory_type: str) -> None:
    adapter: Final = CopilotPlaintextCollaboration()
    request: Final = {"input": [{"type": inventory_type, "tools": _tools()}]}

    assert adapter.prepare_request(request) == {
        "input": [{"type": inventory_type, "tools": _tools(UPSTREAM_NAMESPACE, encrypted=False)}],
        "store": False,
    }
    assert adapter.restore_response({"output": [_call()]}) == {
        "output": [{**_call(CLIENT_NAMESPACE), "encrypted_function_args": []}]
    }


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "namespace", "name": "collaboration", "tools": [{"type": "function", "name": "spawn_agent"}]},
        *_tools(encrypted=False),
        *_tools(namespace="ordinary_tools"),
        {
            "type": "namespace",
            "name": "collaboration",
            "tools": [
                {
                    "type": "function",
                    "name": "unrelated",
                    "parameters": {"properties": {"message": {"type": "string", "encrypted": True}}},
                }
            ],
        },
    ],
)
def test_unrelated_collaboration_schemas_and_unactivated_responses_are_identity_passthrough(
    tool: dict[str, object],
) -> None:
    adapter: Final = CopilotPlaintextCollaboration()
    request: Final = {"tools": [tool], "input": [{"role": "user", "content": MESSAGE}]}
    response: Final = {"output": [_call()]}
    event: Final = {"type": "response.output_item.added", "output_index": 0, "item": _call()}

    assert adapter.prepare_request(request) is request
    assert adapter.restore_response(response) is response
    assert adapter.restore_event(event) is event


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ({"type": "namespace", "name": CLIENT_NAMESPACE}, {"type": "namespace", "name": UPSTREAM_NAMESPACE}),
        (
            {"type": "function", "namespace": CLIENT_NAMESPACE, "name": "send_message"},
            {"type": "function", "namespace": UPSTREAM_NAMESPACE, "name": "send_message"},
        ),
        (
            {"type": "function", "name": "collaboration.spawn_agent"},
            {"type": "function", "name": "codex_collaboration_plaintext.spawn_agent"},
        ),
        (
            {"type": "function", "name": "collaboration__followup_task"},
            {"type": "function", "name": "codex_collaboration_plaintext__followup_task"},
        ),
    ],
)
def test_forced_and_allowed_tool_choices_follow_the_projected_inventory(
    choice: dict[str, object], expected: dict[str, object]
) -> None:
    adapter: Final = CopilotPlaintextCollaboration()
    request: Final = {"tools": _tools(), "tool_choice": choice}
    allowed: Final = {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [choice, {"type": "function", "name": "other"}],
    }

    assert adapter.prepare_request(request) == {
        "tools": _tools(UPSTREAM_NAMESPACE, encrypted=False),
        "tool_choice": expected,
        "store": False,
    }
    assert adapter.prepare_request({"tools": _tools(), "tool_choice": allowed}) == {
        "tools": _tools(UPSTREAM_NAMESPACE, encrypted=False),
        "tool_choice": {**allowed, "tools": [expected, {"type": "function", "name": "other"}]},
        "store": False,
    }


@pytest.mark.parametrize(
    "extra",
    [
        {"store": True},
        {"previous_response_id": "resp_previous"},
        {"conversation": "conv_previous"},
        {"input": [{"type": "additional_tools", "tools": _tools(UPSTREAM_NAMESPACE)}]},
    ],
)
def test_active_projection_rejects_stateful_requests_and_alias_collisions(extra: dict[str, object]) -> None:
    adapter: Final = CopilotPlaintextCollaboration()

    with pytest.raises(ValueError, match=r"stateless Responses|reserved by the Copilot adapter"):
        adapter.prepare_request({"tools": _tools(), **extra})


@pytest.mark.parametrize("separator", [".", "__"])
def test_qualified_history_and_returned_names_keep_call_identifiers(separator: str) -> None:
    adapter: Final = CopilotPlaintextCollaboration()
    history: Final = {
        "type": "function_call",
        "id": "fc_previous",
        "call_id": "call_previous",
        "name": CLIENT_NAMESPACE + separator + "spawn_agent",
        "arguments": ARGUMENTS,
    }

    assert adapter.prepare_request({"tools": _tools(), "input": [history]}) == {
        "tools": _tools(UPSTREAM_NAMESPACE, encrypted=False),
        "input": [{**history, "namespace": UPSTREAM_NAMESPACE, "name": "spawn_agent"}],
        "store": False,
    }
    assert adapter.restore_response(
        {"output": [{**history, "name": UPSTREAM_NAMESPACE + separator + "spawn_agent"}]}
    ) == {"output": [{**history, "namespace": CLIENT_NAMESPACE, "name": "spawn_agent", "encrypted_function_args": []}]}


def test_extra_body_tools_and_history_activate_projection_and_ordinary_replacement_disables_it() -> None:
    config: Final = GithubCopilotResponsesAPIConfig(plaintext_collaboration=True)
    primary: Final = config.transform_responses_api_request(
        model="gpt-6-astra",
        input="hello",
        response_api_optional_request_params={"store": False},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    overlay: Final = {
        "tools": _tools(),
        "tool_choice": {"type": "function", "namespace": CLIENT_NAMESPACE, "name": "send_message"},
        "input": [_call(CLIENT_NAMESPACE)],
    }

    merged: Final = config.transform_extra_body(overlay, primary, "gpt-6-astra", GenericLiteLLMParams())

    assert merged == {
        **primary,
        "tools": _tools(UPSTREAM_NAMESPACE, encrypted=False),
        "tool_choice": {"type": "function", "namespace": UPSTREAM_NAMESPACE, "name": "send_message"},
        "input": [_call()],
    }
    restored: Final = config.transform_response_api_response(
        "gpt-6-astra", httpx.Response(200, json=_response([_call()])), _logging()
    )
    assert JSON_OBJECT.validate_json(restored.model_dump_json(exclude_none=True))["output"] == [
        {**_call(CLIENT_NAMESPACE), "encrypted_function_args": []}
    ]

    replacement: Final = {"tools": _tools(encrypted=False)}
    assert config.transform_extra_body(replacement, primary, "gpt-6-astra", GenericLiteLLMParams()) == {
        **primary,
        **replacement,
    }
    untouched: Final = config.transform_response_api_response(
        "gpt-6-astra", httpx.Response(200, json=_response([_call(CLIENT_NAMESPACE)])), _logging()
    )
    assert JSON_OBJECT.validate_json(untouched.model_dump_json(exclude_none=True))["output"] == [
        _call(CLIENT_NAMESPACE)
    ]


@pytest.mark.parametrize(
    "overlay",
    [
        {"store": True},
        {"previous_response_id": "resp_previous"},
        {"conversation": {"id": "conv_previous"}},
        {"input": [{"type": "tool_search_output", "tools": _tools(UPSTREAM_NAMESPACE)}]},
    ],
)
def test_extra_body_cannot_bypass_stateless_and_namespace_guards(overlay: dict[str, object]) -> None:
    config: Final = GithubCopilotResponsesAPIConfig(plaintext_collaboration=True)
    primary: Final = _activate(config)

    with pytest.raises(ValueError, match=r"stateless Responses|reserved by the Copilot adapter"):
        config.transform_extra_body(overlay, primary, "gpt-6-astra", GenericLiteLLMParams())


def test_explicitly_disabled_provider_preserves_encryption_protocol() -> None:
    config: Final = GithubCopilotResponsesAPIConfig(plaintext_collaboration=False)
    raw: Final = httpx.Response(200, json=_response([_call(CLIENT_NAMESPACE)]))

    assert _activate(config)["tools"] == _tools()
    parsed: Final = config.transform_response_api_response("gpt-6-astra", raw, _logging())
    assert JSON_OBJECT.validate_json(parsed.model_dump_json(exclude_none=True))["output"] == [_call(CLIENT_NAMESPACE)]


def test_provider_nonstream_restoration_survives_serialization_and_keeps_headers_and_usage() -> None:
    config: Final = GithubCopilotResponsesAPIConfig(plaintext_collaboration=True)
    logging: Final = _logging()
    raw: Final = httpx.Response(
        200,
        json={**_response([_call(), _call(name="wait_agent")]), "tools": _tools(UPSTREAM_NAMESPACE, encrypted=False)},
        headers={"x-request-id": "request_1", "x-ratelimit-remaining-requests": "10"},
    )
    assert _activate(config)["tools"] == _tools(UPSTREAM_NAMESPACE, encrypted=False)

    response: Final = config.transform_response_api_response("gpt-6-astra", raw, logging)
    serialized: Final = JSON_OBJECT.validate_json(response.model_dump_json(exclude_none=True))

    assert serialized["output"] == [
        {**_call(CLIENT_NAMESPACE), "encrypted_function_args": []},
        _call(CLIENT_NAMESPACE, name="wait_agent"),
    ]
    assert serialized["tools"] == _tools(encrypted=False)
    assert serialized["provider_extension"] == {"keep": True}
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens) == (9, 4, 13)
    assert response.hidden_params["headers"] == dict(raw.headers)
    assert response.hidden_params["additional_headers"]
    assert logging.model_call_details["original_response"] == raw.text


def _stream(config: GithubCopilotResponsesAPIConfig, event: dict[str, object]) -> dict[str, object]:
    return JSON_OBJECT.validate_json(
        config.transform_streaming_response("gpt-6-astra", event, _logging(stream=True)).model_dump_json(
            exclude_none=True
        )
    )


@pytest.mark.parametrize("identity", [{}, {"name": None, "namespace": None}, {"name": "", "namespace": ""}])
def test_stream_restores_sparse_events_and_completed_snapshot_through_real_event_models(
    identity: dict[str, object],
) -> None:
    config: Final = GithubCopilotResponsesAPIConfig(plaintext_collaboration=True)
    _activate(config)
    added: Final = _stream(config, {"type": "response.output_item.added", "output_index": 0, "item": _call()})
    delta: Final = _stream(
        config,
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fc_changed",
            "namespace": UPSTREAM_NAMESPACE,
            "name": "send_message",
            "delta": ARGUMENTS[:12],
        },
    )
    arguments: Final = _stream(
        config,
        {
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "item_id": "fc_changed",
            "arguments": ARGUMENTS,
            **identity,
        },
    )
    done: Final = _stream(
        config,
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call", "id": "fc_changed", "arguments": ARGUMENTS, **identity},
        },
    )
    completed: Final = _stream(config, {"type": "response.completed", "response": _response([_call()])})

    assert added["item"] == {**_call(CLIENT_NAMESPACE), "encrypted_function_args": []}
    assert delta == {
        "type": "response.function_call_arguments.delta",
        "output_index": 0,
        "item_id": "fc_call",
        "namespace": CLIENT_NAMESPACE,
        "name": "send_message",
        "delta": ARGUMENTS[:12],
    }
    assert arguments == {
        "type": "response.function_call_arguments.done",
        "output_index": 0,
        "item_id": "fc_call",
        "call_id": "call_1",
        "arguments": ARGUMENTS,
        "namespace": CLIENT_NAMESPACE,
        "name": "send_message",
        "encrypted_function_args": [],
    }
    assert done["item"] == {
        "type": "function_call",
        "id": "fc_call",
        "call_id": "call_1",
        "arguments": ARGUMENTS,
        "name": "send_message",
        "namespace": CLIENT_NAMESPACE,
        "encrypted_function_args": [],
    }
    snapshot: Final = JSON_OBJECT.validate_python(completed["response"])
    assert snapshot["output"] == [{**_call(CLIENT_NAMESPACE), "encrypted_function_args": []}]


def test_custom_calls_keep_their_own_encryption_metadata_without_a_plaintext_dispatch_marker() -> None:
    adapter: Final = CopilotPlaintextCollaboration()
    adapter.prepare_request({"tools": _tools()})
    call: Final = {
        "type": "custom_tool_call",
        "call_id": "custom_1",
        "namespace": UPSTREAM_NAMESPACE,
        "name": "send_message",
        "input": "opaque custom payload",
        "encrypted_function_args": ["provider_metadata"],
    }
    restored: Final = {**call, "namespace": CLIENT_NAMESPACE}

    assert adapter.restore_response({"output": [call]}) == {"output": [restored]}
    assert adapter.restore_event({"type": "response.output_item.done", "output_index": 0, "item": call}) == {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": restored,
    }


@pytest.mark.parametrize("namespace", ["different_namespace", CLIENT_NAMESPACE])
def test_stream_rejects_conflicting_tool_identity(namespace: str) -> None:
    adapter: Final = CopilotPlaintextCollaboration()
    adapter.prepare_request({"tools": _tools()})
    adapter.restore_event({"type": "response.output_item.added", "output_index": 0, "item": _call()})

    with pytest.raises(ValueError, match="Conflicting Copilot collaboration identity"):
        adapter.restore_event({"type": "response.output_item.done", "output_index": 0, "item": _call(namespace)})


@pytest.mark.parametrize("encrypted_args", [["message"], [{"name": "message", "content": "ciphertext"}], None])
def test_provider_never_labels_encrypted_arguments_as_plaintext(encrypted_args: object) -> None:
    adapter: Final = CopilotPlaintextCollaboration()
    adapter.prepare_request({"tools": _tools()})
    output: Final = {**_call(), "encrypted_function_args": encrypted_args}

    with pytest.raises(ValueError, match="encrypted arguments for a plaintext collaboration tool"):
        adapter.restore_response({"output": [output]})

    with pytest.raises(ValueError, match="encrypted arguments for a plaintext collaboration tool"):
        adapter.restore_event({"type": "response.output_item.added", "output_index": 0, "item": output})
