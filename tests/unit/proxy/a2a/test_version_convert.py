"""Unit tests for A2A protocol version normalization in
litellm/proxy/a2a/version_convert.py.

These assert the conversion actually changes wire shape in the right direction and
preserves core fields on a round trip, so a mutation that no-ops or flips the direction
fails the suite.
"""

import pytest

from litellm.proxy.a2a.version_convert import (
    normalize_agent_card,
    normalize_jsonrpc_response,
    normalize_request_params,
    normalize_stream_event,
)

a2a = pytest.importorskip("a2a.compat.v0_3.conversions")


def _rpc(result: dict, request_id: str = "1") -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


V03_MESSAGE = {
    "kind": "message",
    "messageId": "m1",
    "role": "agent",
    "parts": [{"kind": "text", "text": "hi"}],
}

V03_TASK = {
    "kind": "task",
    "id": "t1",
    "contextId": "c1",
    "status": {"state": "completed"},
}

V03_STATUS_UPDATE = {
    "kind": "status-update",
    "taskId": "t1",
    "contextId": "c1",
    "status": {"state": "working"},
    "final": False,
}

V03_ARTIFACT_UPDATE = {
    "kind": "artifact-update",
    "taskId": "t1",
    "contextId": "c1",
    "artifact": {"artifactId": "a1", "parts": [{"kind": "text", "text": "out"}]},
}


def test_send_result_0_3_to_1_0_wraps_in_envelope():
    out = normalize_jsonrpc_response(_rpc(V03_MESSAGE), "1.0", method="message/send")
    assert "message" in out["result"]
    assert "kind" not in out["result"]
    assert out["result"]["message"]["messageId"] == "m1"


def test_send_result_1_0_to_0_3_unwraps_to_bare_kind():
    v1 = normalize_jsonrpc_response(_rpc(V03_MESSAGE), "1.0", method="message/send")
    out = normalize_jsonrpc_response(v1, "0.3", method="message/send")
    assert out["result"]["kind"] == "message"
    assert out["result"]["messageId"] == "m1"
    assert out["result"]["parts"][0]["text"] == "hi"


def test_send_result_same_version_is_identity_passthrough():
    rpc = _rpc(V03_MESSAGE)
    out = normalize_jsonrpc_response(rpc, "0.3", method="message/send")
    assert out is rpc


def test_task_result_round_trip_preserves_ids():
    v1 = normalize_jsonrpc_response(_rpc(V03_TASK), "1.0", method="tasks/get")
    assert "kind" not in v1["result"]
    assert v1["result"]["id"] == "t1"
    back = normalize_jsonrpc_response(v1, "0.3", method="tasks/get")
    assert back["result"]["kind"] == "task"
    assert back["result"]["id"] == "t1"
    assert back["result"]["contextId"] == "c1"


def test_error_response_passes_through_untouched():
    err = {"jsonrpc": "2.0", "id": "1", "error": {"code": -32600, "message": "bad"}}
    assert normalize_jsonrpc_response(err, "1.0", method="message/send") is err


def test_malformed_result_falls_back_to_passthrough():
    # A 0.3 message missing required fields can't validate; conversion must not raise.
    rpc = _rpc({"kind": "message"})
    out = normalize_jsonrpc_response(rpc, "1.0", method="message/send")
    assert out["result"] == {"kind": "message"}


def test_unknown_shape_passes_through():
    rpc = _rpc({"unexpected": "shape"})
    out = normalize_jsonrpc_response(rpc, "1.0", method="message/send")
    assert out is rpc


@pytest.mark.parametrize("event", [V03_STATUS_UPDATE, V03_ARTIFACT_UPDATE])
def test_stream_event_round_trip_preserves_kind(event):
    v1 = normalize_stream_event(_rpc(event), "1.0", request_id="1")
    assert "kind" not in v1["result"]
    back = normalize_stream_event(v1, "0.3", request_id="1")
    assert back["result"]["kind"] == event["kind"]
    assert back["result"]["taskId"] == "t1"


def test_stream_event_envelope_key_for_status_update():
    v1 = normalize_stream_event(_rpc(V03_STATUS_UPDATE), "1.0", request_id="1")
    assert "statusUpdate" in v1["result"]


def test_request_params_lowering_is_noop_for_0_3():
    params = {"id": "t1", "historyLength": 5}
    assert normalize_request_params(params, "0.3", method="tasks/get") is params


def test_request_params_lowering_get_task_to_0_3():
    out = normalize_request_params(
        {"id": "t1", "historyLength": 5}, "1.0", method="tasks/get"
    )
    assert out["id"] == "t1"
    assert out["historyLength"] == 5


def test_request_params_lowering_create_push_notification_config_preserves_task_id():
    out = normalize_request_params(
        {
            "parent": "tasks/task-1",
            "configId": "cfg-1",
            "config": {"url": "https://webhook.example.com"},
        },
        "1.0",
        method="tasks/pushNotificationConfig/set",
    )
    assert out["taskId"] == "task-1"
    assert out["pushNotificationConfig"]["url"] == "https://webhook.example.com"
    assert out["pushNotificationConfig"]["id"] == "cfg-1"


def test_flatten_create_push_notification_drops_redundant_envelope_key():
    from litellm.proxy.a2a.version_convert import (
        _flatten_create_push_notification_params,
    )

    flat = _flatten_create_push_notification_params(
        {
            "parent": "tasks/task-1",
            "config": {"url": "https://chosen.example.com"},
            "pushNotificationConfig": {"url": "https://ignored.example.com"},
        }
    )
    assert flat["url"] == "https://chosen.example.com"
    assert "pushNotificationConfig" not in flat
    assert "config" not in flat


def test_request_params_lowering_list_tasks_to_0_3():
    out = normalize_request_params(
        {
            "contextId": "ctx-1",
            "pageSize": 10,
            "status": "TASK_STATE_COMPLETED",
        },
        "1.0",
        method="tasks/list",
    )
    assert out["contextId"] == "ctx-1"
    assert out["pageSize"] == 10
    assert out["status"] == "completed"


@pytest.mark.parametrize(
    "proto_status, expected",
    [
        ("TASK_STATE_COMPLETED", "completed"),
        ("TASK_STATE_INPUT_REQUIRED", "input-required"),
        ("TASK_STATE_AUTH_REQUIRED", "auth-required"),
        ("TASK_STATE_CANCELED", "canceled"),
    ],
)
def test_list_tasks_status_filter_lowers_to_0_3_wire_value(proto_status, expected):
    out = normalize_request_params(
        {"status": proto_status},
        "1.0",
        method="tasks/list",
    )
    assert out["status"] == expected


def test_list_tasks_unspecified_status_is_dropped():
    out = normalize_request_params(
        {"contextId": "ctx-1", "status": "TASK_STATE_UNSPECIFIED"},
        "1.0",
        method="tasks/list",
    )
    assert "status" not in out
    assert out["contextId"] == "ctx-1"


@pytest.mark.parametrize(
    "method, result",
    [
        (
            "message/send",
            {
                "task": {
                    "id": "t1",
                    "contextId": "c1",
                    "status": {"state": "completed"},
                },
                "vendorExtraField": "x",
            },
        ),
        (
            "tasks/get",
            {
                "id": "t1",
                "contextId": "c1",
                "status": {"state": "completed"},
                "vendorExtraField": "x",
            },
        ),
    ],
)
def test_lowering_1_0_to_0_3_tolerates_unknown_upstream_fields(method, result):
    out = normalize_jsonrpc_response(_rpc(result), "0.3", method=method)
    lowered = out["result"]
    assert lowered["kind"] == "task"
    assert lowered["id"] == "t1"
    assert "vendorExtraField" not in lowered


def test_stream_event_lowering_1_0_to_0_3_tolerates_unknown_fields():
    event = {
        "task": {"id": "t1", "contextId": "c1", "status": {"state": "completed"}},
        "vendorExtraField": "x",
    }
    out = normalize_stream_event(_rpc(event), "0.3", request_id="1")
    lowered = out["result"]
    assert lowered["kind"] == "task"
    assert lowered["id"] == "t1"


def test_list_tasks_result_round_trip_preserves_task_ids():
    rpc = _rpc(
        {
            "tasks": [
                {
                    "kind": "task",
                    "id": "t1",
                    "contextId": "c1",
                    "status": {"state": "completed"},
                }
            ],
            "nextPageToken": "tok",
        }
    )
    v1 = normalize_jsonrpc_response(rpc, "1.0", method="tasks/list")
    assert "kind" not in v1["result"]["tasks"][0]
    assert v1["result"]["tasks"][0]["id"] == "t1"
    back = normalize_jsonrpc_response(v1, "0.3", method="tasks/list")
    assert back["result"]["tasks"][0]["kind"] == "task"
    assert back["result"]["tasks"][0]["id"] == "t1"


def _extended_card_1_0() -> dict:
    return {
        "name": "Card",
        "description": "d",
        "version": "1.0.0",
        "supportedInterfaces": [
            {
                "url": "https://upstream.example",
                "protocolBinding": "JSONRPC",
                "protocolVersion": "0.3",
            },
            {
                "url": "http://internal:9999",
                "protocolBinding": "JSONRPC",
                "protocolVersion": "0.3",
            },
        ],
    }


def test_agent_card_lowered_to_0_3_drops_additional_interfaces():
    # A 1.0 card with multiple interfaces would lower into a 0.3 card carrying the
    # secondary backend URLs in ``additionalInterfaces``; those must be stripped so
    # the conversion never re-exposes an upstream backend to A2A clients.
    out = normalize_agent_card(_extended_card_1_0(), "0.3")
    assert out["url"] == "https://upstream.example"
    assert "additionalInterfaces" not in out
    assert "supportedInterfaces" not in out
    assert "http://internal:9999" not in str(out)


def test_agent_card_with_0_3_pin_and_supported_interfaces_is_lowered():
    card = _extended_card_1_0()
    card["protocolVersion"] = "0.3"

    out = normalize_agent_card(card, "0.3")

    assert out["protocolVersion"] == "0.3"
    assert "supportedInterfaces" not in out


def test_agent_card_same_version_passthrough():
    card = _extended_card_1_0()
    assert normalize_agent_card(card, "1.0") is card


def test_detect_card_version_normalizes_semver_protocol_version():
    from litellm.proxy.a2a.version_convert import _detect_card_version

    assert _detect_card_version({"protocolVersion": "1.0.0"}) == "1.0"
    assert (
        _detect_card_version({"protocolVersion": "0.3.0", "supportedInterfaces": []})
        == "0.3"
    )


def test_agent_card_1_0_normalizes_flat_0_3_security_fields():
    from copy import deepcopy

    from a2a.compat.v0_3.conversions import ParseDict, pb2_v10

    from litellm.proxy.a2a.agent_card import (
        merge_agent_card,
        point_agent_card_at_proxy,
    )

    merged = merge_agent_card(
        {
            "protocolVersion": "1.0",
            "name": "Example Agent",
            "description": "Example agent.",
            "url": "https://upstream.example/a2a",
            "skills": [
                {
                    "id": "get_products",
                    "name": "Get products",
                    "tags": ["products"],
                    "description": "Find products in the catalog.",
                }
            ],
        },
        proxy_url="https://proxy.example.com/a2a/example-agent",
        proxy_base_url="https://proxy.example.com",
    )
    served_input = point_agent_card_at_proxy(merged, "https://proxy.example.com/a2a/example-agent")
    snapshot = deepcopy(served_input)

    out = normalize_agent_card(served_input, "1.0")

    assert served_input == snapshot
    assert out["securitySchemes"] == {
        "LiteLLMKey": {
            "httpAuthSecurityScheme": {
                "scheme": "bearer",
                "description": "LiteLLM virtual key",
            }
        }
    }
    assert out["securityRequirements"] == [{"schemes": {"LiteLLMKey": {"list": []}}}]
    assert "security" not in out

    ParseDict(
        out["securitySchemes"]["LiteLLMKey"],
        pb2_v10.SecurityScheme(),
        ignore_unknown_fields=False,
    )
    ParseDict(
        out["securityRequirements"][0],
        pb2_v10.SecurityRequirement(),
        ignore_unknown_fields=False,
    )
    pb_card = pb2_v10.AgentCard()
    ParseDict(out, pb_card, ignore_unknown_fields=True)
    assert pb_card.security_schemes["LiteLLMKey"].http_auth_security_scheme.scheme == "bearer"
    assert len(pb_card.security_requirements) == 1


def test_agent_card_security_normalization_all_scheme_types_round_trip():
    from a2a.compat.v0_3.conversions import ParseDict, pb2_v10

    card_1_0_with_legacy_sec = {
        "protocolVersion": "1.0",
        "name": "Multi Auth Agent",
        "description": "Tests all five A2A security scheme types.",
        "version": "1.0.0",
        "supportedInterfaces": [
            {
                "url": "https://proxy.example.com/a2a/multi",
                "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0",
            }
        ],
        "securitySchemes": {
            "http_s": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "Bearer token",
            },
            "api_s": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "Header key",
            },
            "oauth_s": {
                "type": "oauth2",
                "flows": {
                    "clientCredentials": {
                        "tokenUrl": "https://auth.example/token",
                        "scopes": {"read": "Read access"},
                    }
                },
            },
            "oidc_s": {
                "type": "openIdConnect",
                "openIdConnectUrl": "https://idp.example/.well-known/openid-configuration",
            },
            "mtls_s": {
                "type": "mutualTLS",
                "description": "Client cert",
            },
        },
        "security": [{"http_s": [], "oauth_s": ["read"]}],
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [
            {
                "id": "s1",
                "name": "Skill One",
                "description": "Desc",
                "tags": ["tag"],
                "security": [{"api_s": []}],
            }
        ],
    }

    out_1_0 = normalize_agent_card(card_1_0_with_legacy_sec, "1.0")
    assert "security" not in out_1_0
    assert out_1_0["securityRequirements"] == [{"schemes": {"http_s": {"list": []}, "oauth_s": {"list": ["read"]}}}]
    assert out_1_0["securitySchemes"]["http_s"] == {
        "httpAuthSecurityScheme": {
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Bearer token",
        }
    }
    assert out_1_0["securitySchemes"]["api_s"] == {
        "apiKeySecurityScheme": {
            "location": "header",
            "name": "X-API-Key",
            "description": "Header key",
        }
    }
    assert out_1_0["securitySchemes"]["oauth_s"] == {
        "oauth2SecurityScheme": {
            "flows": {
                "clientCredentials": {
                    "tokenUrl": "https://auth.example/token",
                    "scopes": {"read": "Read access"},
                }
            }
        }
    }
    assert out_1_0["securitySchemes"]["oidc_s"] == {
        "openIdConnectSecurityScheme": {"openIdConnectUrl": "https://idp.example/.well-known/openid-configuration"}
    }
    assert out_1_0["securitySchemes"]["mtls_s"] == {"mtlsSecurityScheme": {"description": "Client cert"}}
    assert "security" not in out_1_0["skills"][0]
    assert out_1_0["skills"][0]["securityRequirements"] == [{"schemes": {"api_s": {"list": []}}}]

    for scheme in out_1_0["securitySchemes"].values():
        ParseDict(scheme, pb2_v10.SecurityScheme(), ignore_unknown_fields=False)

    card_with_0_3_interface = {
        **out_1_0,
        "supportedInterfaces": [
            {
                "url": "https://proxy.example.com/a2a/multi",
                "protocolBinding": "JSONRPC",
                "protocolVersion": "0.3",
            }
        ],
    }
    lowered_0_3 = normalize_agent_card(card_with_0_3_interface, "0.3")
    assert "supportedInterfaces" not in lowered_0_3
    assert "securityRequirements" not in lowered_0_3
    assert lowered_0_3["security"] == [{"http_s": [], "oauth_s": ["read"]}]
    assert lowered_0_3["securitySchemes"]["api_s"] == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "Header key",
    }
    assert lowered_0_3["securitySchemes"]["mtls_s"] == {
        "type": "mutualTLS",
        "description": "Client cert",
    }
    assert lowered_0_3["skills"][0]["security"] == [{"api_s": []}]
    assert "securityRequirements" not in lowered_0_3["skills"][0]

    same_version_0_3_card = {k: v for k, v in out_1_0.items() if k != "supportedInterfaces"}
    same_version_0_3_card["protocolVersion"] = "0.3"
    same_version_0_3_card["url"] = "https://proxy.example.com/a2a/multi"
    same_version_0_3 = normalize_agent_card(same_version_0_3_card, "0.3")
    assert "securityRequirements" not in same_version_0_3
    assert same_version_0_3["security"] == [{"http_s": [], "oauth_s": ["read"]}]
    assert same_version_0_3["securitySchemes"]["http_s"]["type"] == "http"
    assert same_version_0_3["securitySchemes"]["api_s"]["in"] == "header"
    assert "location" not in same_version_0_3["securitySchemes"]["api_s"]
    assert same_version_0_3["skills"][0]["security"] == [{"api_s": []}]
    assert "securityRequirements" not in same_version_0_3["skills"][0]


def test_agent_card_0_3_with_supported_interfaces_and_flat_security_schemes_lowers_cleanly():
    from litellm.proxy.a2a.agent_card import (
        merge_agent_card,
        point_agent_card_at_proxy,
    )

    merged = merge_agent_card(
        {
            "protocolVersion": "0.3",
            "name": "Legacy Agent",
            "description": "Pinned to 0.3",
            "url": "https://upstream.example/a2a",
        },
        proxy_url="https://proxy.example.com/a2a/legacy-agent",
        proxy_base_url="https://proxy.example.com",
    )
    served_input = point_agent_card_at_proxy(merged, "https://proxy.example.com/a2a/legacy-agent")

    out = normalize_agent_card(served_input, "0.3")

    assert "supportedInterfaces" not in out
    assert "securityRequirements" not in out
    assert out["protocolVersion"] == "0.3"
    assert out["url"] == "https://proxy.example.com/a2a/legacy-agent"
    assert out["securitySchemes"] == {
        "LiteLLMKey": {
            "type": "http",
            "scheme": "bearer",
            "description": "LiteLLM virtual key",
        }
    }
    assert out["security"] == [{"LiteLLMKey": []}]
