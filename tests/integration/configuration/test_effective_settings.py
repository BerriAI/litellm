import json
import uuid
from typing import Final

import httpx
import pytest
from pydantic import BaseModel, JsonValue

from tests.integration._support.client import Gateway, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.wire import Reply, Request, Wire, wire_server


def model_identity(gateway: Gateway, alias: str) -> str:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    entry: Final = next(object_value(value) for value in entries if object_value(value)["model_name"] == alias)
    return string_value(object_value(entry["model_info"])["id"])


@pytest.mark.covers("mgmt.model.block.changes_serving_and_preserves_control")
def test_model_block_changes_actual_route_and_leaves_other_route_working(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        other: Final = scenario.model()
        identity: Final = model_identity(gateway, model)
        gateway.chat(model)
        gateway.chat(other)
        gateway.post("/model/block", {"model_id": identity})
        assert read_rows(
            'SELECT blocked FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (identity,)
        ) == [{"blocked": True}]
        response: Final = gateway.request(
            "POST", "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "blocked deployment"}]},
        )
        assert response.status_code == 403, response.text
        assert response.json()["error"]["type"] == "permission_error"
        assert response.json()["error"]["message"] == "litellm.PermissionDeniedError: Model is blocked"
        assert object_value(gateway.chat(other)["usage"])["total_tokens"] == 40
        gateway.post("/model/unblock", {"model_id": identity})
        assert read_rows(
            'SELECT blocked FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (identity,)
        ) == [{"blocked": False}]
        assert object_value(gateway.chat(model)["usage"])["total_tokens"] == 40


@pytest.mark.covers("mgmt.router_settings.update.changes_observed_attempt_count")
def test_saved_retry_setting_controls_real_attempts_and_restores(gateway: Gateway) -> None:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream, gateway.scenario() as scenario:
        original: Final = object_value(gateway.get("/router/settings")["current_values"])["num_retries"]
        provider_model: Final = f"retry-{uuid.uuid4().hex}"
        model: Final = scenario.model(model=f"openai/{provider_model}", input_cost_per_token=0, output_cost_per_token=0)

        def remove_script() -> None:
            response: Final = upstream.delete(f"/__scripts/{provider_model}")
            assert response.status_code in (200, 404), response.text
            assert upstream.get(f"/__scripts/{provider_model}").status_code == 404

        scenario.cleanups.callback(remove_script)
        try:
            for generation, retries in enumerate((0, 1, original)):
                gateway.post("/config/update", {"router_settings": {"num_retries": retries}})
                assert object_value(gateway.get("/router/settings")["current_values"])["num_retries"] == retries
                configured: Final = upstream.post(f"/__scripts/{provider_model}", json={"statuses": [500, 200]})
                assert configured.status_code == 200, configured.text
                upstream.get("/__observations").raise_for_status()
                response: Final = gateway.request(
                    "POST", "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": f"{provider_model} attempt {generation}"}]},
                )
                observed: Final = upstream.get("/__observations")
                observed.raise_for_status()
                requests: Final = observed.json()["requests"]
                assert len(requests) == (1 if retries == 0 else 2), (response.status_code, response.text, requests)
                assert all(value["body"]["model"] == provider_model for value in requests)
                assert response.status_code == (500 if retries == 0 else 200), response.text
                if retries != 0:
                    assert response.json()["usage"]["total_tokens"] == 40
                remaining: Final = upstream.delete(f"/__scripts/{provider_model}")
                assert remaining.status_code == 200, remaining.text
                assert remaining.json()["remaining"] == ([200] if retries == 0 else [])
        finally:
            gateway.post("/config/update", {"router_settings": {"num_retries": original}})
            assert object_value(gateway.get("/router/settings")["current_values"])["num_retries"] == original


class _MaskedCredential(BaseModel):
    credential_name: str
    credential_info: dict[str, JsonValue]
    credential_values: dict[str, JsonValue]


def _openai_reply(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-credential-merge",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
    )


def _stored_credential_values(name: str) -> dict[str, object]:
    rows: Final = read_rows(
        'SELECT credential_values FROM "LiteLLM_CredentialsTable" WHERE credential_name = %s', (name,)
    )
    assert len(rows) == 1, rows
    return object_value(rows[0]["credential_values"])


def _masked_credential(gateway: Gateway, name: str) -> _MaskedCredential:
    response: Final = gateway.request("GET", f"/credentials/by_name/{name}")
    assert response.status_code == 200, response.text
    return _MaskedCredential.model_validate_json(response.content)


def _patch_credential(gateway: Gateway, name: str, body: dict[str, JsonValue]) -> None:
    response: Final = gateway.request("PATCH", f"/credentials/{name}", {"credential_name": name, **body})
    assert response.status_code == 200, response.text
    assert response.json() == {"success": True, "message": "Credential updated successfully"}, response.text


def _provider_call(gateway: Gateway, model: str, text: str, *wires: Wire) -> tuple[tuple[Request, ...], ...]:
    for wire in wires:
        wire.drain()
    response: Final = gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": text}]}
    )
    assert response.status_code == 200, response.text
    assert object_value(response.json()["usage"])["total_tokens"] == 2, response.text
    return tuple(wire.drain() for wire in wires)


def _sent_headers(request: Request) -> tuple[str, str | None]:
    return request.headers["authorization"], request.headers.get("openai-organization")


def test_partial_credential_patch_merges_values_and_deletes_only_the_named_field(gateway: Gateway) -> None:
    api_key: Final = f"synthetic-merge-{uuid.uuid4().hex}"
    with (
        wire_server(_openai_reply) as original,
        wire_server(_openai_reply) as moved,
        gateway.scenario() as scenario,
    ):
        name: Final = f"credential-{uuid.uuid4().hex}"
        gateway.post("/credentials", {
            "credential_name": name,
            "credential_values": {"api_key": api_key, "api_base": f"{original.url}/v1", "organization": "org-merge"},
            "credential_info": {},
        })
        scenario.cleanups.callback(gateway.request, "DELETE", f"/credentials/{name}")
        model: Final = scenario.model(api_key=None, api_base=None, litellm_credential_name=name)
        first, untouched = _provider_call(gateway, model, "before patch", original, moved)
        assert [_sent_headers(request) for request in first] == [(f"Bearer {api_key}", "org-merge")]
        assert untouched == ()
        stored: Final = _stored_credential_values(name)
        assert sorted(stored) == ["api_base", "api_key", "organization"], stored

        _patch_credential(gateway, name, {"credential_values": {"api_base": f"{moved.url}/v1"}, "credential_info": {}})
        merged: Final = _stored_credential_values(name)
        assert sorted(merged) == ["api_base", "api_key", "organization"], merged
        assert (merged["api_key"], merged["organization"]) == (stored["api_key"], stored["organization"]), merged
        assert merged["api_base"] != stored["api_base"], merged
        assert _masked_credential(gateway, name) == _MaskedCredential(
            credential_name=name,
            credential_info={},
            credential_values={"api_key": "sy****" + api_key[-2:], "api_base": f"{moved.url}/v1", "organization": "org-merge"},
        )
        old_base, new_base = _provider_call(gateway, model, "after api_base patch", original, moved)
        assert old_base == ()
        assert [(request.method, request.target) for request in new_base] == [("POST", "/v1/chat/completions")]
        assert [_sent_headers(request) for request in new_base] == [(f"Bearer {api_key}", "org-merge")]

        _patch_credential(gateway, name, {"credential_values_to_delete": ["organization"], "credential_info": {}})
        pruned: Final = _stored_credential_values(name)
        assert pruned == {"api_key": merged["api_key"], "api_base": merged["api_base"]}, pruned
        assert _masked_credential(gateway, name) == _MaskedCredential(
            credential_name=name,
            credential_info={},
            credential_values={"api_key": "sy****" + api_key[-2:], "api_base": f"{moved.url}/v1"},
        )
        old_base_after_delete, after_delete = _provider_call(gateway, model, "after field delete", original, moved)
        assert old_base_after_delete == ()
        assert [_sent_headers(request) for request in after_delete] == [(f"Bearer {api_key}", None)]


@pytest.mark.covers("mgmt.credential.update.saved_value_reaches_wire")
def test_credential_value_update_and_model_reload_reach_provider(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        name: Final = f"credential-{uuid.uuid4().hex}"
        gateway.post("/credentials", {
            "credential_name": name, "credential_values": {"api_key": "synthetic-credential-first"}, "credential_info": {}
        })

        def remove_credential() -> None:
            response: Final = gateway.request("DELETE", f"/credentials/{name}")
            assert response.status_code == 200, response.text
            assert read_rows(
                'SELECT credential_name FROM "LiteLLM_CredentialsTable" WHERE credential_name = %s', (name,)
            ) == []

        scenario.cleanups.callback(remove_credential)
        model: Final = scenario.model(api_key=None, litellm_credential_name=name)
        identity: Final = model_identity(gateway, model)
        for value in ("synthetic-credential-first", "synthetic-credential-second"):
            patched: Final = gateway.request("PATCH", f"/credentials/{name}", {
                "credential_name": name, "credential_values": {"api_key": value}, "credential_info": {}
            })
            assert patched.status_code == 200, patched.text
            rows: Final = read_rows(
                'SELECT credential_values FROM "LiteLLM_CredentialsTable" WHERE credential_name = %s', (name,)
            )
            assert len(rows) == 1
            stored: Final = object_value(rows[0]["credential_values"])
            assert isinstance(stored["api_key"], str) and stored["api_key"] != value
            for reload in (False, True):
                if reload:
                    response: Final = gateway.request("PATCH", f"/model/{identity}/update", {"model_info": {"description": value}})
                    assert response.status_code == 200, response.text
                upstream.get("/__observations").raise_for_status()
                assert object_value(gateway.chat(model, text=f"{name} {value} reload={reload}")["usage"])["total_tokens"] == 40
                observed: Final = upstream.get("/__observations")
                observed.raise_for_status()
                assert len(observed.json()["requests"]) == 1, (value, reload, observed.text)
                assert observed.json()["requests"][0]["authorization"] == f"Bearer {value}"
