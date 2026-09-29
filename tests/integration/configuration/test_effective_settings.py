import uuid
from typing import Final

import httpx
import pytest

from tests.integration._support.client import Gateway, object_value, string_value
from tests.integration._support.database import read_rows


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
