from typing import Final

import httpx
import pytest

from tests.integration._support.client import Gateway, JSON_OBJECT, object_value


@pytest.mark.covers("other.provider_wire.internal_parameters_filtered")
def test_internal_request_state_does_not_reach_provider(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
        upstream.get("/__observations").raise_for_status()
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model], tpm_limit=10000, rpm_limit=100)
        result: Final = gateway.post(
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "wire contract"}],
                "temperature": 0.4,
                "max_tokens": 20,
                "timeout": 12,
            },
            key=key,
        )
        assert object_value(result["usage"])["total_tokens"] == 40
        observations: Final = JSON_OBJECT.validate_json(upstream.get("/__observations").content)["requests"]
        assert isinstance(observations, list)
        assert len(observations) == 1
        observed: Final = object_value(observations[0])
        body: Final = object_value(observed["body"])
        assert body["model"] == "gpt-4o-mini"
        assert body["messages"] == [{"role": "user", "content": "wire contract"}]
        assert body["temperature"] == 0.4
        assert body["max_tokens"] == 20
        assert observed["authorization"] == "Bearer integration-provider-key"
        assert "litellm_metadata" not in body
        assert "litellm_params" not in body
        assert "timeout" not in body
        assert "tpm" not in body


@pytest.mark.covers("other.provider_wire.validator_rejects_corruption")
def test_upstream_rejects_corruption_and_accepts_supported_metadata(gateway: Gateway) -> None:
    with httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
        missing: Final = upstream.post("/v1/chat/completions", json={"model": "gpt-4o-mini"})
        assert missing.status_code == 400
        assert (
            object_value(object_value(missing.json())["error"])["message"] == "model and nonempty messages are required"
        )
        body: Final = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "strict control"}]}
        leaked: Final = upstream.post("/v1/chat/completions", json={**body, "litellm_metadata": {"hidden": "value"}})
        assert leaked.status_code == 400
        assert "litellm_metadata" in str(object_value(object_value(leaked.json())["error"])["message"])
        valid: Final = upstream.post("/v1/chat/completions", json={**body, "metadata": {"purpose": "synthetic"}})
        assert valid.status_code == 200, valid.text
        assert object_value(object_value(valid.json())["usage"])["total_tokens"] == 40
