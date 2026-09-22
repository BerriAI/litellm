import json
import logging
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.internal_key_emission_guard import internal_key_leak_counter
from tests.integration._support.client import JSON_OBJECT, Gateway, object_value
from tests.integration._support.wire import Reply, Request, wire_server

CONVERSE_MODEL: Final = "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0"
CONVERSE_REPLY: Final = json.dumps(
    {
        "output": {"message": {"role": "assistant", "content": [{"text": "converse guard control"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15},
        "metrics": {"latencyMs": 1},
    }
).encode()


def converse_peer(request: Request) -> Reply:
    return Reply(body=CONVERSE_REPLY)


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


@pytest.mark.covers("other.provider_wire.internal_key_emission_observed")
def test_internal_key_forced_into_body_is_observed_at_emission(
    gateway: Gateway, caplog: pytest.LogCaptureFixture
) -> None:
    with httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
        upstream.get("/__observations").raise_for_status()
        before: Final = internal_key_leak_counter.value
        with caplog.at_level(logging.WARNING, logger="LiteLLM"), pytest.raises(litellm.BadRequestError):
            litellm.completion(
                model="deepseek/synthetic-model",
                api_base=f"{gateway.upstream_url}/v1",
                api_key="sk-synthetic",
                messages=[{"role": "user", "content": "emission guard"}],
                extra_body={"litellm_call_id": "forced-through-extra-body"},
            )
        observations: Final = JSON_OBJECT.validate_json(upstream.get("/__observations").content)["requests"]
        assert isinstance(observations, list) and len(observations) == 1
        assert object_value(object_value(observations[0])["body"])["litellm_call_id"] == "forced-through-extra-body"
        assert internal_key_leak_counter.value == before + 1
        assert [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING] == [
            "LiteLLM internal keys reached the deepseek provider request body: litellm_call_id"
        ]


@pytest.mark.covers("other.provider_wire.internal_key_emission_observed_bedrock_converse")
def test_internal_key_forced_into_converse_body_is_observed_at_emission(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "false")
    before: Final = internal_key_leak_counter.value
    with wire_server(converse_peer) as wire, caplog.at_level(logging.WARNING, logger="LiteLLM"):
        result: Final = litellm.completion(
            model=CONVERSE_MODEL,
            api_key="synthetic-bedrock-bearer",
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            messages=[{"role": "user", "content": "converse emission guard"}],
            extra_body={"litellm_call_id": "forced-through-extra-body"},
            timeout=5,
            num_retries=0,
        )
        received: Final = wire.drain()
    assert isinstance(result, litellm.ModelResponse)
    choice: Final = result.choices[0]
    assert isinstance(choice, litellm.Choices)
    assert choice.message.content == "converse guard control"
    assert len(received) == 1
    body: Final = JSON_OBJECT.validate_json(received[0].body)
    assert object_value(object_value(body["additionalModelRequestFields"])["extra_body"]) == {
        "litellm_call_id": "forced-through-extra-body"
    }
    assert internal_key_leak_counter.value == before + 1
    assert [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING] == [
        "LiteLLM internal keys reached the bedrock provider request body: "
        "additionalModelRequestFields.extra_body.litellm_call_id"
    ]


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
