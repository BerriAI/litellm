import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0"
TOKEN: Final = "synthetic-bedrock-bearer"
ACCESS_KEY: Final = "AKIAINTEGRATION000002"
CLIENT_OAUTH_TOKEN: Final = "Bearer sk-ant-oat01-synthetic-client-subscription-token"
RESPONSE: Final = json.dumps(
    {
        "output": {"message": {"role": "assistant", "content": [{"text": "bedrock wire control"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15},
        "metrics": {"latencyMs": 1},
    }
).encode()


def bearer_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert "x-amz-security-token" not in request.headers
    body: Final = json.loads(request.body)
    assert body["messages"] == [{"role": "user", "content": [{"text": "synthetic bearer request"}]}]
    assert body["system"] == [{"text": "synthetic system"}]
    assert body["inferenceConfig"]["maxTokens"] == 16
    assert not {"timeout", "stream_chunk_size", "litellm_params", "litellm_metadata", "api_key"}.intersection(body)
    return Reply(body=RESPONSE)


@pytest.mark.covers("other.provider_wire.bedrock.bearer_sdk_skips_credential_chain")
async def test_bearer_only_sdk_sync_async_requests_do_not_require_aws_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import litellm

    empty: Final = tmp_path / "empty-aws-config"
    empty.write_text("")
    for name in tuple(name for name in os.environ if name.startswith("AWS_")):
        monkeypatch.delenv(name, raising=False)
    for name, value in {
        "AWS_CONFIG_FILE": str(empty),
        "AWS_SHARED_CREDENTIALS_FILE": str(empty),
        "AWS_EC2_METADATA_DISABLED": "true",
        "LITELLM_RUST": "false",
    }.items():
        monkeypatch.setenv(name, value)
    with wire_server(bearer_peer) as wire:
        with pytest.raises(litellm.APIConnectionError, match=r"config profile .* could not be found"):
            await asyncio.to_thread(
                litellm.completion,
                model=MODEL,
                aws_profile_name="integration-profile-must-not-be-read",
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint=wire.url,
                messages=[{"role": "user", "content": "synthetic credential control"}],
                timeout=5,
                num_retries=0,
            )
        assert wire.drain() == ()
        for source in ("argument", "environment"):
            if source == "environment":
                monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", TOKEN)
            parameters: Final = {
                "model": MODEL,
                "api_key": TOKEN if source == "argument" else None,
                "aws_region_name": "us-east-1",
                "aws_profile_name": "integration-profile-must-not-be-read",
                "aws_bedrock_runtime_endpoint": wire.url,
                "timeout": 5,
                "num_retries": 0,
                "messages": [
                    {"role": "system", "content": "synthetic system"},
                    {"role": "user", "content": "synthetic bearer request"},
                ],
                "max_tokens": 16,
            }
            for asynchronous in (False, True):
                result: Final = (
                    await litellm.acompletion(**parameters)
                    if asynchronous
                    else await asyncio.to_thread(litellm.completion, **parameters)
                )
                assert result.choices[0].message.content == "bedrock wire control"
                assert result.choices[0].finish_reason == "stop"
                assert result.usage.prompt_tokens == 11 and result.usage.completion_tokens == 4
                assert len(wire.drain()) == 1


@pytest.mark.covers("other.provider_wire.bedrock.bearer_db_yaml_survives_reload")
def test_bearer_environment_reference_loads_from_db_and_yaml_and_survives_reload(
    gateway: Gateway, tmp_path: Path
) -> None:
    empty: Final = tmp_path / "empty-aws-config"
    empty.write_text("")
    with wire_server(bearer_peer) as wire:
        parameters: Final = {
            "model": MODEL,
            "api_key": "os.environ/INTEGRATION_BEARER_TOKEN",
            "aws_region_name": "us-east-1",
            "aws_profile_name": "integration-profile-must-not-be-read",
            "aws_bedrock_runtime_endpoint": wire.url,
        }
        alias: Final = f"integration-yaml-{uuid.uuid4().hex}"
        configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        configuration["model_list"] = [{"model_name": alias, "litellm_params": parameters, "model_info": {"id": alias}}]
        path: Final = tmp_path / "bedrock.yaml"
        path.write_text(yaml.safe_dump(configuration))
        overrides: Final = {
            "INTEGRATION_BEARER_TOKEN": TOKEN,
            "AWS_CONFIG_FILE": str(empty),
            "AWS_SHARED_CREDENTIALS_FILE": str(empty),
            "AWS_EC2_METADATA_DISABLED": "true",
            "LITELLM_RUST": "false",
        }
        with (
            owned_proxy(
                gateway,
                tmp_path,
                overrides,
                config=path,
                remove_environment=tuple(name for name in os.environ if name.startswith("AWS_")),
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            database_model: Final = scenario.model(**parameters)
            for generation in range(2):
                for model in (alias, database_model):
                    response: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": "synthetic system"},
                                {"role": "user", "content": "synthetic bearer request"},
                            ],
                            "max_tokens": 16,
                            "cache": {"no-cache": True},
                        },
                    )
                    assert response.status_code == 200, response.text
                    assert response.json()["choices"][0]["message"]["content"] == "bedrock wire control"
                    assert response.json()["usage"]["total_tokens"] == 15
                    assert len(wire.drain()) == 1, f"Expected actual provider call after reload {generation}"
                if generation == 0:
                    entries: Final = candidate.get("/model/info")["data"]
                    target: Final = next(entry for entry in entries if entry["model_name"] == database_model)
                    response: Final = candidate.request(
                        "PATCH",
                        f"/model/{target['model_info']['id']}/update",
                        {"model_info": {"description": "bearer reload"}},
                    )
                    assert response.status_code == 200, response.text


INVOKE_MODEL: Final = "bedrock/invoke/anthropic.claude-3-haiku-20240307-v1:0"
INVOKE_RESPONSE: Final = json.dumps(
    {
        "id": "msg_synthetic",
        "type": "message",
        "role": "assistant",
        "model": "anthropic.claude-3-haiku-20240307-v1:0",
        "content": [{"type": "text", "text": "bedrock invoke wire control"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 4},
    }
).encode()


def sigv4_invoke_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/model/anthropic.claude-3-haiku-20240307-v1:0/invoke"
    assert request.headers["authorization"].startswith(f"AWS4-HMAC-SHA256 Credential={ACCESS_KEY}/"), dict(
        request.headers
    )
    assert CLIENT_OAUTH_TOKEN not in request.headers.values(), dict(request.headers)
    assert json.loads(request.body)["messages"] == [{"role": "user", "content": "synthetic oauth isolation request"}]
    return Reply(body=INVOKE_RESPONSE)


@pytest.mark.covers("providers.bedrock_auth.client_anthropic_oauth_token_never_replaces_sigv4_authorization")
def test_client_anthropic_oauth_authorization_header_does_not_replace_bedrock_sigv4_signature(gateway: Gateway) -> None:
    with wire_server(sigv4_invoke_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=INVOKE_MODEL,
            api_key=None,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key="synthetic-secret-key-for-testing",
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            api_base=wire.url,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "messages": [{"role": "user", "content": "synthetic oauth isolation request"}],
                "max_tokens": 16,
            },
            headers={"Authorization": CLIENT_OAUTH_TOKEN, "x-litellm-api-key": f"Bearer {gateway.key}"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["content"] == [{"type": "text", "text": "bedrock invoke wire control"}], response.text
        assert len(wire.drain()) == 1, response.text
