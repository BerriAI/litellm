import json
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, object_value
from integration._support.process import OwnedState, owned_proxy
from integration._support.wire import Reply, Request, wire_server


def _tokenizer(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", "/tokenize")
    assert request.headers["authorization"] == "Bearer deployment-token"
    body: Final = json.loads(request.body)
    assert body["model"] == "acme/model"
    assert body["prompt"] == "Hello world"
    return Reply(body=json.dumps({"count": 2, "tokens": [101, 102], "max_model_len": 4096}).encode())


@pytest.mark.parametrize("path", ["/tokenize", "/v1/tokenize"])
def test_tokenization_routes_model_alias_with_inference_auth(tmp_path: Path, path: str) -> None:
    with (
        wire_server(_tokenizer) as wire,
        httpx.Client(base_url="http://127.0.0.1:1", trust_env=False) as base_client,
    ):
        base_gateway: Final = Gateway(base_client, "sk-integration-tokenize", wire.url)
        with (
            owned_proxy(base_gateway, tmp_path, {"STORE_MODEL_IN_DB": "False"}, state=OwnedState.PRIVATE) as gateway,
            gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model="hosted_vllm/acme/model",
                api_base=wire.url,
                api_key="deployment-token",
            )
            denied_key: Final = scenario.key(models=["another-model"])
            denied: Final = gateway.request("POST", path, {"model": model, "prompt": "Hello world"}, key=denied_key)
            assert denied.status_code in (401, 403), denied.text
            assert wire.drain() == ()
            allowed_key: Final = scenario.key(models=[model])
            invalid: Final = gateway.request(
                "POST",
                path,
                {"model": model, "prompt": "Hello world", "unsupported_option": True},
                key=allowed_key,
            )
            assert invalid.status_code == 400, invalid.text
            assert wire.drain() == ()
            response: Final = gateway.request(
                "POST",
                path,
                {"model": model, "prompt": "Hello world", "include": []},
                key=allowed_key,
            )
            assert response.status_code == 200, response.text
            payload: Final = response.json()
            assert payload["model"] == "acme/model"
            assert payload["count"] == 2
            assert payload["token_ids"] is None
            assert payload["tokens"] is None
            assert float(response.headers["x-litellm-response-cost"]) == 0
            assert response.headers["x-litellm-rust"] == "true"
            assert len(wire.drain()) == 1


@pytest.mark.parametrize("path", ["/tokenize", "/v1/tokenize"])
@pytest.mark.parametrize(
    ("include", "has_token_ids"),
    [([], False), (["token_ids"], True)],
    ids=["count-only", "token-ids"],
)
def test_tokenization_http_hooks_rewrite_and_replace_once(
    tmp_path: Path,
    path: str,
    include: list[str],
    has_token_ids: bool,
) -> None:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["callbacks"] = ["integration._support.tokenize_hook_probe.probe"]
    config_path: Final = tmp_path / "tokenize-hooks.yaml"
    config_path.write_text(yaml.safe_dump(config))

    def tokenize_upstream(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/tokenize")
        assert request.headers["authorization"] == "Bearer deployment-token"
        body: Final = json.loads(request.body)
        assert body["model"] == "acme/model"
        assert body["prompt"] == "rewritten by tokenization hook"
        return Reply(body=json.dumps({"count": 3, "tokens": [101, 102, 103], "max_model_len": 4096}).encode())

    marker: Final = f"tokenize-hook-{'-'.join(include) or 'count-only'}"
    with (
        wire_server(tokenize_upstream) as upstream,
        wire_server(lambda _: Reply()) as hook_sink,
        httpx.Client(base_url="http://127.0.0.1:1", trust_env=False) as base_client,
    ):
        base_gateway: Final = Gateway(base_client, "sk-integration-tokenize", upstream.url)
        with (
            owned_proxy(
                base_gateway,
                tmp_path,
                {"TOKENIZE_HOOK_PROBE_URL": hook_sink.url, "STORE_MODEL_IN_DB": "False"},
                config=config_path,
                state=OwnedState.PRIVATE,
            ) as proxy,
            proxy.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model="hosted_vllm/acme/model",
                api_base=upstream.url,
                api_key="deployment-token",
            )
            key: Final = scenario.key(models=[model], user_id="tokenize-hook-user")
            response: Final = proxy.request(
                "POST",
                path,
                {
                    "model": model,
                    "prompt": "original prompt",
                    "include": include,
                    "user": "tokenize-hook-user",
                    "metadata": {"request_marker": marker},
                },
                key=key,
                headers={"x-request-id": marker},
            )
            assert response.status_code == 200, response.text
            assert int(response.headers["content-length"]) == len(response.content)
            payload: Final = object_value(response.json())
            assert payload["model"] == "hook-replaced-model"
            assert payload["count"] == 777
            assert float(response.headers["x-litellm-response-cost"]) == 0
            assert response.headers["x-litellm-rust"] == "true"
            events: Final = tuple(json.loads(request.body) for request in hook_sink.drain())
            assert tuple(event["stage"] for event in events) == ("pre", "post")
            assert events[0]["call_id"]
            assert events[0]["call_id"] == events[1]["call_id"]
            assert events[0]["call_type"] == "atokenize"
            assert events[0]["prompt"] == "rewritten by tokenization hook"
            assert events[1]["call_id"] == response.headers["x-litellm-call-id"]
            assert events[1]["user_id"] == "tokenize-hook-user"
            assert events[1]["user"] == "tokenize-hook-user"
            assert marker in json.dumps(events[1]["metadata"])
            assert payload["token_ids"] == events[1]["token_ids"]
            assert payload["tokens"] == events[1]["tokens"]
            assert events[1]["count"] == 3
            assert events[1]["model"] == "acme/model"
            assert events[1]["hidden_params"]["x-litellm-rust"] is True
            assert events[1]["hidden_params"]["response_cost"] == 0
            if not has_token_ids:
                assert events[1]["token_ids"] is None
                assert events[1]["tokens"] is None
            else:
                assert events[1]["token_ids"] is not None
                assert events[1]["tokens"] is None
            recorded: Final = upstream.drain()
            assert len(recorded) == 1
            assert int(recorded[0].headers["content-length"]) == len(recorded[0].body)


def test_tokenization_proxy_authorizes_watsonx_overrides_before_provider_io(tmp_path: Path) -> None:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())

    def watsonx_upstream(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target.startswith("/ml/v1/text/tokenization?")
        body: Final = json.loads(request.body)
        assert body["input"] == "Hello world"
        expected_project: Final = "caller-project" if "version=2023-05-29" in request.target else "deployment-project"
        assert body["project_id"] == expected_project
        return Reply(body=json.dumps({"result": {"token_count": 2, "tokens": ["Hello", " world"]}}).encode())

    with (
        wire_server(watsonx_upstream) as upstream,
        httpx.Client(base_url="http://127.0.0.1:1", trust_env=False) as base_client,
    ):
        config["model_list"] = [
            {
                "model_name": model,
                "litellm_params": {
                    "model": "watsonx/ibm/granite-3-8b-instruct",
                    "api_base": upstream.url,
                    "token": "deployment-token",
                    "project_id": "deployment-project",
                    "api_version": "2023-05-29",
                    **(
                        {"configurable_clientside_auth_params": ["project_id", "api_version"]}
                        if model == "watsonx-tokenize-allowed"
                        else {}
                    ),
                },
            }
            for model in ("watsonx-tokenize-denied", "watsonx-tokenize-allowed")
        ]
        config_path: Final = tmp_path / "tokenize-auth.yaml"
        config_path.write_text(yaml.safe_dump(config))
        base_gateway: Final = Gateway(base_client, "sk-integration-tokenize", upstream.url)
        with owned_proxy(
            base_gateway,
            tmp_path,
            {"STORE_MODEL_IN_DB": "False"},
            config=config_path,
            state=OwnedState.PRIVATE,
        ) as proxy:

            def verify_override(field: str, value: str, expected_project: str, expected_version: str) -> None:
                denied: Final = proxy.request(
                    "POST",
                    "/tokenize",
                    {"model": "watsonx-tokenize-denied", "prompt": "Hello world", field: value},
                )
                assert denied.status_code >= 400, denied.text
                assert upstream.drain() == ()
                allowed: Final = proxy.request(
                    "POST",
                    "/tokenize",
                    {"model": "watsonx-tokenize-allowed", "prompt": "Hello world", field: value},
                )
                assert allowed.status_code == 200, allowed.text
                recorded: Final = upstream.drain()
                assert len(recorded) == 1
                request_body: Final = json.loads(recorded[0].body)
                assert request_body["project_id"] == expected_project
                assert f"version={expected_version}" in recorded[0].target

            verify_override("project_id", "caller-project", "caller-project", "2023-05-29")
            verify_override("api_version", "2025-01-01", "deployment-project", "2025-01-01")
            forged: Final = proxy.request(
                "POST",
                "/tokenize",
                {
                    "model": "watsonx-tokenize-allowed",
                    "prompt": "Hello world",
                    "metadata": {"nested": [{"backend": {"type": "local"}}]},
                },
            )
            assert forged.status_code == 400, forged.text
            assert upstream.drain() == ()
