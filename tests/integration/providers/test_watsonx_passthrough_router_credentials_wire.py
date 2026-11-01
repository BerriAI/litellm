import json
import uuid
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway, JsonValue, object_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_DEPLOYMENT_KEY: Final = "synthetic-watsonx-deployment-key"
_IAM_TOKEN: Final = "synthetic-watsonx-iam-token"
_WATSONX_ENV: Final = (
    "WATSONX_API_KEY",
    "WATSONX_APIKEY",
    "WX_API_KEY",
    "WATSONX_API_BASE",
    "WATSONX_URL",
    "WX_URL",
    "WML_URL",
    "WATSONX_TOKEN",
    "WATSONX_ZENAPIKEY",
)


def _config(directory: Path, api_base: str) -> Path:
    stock: Final = object_value(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final[dict[str, JsonValue]] = {
        **stock,
        "model_list": [
            {
                "model_name": "granite",
                "litellm_params": {
                    "model": "watsonx/ibm/granite-3-8b-instruct",
                    "api_key": _DEPLOYMENT_KEY,
                    "api_base": api_base,
                    "project_id": "synthetic-project",
                    "use_in_pass_through": True,
                },
            }
        ],
    }
    path: Final = directory / "watsonx-passthrough.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_watsonx_passthrough_uses_the_flagged_deployment_key_and_base(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "watsonx-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "POST" and request.target == "/identity/token":
            assert f"apikey={_DEPLOYMENT_KEY}".encode() in request.body, request.body
            return Reply(body=json.dumps({"access_token": _IAM_TOKEN, "expires_in": 3600}).encode())
        if request.method == "POST" and request.target.startswith("/ml/v1/text/generation?"):
            assert request.headers["authorization"] == f"Bearer {_IAM_TOKEN}", request.headers
            assert marker.encode() in request.body, request.body[:400]
            return Reply(
                body=json.dumps(
                    {"model_id": "ibm/granite-3-8b-instruct", "results": [{"generated_text": marker}]}
                ).encode()
            )
        return Reply(status=404)

    with wire_server(respond) as wire:
        with owned_proxy(
            gateway,
            tmp_path,
            {"WATSONX_IAM_URL": f"{wire.url}/identity/token"},
            config=_config(tmp_path, wire.url),
            remove_environment=_WATSONX_ENV,
        ) as candidate:
            response: Final = candidate.request(
                "POST",
                "/watsonx/ml/v1/text/generation",
                {"model_id": "ibm/granite-3-8b-instruct", "input": marker, "project_id": "synthetic-project"},
            )
            assert response.status_code == 200, response.text
            assert response.json()["results"][0]["generated_text"] == marker
        forwarded: Final = tuple((request.method, request.target.split("?")[0]) for request in wire.drain())
        assert forwarded == (("POST", "/identity/token"), ("POST", "/ml/v1/text/generation")), forwarded
