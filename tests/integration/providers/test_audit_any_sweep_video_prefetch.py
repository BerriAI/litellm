from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import openai
import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy
from pydantic import JsonValue

PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"

_PRIVATE_KEY: Final = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDETwmymH0fOIKM
+nxQ24iK33rwAXYw1PpMukoW9WkwwvUOS1RX1Ow5+IVWbUll2OYfpxXQbo6gUm3n
7fUyMvLK+FbgGUC0/9D884xGbY/jZdxQxzfYy9tCy6H5DYGMZbSY51eqZr6OSWnc
KLnocQpmkoAY3ZpsFrl8rzXxba9gjG3iV5VvTG0RA/Q2cDY7w0/G57bWYWhweAuC
qlMHRNmTGh7eoAJIZSJxR4+Beb4OEuaUbEe85hIo1Wb/6yicMLBeV6Xrh8Odl7gG
20nEhziF8yFJ6BA1BY3mTVwkj1LV7rJUcba209waV7HvISY/fUzWWGXd2DLVVrGc
+sfmOH31AgMBAAECggEAErG4KbLNz4hRHl3IMucFFLx9d5ecCsBtaZsf7RnHv6YB
MjBIQuj66fZZtEZKBBHosED4GdY9bNUMtFZ70CD78ApAM60T9WibHUuQIzDR0ATm
OycNkW6wXacIsK/xk04oKZt2RLruEJy7IQUb1x/UrrnqyvNnfj2XwmDpESmx1s69
yKWsVEQOy3UM4LK+50I+BGnsJzIOkFsE4p70r4LzV8BbMtSd6Oum1K/+w56KON7g
MsWUrx0LHc5uTiXdZZ/fVDFGm8eVz9k3qZ4xjtTNcvm3s3RE/IFinOLuTOx0GNMc
ZHBry1vz7/GzwIRvxcPoK1Omen/QfHVB6M+w/i14DQKBgQD4S7AeDctINtF2aL3A
m/XnfmslrVzSYupngnSQFynvKHHG+Abp8+qpZx3br4Lj/jrM0Lbddig25kmEjdgR
/5PcTMDbEXc7uuspRkVRTcsTii2AbqP3XyS5wGMQNerEWxPzk19iz3GRgTByxbKb
mIA+lOxEFamxBXhYSmKUoyus9wKBgQDKZmWhtw3rPY1AvIaLXTjyshJ37bybiFco
TVUWZH28kc1YOKTLzrEz051bkB2+j/FTFSP73ZR1dO+4axgm/9OhvFOcf3yM/rra
MW9nyd2lQlvk+20ZFz9U79Rs/+GCkq2TAVsrBFARF//CAPAw5PDArc1a6laqU+Dg
hMb4UqjNcwKBgQDcduAPH3JjUYFVIRVzQ0fp358l2jPz/q5hcD0tf9PVLdUwtSP2
cYToG84wNTwEUsVE/QGRzItSi9G1Z/j1qjhQwLnxCiYTnBmYEXQVB/0MAYH7asIe
MJW8dS3AxA8LgE9L1Iw6gTU23D33esDnVVZxuuu6XH3K16+OvaWgku2B6wKBgD69
TQW686KW+uE1T1OF+zAoCNeKJyvjMHk6t4RwdBVu8MNHE7ZlhMJ291E5tC6s+fZy
GCxrvChIrkJxWBRGuvhoo1ZgomAmWg9mwYzfo0gl8MxjL6dWcHhIu4VqMCJEi2+F
PRhxhsw1rz8ZAVt2rDZd6ckj2OeFhrCyF+dD4DWHAoGAboLVvPHyqIct41q2iSqv
8824wHwCDtli0ujY8F5j0LLQ5MvMYyD6+Wx4BTfH1vzHh3geWW8Xuf+038wf8vbU
YjwfNBvU2JsY6Y1zqt8AU/AkW+R1OgF4IC8g5HjFRFpXKMCuhZ1I2nb44J15lw7b
WiGVerclMK3keqMh6veZzdk=
-----END PRIVATE KEY-----
"""

OPERATION_NAME: Final = (
    "projects/test-project/locations/us-central1/publishers/google/models/veo-3.0-generate-preview/operations/op-123"
)

VERTEX_MODELS_PREFIX: Final = "v1/projects/test-project/locations/global/publishers/google/models"

_DONE_PREFETCH: Final = {
    "content_type": "application/json",
    "body": {
        "done": True,
        "response": {"videos": [{"gcsUri": "gs://bucket/source.mp4", "mimeType": "video/mp4"}]},
    },
}

_EDIT_RESULT: Final = {
    "content_type": "application/json",
    "body": {"name": OPERATION_NAME.replace("op-123", "op-edited"), "done": False},
}

_DEPLOYMENTS: Final = ("audit-veo-raw", "audit-veo-sdk", "audit-veo-async", "audit-veo-nonobj", "audit-veo-err")


def _vertex_credentials(upstream_url: str) -> str:
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "audit-key-id",
            "private_key": _PRIVATE_KEY,
            "client_email": "audit-sa@test-project.iam.gserviceaccount.com",
            "token_uri": f"{upstream_url}/_oauth/token",
            "universe_domain": "googleapis.com",
        }
    )


def _video_config(directory: Path, upstream_url: str) -> Path:
    config: Final = yaml.safe_load(PROXY_CONFIG.read_text())
    credentials: Final = _vertex_credentials(upstream_url)
    config["model_list"] = [
        {
            "model_name": name,
            "litellm_params": {
                "model": "vertex_ai/veo-3.0-generate-preview",
                "api_base": f"{upstream_url}/{name}",
                "vertex_project": "test-project",
                "vertex_credentials": credentials,
            },
        }
        for name in _DEPLOYMENTS
    ]
    path: Final = directory / "video_proxy_config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def video_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with gateway_from_environment() as upstream_gateway:
        directory: Final = tmp_path_factory.mktemp("audit_video")
        config: Final = _video_config(directory, upstream_gateway.upstream_url)
        with owned_proxy(upstream_gateway, directory, {}, config=config, workers=2) as owned:
            yield owned


def _register_scenario(gateway: Gateway, scenario_id: str, response: dict[str, JsonValue]) -> None:
    with httpx.Client(timeout=5, trust_env=False) as client:
        result: Final = client.post(
            f"{gateway.upstream_url}/__scenarios",
            json={"scenario_id": scenario_id, "response": response},
        )
    assert result.status_code == 200, result.text


def _video_edit_body(model: str) -> dict[str, JsonValue]:
    return {
        "model": model,
        "prompt": "make it brighter",
        "video": {"id": OPERATION_NAME},
        "custom_llm_provider": "vertex_ai",
    }


def _routed_veo(prefetch_response: dict[str, JsonValue], edit_response: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "content_type": "application/x-routed",
        "routes": {
            f"POST /{VERTEX_MODELS_PREFIX}/veo-3.0-generate-preview:fetchPredictOperation": prefetch_response,
            f"POST /{VERTEX_MODELS_PREFIX}/veo-3.0-generate-preview:predictLongRunning": edit_response,
        },
    }


def test_vertex_video_edit_prefetches_source_object_raw_httpx(video_gateway: Gateway) -> None:
    _register_scenario(video_gateway, "audit-veo-raw", _routed_veo(_DONE_PREFETCH, _EDIT_RESULT))
    response: Final = video_gateway.request("POST", "/v1/videos/edits", _video_edit_body("audit-veo-raw"))
    assert response.status_code == 200, response.text
    payload: Final = response.json()
    assert payload.get("id") or payload.get("name"), payload


def test_vertex_video_edit_prefetches_source_object_openai_sdk(video_gateway: Gateway) -> None:
    _register_scenario(video_gateway, "audit-veo-sdk", _routed_veo(_DONE_PREFETCH, _EDIT_RESULT))
    client: Final = openai.OpenAI(
        base_url=f"{video_gateway.client.base_url}/v1",
        api_key=video_gateway.key,
        http_client=httpx.Client(trust_env=False),
    )
    response: Final = client.post("/videos/edits", body=_video_edit_body("audit-veo-sdk"), cast_to=httpx.Response)
    payload: Final = response.json()
    assert isinstance(payload, dict) and (payload.get("id") or payload.get("name")), payload


async def test_vertex_video_edit_prefetches_source_object_openai_async(video_gateway: Gateway) -> None:
    _register_scenario(video_gateway, "audit-veo-async", _routed_veo(_DONE_PREFETCH, _EDIT_RESULT))
    client: Final = openai.AsyncOpenAI(
        base_url=f"{video_gateway.client.base_url}/v1",
        api_key=video_gateway.key,
        http_client=httpx.AsyncClient(trust_env=False),
    )
    response: Final = await client.post(
        "/videos/edits", body=_video_edit_body("audit-veo-async"), cast_to=httpx.Response
    )
    payload: Final = response.json()
    assert isinstance(payload, dict) and (payload.get("id") or payload.get("name")), payload


@pytest.mark.parametrize("prefetch_body", ('["a", "b"]', '"just-a-string"', "null"))
def test_vertex_video_edit_prefetch_non_object_json(video_gateway: Gateway, prefetch_body: str) -> None:
    _register_scenario(
        video_gateway,
        "audit-veo-nonobj",
        _routed_veo({"content_type": "application/jsonl", "body": prefetch_body}, _EDIT_RESULT),
    )
    response: Final = video_gateway.request("POST", "/v1/videos/edits", _video_edit_body("audit-veo-nonobj"))
    assert response.status_code != 200, response.text
    assert response.status_code < 600, response.text
    payload: Final = response.json()
    assert "error" in payload, payload


@pytest.mark.parametrize("status", (400, 500))
def test_vertex_video_edit_prefetch_http_error(video_gateway: Gateway, status: int) -> None:
    _register_scenario(
        video_gateway,
        "audit-veo-err",
        _routed_veo(
            {"content_type": "application/json", "body": {"error": "upstream said no"}, "status": status}, _EDIT_RESULT
        ),
    )
    response: Final = video_gateway.request("POST", "/v1/videos/edits", _video_edit_body("audit-veo-err"))
    assert response.status_code != 200, response.text
    payload: Final = response.json()
    assert "error" in payload, payload
