import json
import os
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.forward_proxy import tunnelling_forward_proxy
from integration._support.process import owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server

pytestmark = pytest.mark.timeout(240)

_MODEL: Final = "gemini-3.8-flash"
_LOCATION: Final = "us-central1"
_HOST: Final = f"{_LOCATION}-aiplatform.googleapis.com"
_PROJECTS: Final = ("least-busy-vertex-a", "least-busy-vertex-b")
_BURST: Final = 4
_REPLY: Final = json.dumps(
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4},
    }
).encode()
_SCRUBBED: Final = frozenset(
    {
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "DEFAULT_VERTEXAI_PROJECT",
        "DEFAULT_VERTEXAI_LOCATION",
        "DEFAULT_GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }
)


def _held_until_the_burst_arrives(arrived: threading.Barrier) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        arrived.wait()
        return Reply(body=_REPLY)

    return respond


def _token(request: Request) -> Reply:
    return Reply(body=json.dumps({"access_token": "scripted-token", "expires_in": 3600, "token_type": "Bearer"}).encode())


def _write_config(path: Path, token_url: str) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    {
                        "model_name": _MODEL,
                        "litellm_params": {
                            "model": f"vertex_ai/{_MODEL}",
                            "vertex_project": project,
                            "vertex_location": _LOCATION,
                            "vertex_credentials": service_account_json(project, token_url),
                            "use_in_pass_through": True,
                        },
                        "model_info": {"id": project},
                    }
                    for project in _PROJECTS
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "disable_model_info_refresh": True,
                },
                "router_settings": {"routing_strategy": "least-busy", "num_retries": 0, "disable_cooldowns": True},
            }
        )
    )
    return path


def _generate(proxy: Gateway) -> httpx.Response:
    return proxy.client.post(
        f"/vertex_ai/v1/projects/caller-project/locations/{_LOCATION}/publishers/google/models/{_MODEL}:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        headers={"x-litellm-api-key": f"Bearer {proxy.key}"},
        timeout=120,
    )


def _project_of(target: str) -> str:
    return target.split("/projects/", 1)[1].split("/", 1)[0]


def test_least_busy_spreads_a_burst_of_vertex_pass_through_calls_across_the_group(tmp_path: Path) -> None:
    arrived: Final = threading.Barrier(_BURST, timeout=60)
    cert, key = write_self_signed_cert(tmp_path, (_HOST,))
    with (
        wire_server(_held_until_the_burst_arrives(arrived), tls=server_context(cert, key)) as vertex,
        wire_server(_token) as tokens,
        tunnelling_forward_proxy(int(vertex.url.rsplit(":", 1)[1])) as tunnel,
        gateway_from_environment() as rig,
        owned_proxy(
            rig,
            tmp_path,
            {
                "STORE_MODEL_IN_DB": "False",
                "HTTPS_PROXY": tunnel.url,
                "NO_PROXY": "127.0.0.1,localhost",
                "SSL_CERT_FILE": str(cert),
            },
            remove_environment=tuple(name for name in os.environ if name.upper() in _SCRUBBED),
            config=_write_config(tmp_path / "least-busy-vertex.yaml", tokens.url),
        ) as proxy,
        ThreadPoolExecutor(max_workers=_BURST) as pool,
    ):
        responses: Final = tuple(pool.map(lambda _: _generate(proxy), range(_BURST)))
        sent: Final = vertex.drain()
    assert [response.status_code for response in responses] == [200] * _BURST, [r.text for r in responses]
    assert Counter(_project_of(request.target) for request in sent) == Counter(
        {project: _BURST // len(_PROJECTS) for project in _PROJECTS}
    )
