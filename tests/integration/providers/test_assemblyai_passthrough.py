import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from integration._support.client import Gateway, gateway_from_environment
from integration._support.forward_proxy import tunnelling_forward_proxy
from integration._support.process import owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server

_HOSTS: Final = ("api.assemblyai.com", "api.eu.assemblyai.com")
_PROVIDER_KEY: Final = "synthetic-assemblyai-key"
_TRANSCRIPT_REQUEST: Final = {"audio_url": "https://assembly.ai/wildfires.mp3", "speech_models": ["universal-2"]}
_QUEUED: Final = {"id": "transcript-1", "status": "queued", "audio_url": "https://assembly.ai/wildfires.mp3"}
_COMPLETED: Final = {"id": "transcript-1", "status": "completed", "text": "Smoke from wildfires in Canada."}
_SCRUBBED: Final = frozenset({"HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "SSL_CERT_FILE", "ASSEMBLYAI_API_KEY"})


@dataclass(frozen=True, slots=True)
class _AssemblyAI:
    wire: Wire
    proxy: Gateway


def _respond(request: Request) -> Reply:
    if (request.method, request.target) == ("POST", "/v2/transcript"):
        return Reply(body=json.dumps(_QUEUED).encode())
    if (request.method, request.target) == ("GET", "/v2/transcript/transcript-1"):
        return Reply(body=json.dumps(_COMPLETED).encode())
    return Reply(status=404, body=b'{"error": "unexpected AssemblyAI request"}')


@pytest.fixture(scope="module")
def assemblyai(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_AssemblyAI]:
    directory: Final = tmp_path_factory.mktemp("assemblyai")
    cert, key = write_self_signed_cert(directory, _HOSTS)
    with (
        wire_server(_respond, tls=server_context(cert, key)) as wire,
        tunnelling_forward_proxy(int(wire.url.rsplit(":", 1)[1])) as tunnel,
        gateway_from_environment() as rig,
        owned_proxy(
            rig,
            directory,
            {"HTTPS_PROXY": tunnel.url, "NO_PROXY": "127.0.0.1,localhost", "SSL_CERT_FILE": str(cert)},
            remove_environment=tuple(name for name in os.environ if name.upper() in _SCRUBBED),
        ) as proxy,
        proxy.scenario() as scenario,
    ):
        scenario.model(
            model="assemblyai/*",
            custom_llm_provider="assemblyai",
            api_key=_PROVIDER_KEY,
            api_base="https://api.assemblyai.com",
            use_in_pass_through=True,
        )
        yield _AssemblyAI(wire, proxy)




def test_a_db_assemblyai_deployment_serves_the_transcript_routes(assemblyai: _AssemblyAI) -> None:
    proxy: Final = assemblyai.proxy
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        assemblyai.wire.drain()
        created: Final = proxy.request("POST", "/assemblyai/v2/transcript", _TRANSCRIPT_REQUEST, key=key)
        polled: Final = proxy.request("GET", "/assemblyai/v2/transcript/transcript-1", key=key)
        sent: Final = assemblyai.wire.drain()

    assert (created.status_code, created.json()) == (200, _QUEUED), created.text
    assert (polled.status_code, polled.json()) == (200, _COMPLETED), polled.text
    posts: Final = tuple(request for request in sent if request.method == "POST")
    assert [request.target for request in posts] == ["/v2/transcript"]
    assert json.loads(posts[0].body) == _TRANSCRIPT_REQUEST
    assert {(request.method, request.target) for request in sent} == {
        ("POST", "/v2/transcript"),
        ("GET", "/v2/transcript/transcript-1"),
    }
    assert posts[0].headers["authorization"] == _PROVIDER_KEY


@pytest.mark.parametrize("route", ["/assemblyai/v2/transcript", "/eu.assemblyai/v2/transcript"])
def test_an_invalid_proxy_key_is_rejected_before_assemblyai_is_called(assemblyai: _AssemblyAI, route: str) -> None:
    assemblyai.wire.drain()
    response: Final = assemblyai.proxy.request("POST", route, _TRANSCRIPT_REQUEST, key="sk-12222")

    assert response.status_code == 401, response.text
    assert [request for request in assemblyai.wire.drain() if request.method == "POST"] == []
