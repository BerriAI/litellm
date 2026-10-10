import json
import time
from typing import Final

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_BACKEND: Final = "gemini-3.7-flash"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_BACKEND}"
_PROMPT: Final = "Write a very long numbered list."
_PART_COUNT: Final = 8000
_LINES_PER_FRAGMENT: Final = 64
_STREAM_BUDGET_SECONDS: Final = 10.0
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class _Delta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    content: str | None = None


class _Choice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    delta: _Delta
    finish_reason: str | None = None


class _Chunk(BaseModel):
    model_config = ConfigDict(extra="ignore")
    choices: tuple[_Choice, ...]


def _service_account_json(token_url: str) -> str:
    private_key: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": _PROJECT,
            "private_key_id": "scripted",
            "private_key": private_key,
            "client_email": f"scripted@{_PROJECT}.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": f"{token_url}/_oauth/authorize",
            "token_uri": f"{token_url}/_oauth/token",
        }
    )


def _expected_text() -> str:
    return "".join(f"{index}. item\n" for index in range(_PART_COUNT))


def _gemini_response_fragments() -> tuple[bytes, ...]:
    document: Final = json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": f"{index}. item\n"} for index in range(_PART_COUNT)],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 40000, "totalTokenCount": 40009},
            "modelVersion": _BACKEND,
        },
        indent=2,
    )
    lines: Final = document.split("\n")
    fragments: Final = tuple(
        "\n".join(lines[start : start + _LINES_PER_FRAGMENT]).encode() + b"\n"
        for start in range(0, len(lines), _LINES_PER_FRAGMENT)
    )
    return (b"data: " + fragments[0], *fragments[1:], b"\n")


@pytest.mark.covers("providers.vertex_gemini.fragmented_stream_json_is_parsed_once_and_stays_live")
def test_vertex_gemini_stream_split_across_many_fragments_completes_without_stalling(gateway: Gateway) -> None:
    fragments: Final = _gemini_response_fragments()

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == f"{_MODEL_PATH}:streamGenerateContent?alt=sse"
        assert request.headers["authorization"] == "Bearer scripted-token"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["contents"] == [{"role": "user", "parts": [{"text": _PROMPT}]}]
        assert body["generationConfig"] == {"temperature": 0.0}
        return Reply(content_type="text/event-stream", chunks=fragments)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"vertex_ai/{_BACKEND}",
            api_base=f"{wire.url}{_MODEL_PATH}",
            api_key=None,
            vertex_project=_PROJECT,
            vertex_location=_LOCATION,
            vertex_credentials=_service_account_json(gateway.upstream_url.rstrip("/")),
        )
        started: Final = time.monotonic()
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": _PROMPT}],
                "stream": True,
                "temperature": 0.0,
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=_STREAM_BUDGET_SECONDS,
        ) as response:
            assert response.status_code == 200, response.read()
            lines: Final = tuple(line for line in response.iter_lines() if line.startswith("data: "))
        elapsed: Final = time.monotonic() - started
        assert elapsed < _STREAM_BUDGET_SECONDS, f"stream took {elapsed:.1f}s for {len(fragments)} fragments"
        assert lines[-1] == "data: [DONE]", lines[-3:]
        chunks: Final = tuple(_Chunk.model_validate_json(line.removeprefix("data: ")) for line in lines[:-1])
        choices: Final = tuple(choice for chunk in chunks for choice in chunk.choices)
        assert "".join(choice.delta.content or "" for choice in choices) == _expected_text()
        assert tuple(choice.finish_reason for choice in choices if choice.finish_reason) == ("stop",)
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", f"{_MODEL_PATH}:streamGenerateContent?alt=sse")
        ]
