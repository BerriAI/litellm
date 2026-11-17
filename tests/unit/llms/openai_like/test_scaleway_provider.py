import json
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx

import litellm

REPO_ROOT: Final = Path(__file__).resolve().parents[4]
_SCALEWAY_KEY: Final = "synthetic-scaleway-secret"
_WAV_BYTES: Final = (
    b"RIFF\x30\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x08\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07"
)


@pytest.mark.parametrize(
    "cost_map_path",
    [
        REPO_ROOT / "model_prices_and_context_window.json",
        REPO_ROOT / "litellm/model_prices_and_context_window_backup.json",
    ],
)
@pytest.mark.parametrize(
    ("bare_name", "namespaced_name"),
    [
        ("scaleway/qwen3-embedding-8b", "scaleway/qwen/qwen3-embedding-8b"),
        ("scaleway/bge-multilingual-gemma2", "scaleway/BAAI/bge-multilingual-gemma2"),
    ],
)
def test_scaleway_bare_embedding_name_is_priced_like_the_namespaced_name(
    cost_map_path: Path, bare_name: str, namespaced_name: str
) -> None:
    cost_map: Final = json.loads(cost_map_path.read_text())

    assert cost_map[bare_name] == cost_map[namespaced_name], (
        f"{bare_name} drifted from {namespaced_name} in {cost_map_path.name}"
    )


@respx.mock
def test_scaleway_embedding_posts_to_the_scaleway_embeddings_endpoint_with_the_scw_secret_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCW_SECRET_KEY", _SCALEWAY_KEY)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx.post("https://api.scaleway.ai/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5, 0.75]}],
                "model": "qwen3-embedding-8b",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )
    )

    response: Final = litellm.embedding(
        model="scaleway/qwen3-embedding-8b", input=["hello"], encoding_format="float", dimensions=8
    )

    sent: Final = route.calls.last.request
    assert (route.call_count, str(sent.url), sent.headers["authorization"], json.loads(sent.content)) == (
        1,
        "https://api.scaleway.ai/v1/embeddings",
        f"Bearer {_SCALEWAY_KEY}",
        {"model": "qwen3-embedding-8b", "input": ["hello"], "encoding_format": "float", "dimensions": 8},
    )
    assert [row["embedding"] for row in response.data] == [[0.25, 0.5, 0.75]]


def _multipart_parts(request: httpx.Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(parsed.iter_parts())


@respx.mock
def test_scaleway_transcription_posts_every_form_field_through_the_shared_openai_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCW_SECRET_KEY", _SCALEWAY_KEY)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx.post("https://api.scaleway.ai/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "bonjour le monde"})
    )

    response: Final = litellm.transcription(
        model="scaleway/whisper-large-v3", file=("a.wav", _WAV_BYTES), language="fr", response_format="json"
    )

    sent: Final = route.calls.last.request
    parts: Final = _multipart_parts(sent)
    assert (route.call_count, str(sent.url), sent.headers["authorization"]) == (
        1,
        "https://api.scaleway.ai/v1/audio/transcriptions",
        f"Bearer {_SCALEWAY_KEY}",
    )
    assert sorted(
        (part.get_param("name", header="content-disposition"), part.get_filename(), part.get_payload(decode=True))
        for part in parts
    ) == [
        ("file", "a.wav", _WAV_BYTES),
        ("language", None, b"fr"),
        ("model", None, b"whisper-large-v3"),
        ("response_format", None, b"json"),
    ]
    assert response.text == "bonjour le monde"
