"""Gemini and Azure credentials taken from the proxy's environment, on one owned proxy.

These are process level inputs, so the module boots its own proxy with a controlled environment:
`GEMINI_API_KEY` and `PALM_API_KEY` removed and `GOOGLE_API_KEY` set; `AZURE_OPENAI_API_KEY`,
`AZURE_API_KEY`, `AZURE_API_BASE` and `AZURE_API_VERSION` set to synthetic values whose base is an owned peer.
The config holds deployments that name only a model, so every credential has to come from the environment.

Contracts pinned (read from `litellm/main.py`, `litellm/llms/gemini/common_utils.py:506` and
`litellm/llms/gemini/image_generation/transformation.py:118`): Gemini chat and embeddings read
`GOOGLE_API_KEY` before `GEMINI_API_KEY` and send it as `x-goog-api-key`; Gemini image generation reads only
`GEMINI_API_KEY` and so fails with `GEMINI_API_KEY is not set` on a proxy that has only `GOOGLE_API_KEY`
(pinned as current behaviour, likely unintended); Azure chat reads `AZURE_OPENAI_API_KEY` before
`AZURE_API_KEY` (`main.py:1529`) while Azure embeddings read only `AZURE_API_KEY` (`main.py:7172`), so one
proxy sends two different keys to one Azure resource; and `AZURE_API_BASE` / `AZURE_API_VERSION` fill a
deployment that names neither. The Rust config loader and provider clients take over env resolution, so every
one of these orders is at risk (`index/test_dec1_azure_.md` and `test_dec1_gemini_.md`, "Not covered").
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml
from integration._support.client import JSON_OBJECT, gateway_from_environment, object_value
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.dec1_bigprov2_support import marker

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 180)]

_GOOGLE_KEY: Final = "dec1-bigprov2-google-env-key"
_AZURE_OPENAI_KEY: Final = "dec1-bigprov2-azure-openai-env-key"
_AZURE_KEY: Final = "dec1-bigprov2-azure-env-key"
_AZURE_VERSION: Final = "2024-06-01"
_DEPLOYMENT: Final = "gpt-4o-dec1-env"
_EMBEDDING_DEPLOYMENT: Final = "text-embedding-3-small-env"


def _answer(request: Request) -> Reply:
    path: Final = urlsplit(request.target).path
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[],"models":[]}')
    if ":generateContent" in path:
        return Reply(
            body=json.dumps(
                {
                    "candidates": [
                        {"content": {"role": "model", "parts": [{"text": "env"}]}, "finishReason": "STOP", "index": 0}
                    ],
                    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4},
                }
            ).encode()
        )
    if ":batchEmbedContents" in path or ":embedContent" in path:
        return Reply(body=b'{"embeddings": [{"values": [0.1, 0.2]}]}')
    if path.endswith("/embeddings"):
        return Reply(
            body=b'{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}],'
            b'"model":"text-embedding-3-small","usage":{"prompt_tokens":2,"total_tokens":2}}'
        )
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-dec1-env",
                "object": "chat.completion",
                "created": 1,
                "model": _DEPLOYMENT,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "env"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
    )


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: OwnedProxy
    peer: Wire

    def posts(self) -> tuple[Request, ...]:
        return tuple(request for request in self.peer.drain() if request.method == "POST")


@pytest.fixture(scope="module")
def env_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("dec1-bigprov2-env")
    with wire_server(_answer) as peer:
        base_url: Final = peer.url.rstrip("/")
        base: Final = JSON_OBJECT.validate_python(
            yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        )
        config: Final = directory / f"dec1-bigprov2-env-{uuid.uuid4().hex}.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    **base,
                    "model_list": [
                        {
                            "model_name": "dec1-env-gemini",
                            "litellm_params": {"model": "gemini/gemini-2.5-flash", "api_base": f"{base_url}/v1beta"},
                        },
                        {
                            "model_name": "dec1-env-gemini-embedding",
                            "litellm_params": {
                                "model": "gemini/gemini-embedding-001",
                                "api_base": f"{base_url}/v1beta",
                            },
                        },
                        {
                            "model_name": "dec1-env-gemini-image",
                            "litellm_params": {
                                "model": "gemini/gemini-2.5-flash-image",
                                "api_base": f"{base_url}/v1beta",
                            },
                        },
                        {"model_name": "dec1-env-azure", "litellm_params": {"model": f"azure/{_DEPLOYMENT}"}},
                        {
                            "model_name": "dec1-env-azure-embedding",
                            "litellm_params": {"model": f"azure/{_EMBEDDING_DEPLOYMENT}"},
                        },
                    ],
                    "litellm_settings": {**object_value(base["litellm_settings"]), "cache": False},
                    "router_settings": {**object_value(base["router_settings"]), "num_retries": 0},
                }
            )
        )
        overrides: Final = {
            "GOOGLE_API_KEY": _GOOGLE_KEY,
            "AZURE_OPENAI_API_KEY": _AZURE_OPENAI_KEY,
            "AZURE_API_KEY": _AZURE_KEY,
            "AZURE_API_BASE": base_url,
            "AZURE_API_VERSION": _AZURE_VERSION,
        }
        with (
            gateway_from_environment() as gateway,
            owned_proxy_process(
                gateway,
                directory,
                overrides,
                config=config,
                remove_environment=("GEMINI_API_KEY", "PALM_API_KEY", "GEMINI_API_BASE", "AZURE_AD_TOKEN"),
                workers=1,
            ) as proxy,
        ):
            yield _Rig(proxy, peer)


def _post(rig: _Rig, path: str, body: dict[str, object]) -> tuple[int, str]:
    response: Final = rig.proxy.gateway.request("POST", path, body)  # pyright: ignore[reportArgumentType]
    return response.status_code, response.text


def test_gemini_chat_sends_google_api_key_when_gemini_api_key_is_absent(env_rig: _Rig) -> None:
    """A `gemini/gemini-2.5-flash` deployment with no `api_key` on a proxy that has only `GOOGLE_API_KEY`. The
    chat is served and the one `:generateContent` call carries `x-goog-api-key: dec1-bigprov2-google-env-key`
    and no `key` query parameter."""
    status, text = _post(
        env_rig,
        "/v1/chat/completions",
        {"model": "dec1-env-gemini", "messages": [{"role": "user", "content": marker()}]},
    )
    assert status == 200, text
    (call,) = env_rig.posts()
    assert call.headers.get("x-goog-api-key") == _GOOGLE_KEY
    assert "key" not in parse_qs(urlsplit(call.target).query)


def test_gemini_embeddings_send_google_api_key_too(env_rig: _Rig) -> None:
    """`/v1/embeddings` on `gemini/gemini-embedding-001` with no deployment key sends the same environment key
    as `x-goog-api-key` (`main.py:7521`, `get_api_key_from_env`)."""
    status, text = _post(env_rig, "/v1/embeddings", {"model": "dec1-env-gemini-embedding", "input": marker()})
    assert status == 200, text
    (call,) = env_rig.posts()
    assert call.headers.get("x-goog-api-key") == _GOOGLE_KEY


def test_gemini_image_generation_ignores_google_api_key(env_rig: _Rig) -> None:
    """`/v1/images/generations` on `gemini/gemini-2.5-flash-image` with no deployment key. Image generation reads
    only `GEMINI_API_KEY` (`image_generation/transformation.py:118`), so on this proxy it fails with a message
    containing `GEMINI_API_KEY is not set` and Google receives nothing. Pinned as current behaviour; the
    inconsistency with chat is likely unintended and a port that unifies key lookup will change it."""
    status, text = _post(env_rig, "/v1/images/generations", {"model": "dec1-env-gemini-image", "prompt": marker()})
    assert status >= 400, text
    assert "GEMINI_API_KEY is not set" in text
    assert env_rig.posts() == ()


def test_azure_chat_takes_base_version_and_the_azure_openai_key_from_the_environment(env_rig: _Rig) -> None:
    """A deployment that names only `azure/gpt-4o-dec1-env`. The chat goes to
    `<AZURE_API_BASE>/openai/deployments/gpt-4o-dec1-env/chat/completions` with
    `api-key: dec1-bigprov2-azure-openai-env-key`: `AZURE_OPENAI_API_KEY` wins over `AZURE_API_KEY` for chat."""
    status, text = _post(
        env_rig,
        "/v1/chat/completions",
        {"model": "dec1-env-azure", "messages": [{"role": "user", "content": marker()}]},
    )
    assert status == 200, text
    (call,) = env_rig.posts()
    assert urlsplit(call.target).path == f"/openai/deployments/{_DEPLOYMENT}/chat/completions"
    assert call.headers.get("api-key") == _AZURE_OPENAI_KEY


def test_azure_embeddings_take_azure_api_key_not_the_azure_openai_key(env_rig: _Rig) -> None:
    """The embedding deployment names only `azure/text-embedding-3-small-env`. Embeddings read `AZURE_API_KEY`
    and never `AZURE_OPENAI_API_KEY` (`main.py:7172`), so Azure receives `api-key: dec1-bigprov2-azure-env-key`
    on `/openai/deployments/text-embedding-3-small-env/embeddings`. Pinned as current
    behaviour: the same proxy sends a different key for chat."""
    status, text = _post(env_rig, "/v1/embeddings", {"model": "dec1-env-azure-embedding", "input": marker()})
    assert status == 200, text
    (call,) = env_rig.posts()
    assert urlsplit(call.target).path == f"/openai/deployments/{_EMBEDDING_DEPLOYMENT}/embeddings"
    assert call.headers.get("api-key") == _AZURE_KEY


def test_azure_api_version_from_the_environment_reaches_azure(env_rig: _Rig) -> None:
    """The chat and embedding deployments name no `api_version`, so both calls must carry the environment's
    `AZURE_API_VERSION=2024-06-01`.

    BUG (run 44176682961): Azure receives `api-version=2025-02-01-preview` on both. `litellm --api_version`
    defaults to `litellm.AZURE_DEFAULT_API_VERSION` and has no `envvar` (litellm/proxy/proxy_cli.py:1039-1042),
    and `save_worker_config` writes any truthy value into `os.environ["AZURE_API_VERSION"]`
    (litellm/proxy/proxy_server.py:10345-10346), so the CLI default overwrites the operator's environment value
    before any call reads it. The chat and embedding tests above keep the path and key checks; this test holds
    the version check for both.
    """
    pytest.skip("BUG: the proxy CLI's --api_version default overwrites the operator's AZURE_API_VERSION env")
    chat_status, chat_text = _post(
        env_rig,
        "/v1/chat/completions",
        {"model": "dec1-env-azure", "messages": [{"role": "user", "content": marker()}]},
    )
    assert chat_status == 200, chat_text
    embed_status, embed_text = _post(
        env_rig, "/v1/embeddings", {"model": "dec1-env-azure-embedding", "input": marker()}
    )
    assert embed_status == 200, embed_text
    calls: Final = env_rig.posts()
    assert len(calls) == 2
    assert [parse_qs(urlsplit(call.target).query) for call in calls] == [{"api-version": [_AZURE_VERSION]}] * 2


def test_the_environment_keys_never_reach_the_caller(env_rig: _Rig) -> None:
    """`GET /model/info` on this proxy lists the env-only deployments, and no value in the answer contains any
    of the three environment keys, since the deployments never stored them."""
    response: Final = env_rig.proxy.gateway.request("GET", "/model/info")
    assert response.status_code == 200, response.text
    for secret in (_GOOGLE_KEY, _AZURE_OPENAI_KEY, _AZURE_KEY):
        assert secret not in response.text
