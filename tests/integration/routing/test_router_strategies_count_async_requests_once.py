"""Router strategies that keep both a sync and an async hook count a finished async request once: least-busy
keeps counting a request still in flight while others finish around it, and usage-based routing v2 adds a
request's tokens to its deployment once.

A legacy sync callback is configured, because that is what makes the sync handler run for an async call."""

import base64
import json
import os
import threading
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, held
from integration._support.hook_counts import PNG_BYTES, WAV_BYTES, HookRecords
from integration._support.process import OwnedState, owned_proxy
from integration._support.wire import Reply, Request, wire_server
from redis import Redis

pytestmark = pytest.mark.timeout(240)

_MODEL: Final = "router-strategy-model"
_PROBES: Final = "integration._support.hook_count_probe"
_REJECTION: Final = b'{"error":{"message":"scripted provider rejection","type":"invalid_request_error"}}'
_IMAGE_TOKENS: Final = 12
_IMAGE_REPLY: Final = json.dumps(
    {
        "created": 1700000000,
        "data": [{"b64_json": base64.b64encode(PNG_BYTES).decode()}],
        "usage": {
            "input_tokens": 5,
            "output_tokens": 7,
            "total_tokens": _IMAGE_TOKENS,
            "input_tokens_details": {"text_tokens": 5, "image_tokens": 0},
        },
    }
).encode()


@dataclass(frozen=True)
class _Rig:
    proxy: Gateway
    records: HookRecords
    release: threading.Event
    deployment_id: str


def _provider(release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == "/v1/images/generations":
            return Reply(body=_IMAGE_REPLY)
        spoken: Final = str(json.loads(request.body or b"{}").get("input", ""))
        if spoken.startswith("fail"):
            return Reply(status=400, body=_REJECTION)
        if spoken.startswith("hold"):
            return Reply(chunks=(WAV_BYTES,), content_type="audio/wav", gate_before_first=release)
        return Reply(body=WAV_BYTES, content_type="audio/wav")

    return respond


def _write_config(path: Path, provider_url: str, strategy: str, model: str, deployment_id: str) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    {
                        "model_name": _MODEL,
                        "litellm_params": {
                            "model": model,
                            "api_base": provider_url + "/v1",
                            "api_key": "synthetic-provider-key",
                        },
                        "model_info": {"id": deployment_id},
                    }
                ],
                "litellm_settings": {
                    "success_callback": [f"{_PROBES}.sync_success_function"],
                    "failure_callback": [f"{_PROBES}.sync_failure_function"],
                },
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "disable_model_info_refresh": True,
                },
                "router_settings": {
                    "routing_strategy": strategy,
                    "routing_strategy_args": {"ttl": 600},
                    "num_retries": 0,
                    "disable_cooldowns": True,
                    "redis_host": "os.environ/REDIS_HOST",
                    "redis_port": "os.environ/REDIS_PORT",
                },
            }
        )
    )
    return path


@contextmanager
def _router_proxy(tmp_path: Path, strategy: str, model: str) -> Iterator[_Rig]:
    release: Final = threading.Event()
    deployment_id: Final = f"{strategy}-{uuid.uuid4().hex}"
    with (
        wire_server(_provider(release)) as upstream,
        wire_server(lambda request: Reply()) as sink,
        gateway_from_environment() as rig,
    ):
        config: Final = _write_config(tmp_path / f"{strategy}.yaml", upstream.url, strategy, model, deployment_id)
        try:
            with owned_proxy(
                rig,
                tmp_path,
                {"HOOK_COUNT_PROBE_URL": sink.url, "STORE_MODEL_IN_DB": "False"},
                config=config,
                workers=2,
                state=OwnedState.SHARED_WITH_WORKER,
            ) as proxy:
                yield _Rig(proxy, HookRecords(sink, function_callbacks=True), release, deployment_id)
        finally:
            release.set()


@pytest.fixture(scope="module")
def least_busy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    with _router_proxy(tmp_path_factory.mktemp("least-busy"), "least-busy", "openai/gpt-4o-mini-tts") as rig:
        yield rig


@pytest.fixture(scope="module")
def usage_based(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    with _router_proxy(tmp_path_factory.mktemp("usage-based"), "usage-based-routing-v2", "openai/gpt-image-1") as rig:
        yield rig


def _redis() -> Redis:
    return Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))


def _in_flight(rig: _Rig) -> int:
    with _redis() as cache:
        stored: Final = cache.get(f"{_MODEL}_request_count:{rig.deployment_id}")
    return 0 if stored is None else int(stored)


def _tokens_counted(rig: _Rig) -> int:
    with _redis() as cache:
        keys: Final = tuple(cache.scan_iter(match=f"{rig.deployment_id}:*:tpm:*"))
        return sum(int(cache.get(key) or 0) for key in keys)


def _post(rig: _Rig, path: str, body: dict[str, object], call_id: str) -> httpx.Response:
    return rig.proxy.client.post(
        path,
        json={"model": _MODEL, **body},
        headers={"Authorization": f"Bearer {rig.proxy.key}", "x-litellm-call-id": call_id},
        timeout=120,
    )


def _speak(rig: _Rig, text: str, call_id: str) -> httpx.Response:
    return _post(rig, "/v1/audio/speech", {"input": text, "voice": "alloy", "response_format": "wav"}, call_id)


def _logged(rig: _Rig, call_id: str) -> None:
    """The request's success or failure logging, the legacy sync callback included, has run."""
    eventually(lambda: rig.records.for_call(call_id), bool, seconds=20)


def _finish_a_speech_request(rig: _Rig, outcome: str) -> None:
    call_id: Final = str(uuid.uuid4())
    finished: Final = _speak(rig, f"{'ok' if outcome == 'success' else 'fail'} {call_id}", call_id)
    assert finished.status_code == (200 if outcome == "success" else 400), finished.text
    _logged(rig, call_id)
    held(lambda: _in_flight(rig), lambda count: count == 1, holding=3, seconds=20)


def _generate_an_image(rig: _Rig) -> None:
    call_id: Final = str(uuid.uuid4())
    response: Final = _post(rig, "/v1/images/generations", {"prompt": f"tokens {call_id}", "n": 1}, call_id)
    assert response.status_code == 200, response.text
    _logged(rig, call_id)


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_least_busy_keeps_counting_a_request_in_flight_while_async_speech_requests_finish(
    least_busy: _Rig, outcome: str
) -> None:
    held(lambda: _in_flight(least_busy), lambda count: count == 0, holding=1, seconds=30)
    least_busy.release.clear()
    with ThreadPoolExecutor(max_workers=1) as pool:
        in_flight: Final = pool.submit(_speak, least_busy, f"hold {uuid.uuid4().hex}", str(uuid.uuid4()))
        eventually(lambda: _in_flight(least_busy), lambda count: count == 1, seconds=30)
        try:
            for _ in range(3):
                _finish_a_speech_request(least_busy, outcome)
        finally:
            least_busy.release.set()
        assert in_flight.result(timeout=60).status_code == 200
    held(lambda: _in_flight(least_busy), lambda count: count == 0, holding=3, seconds=30)


def test_usage_based_routing_counts_an_async_image_request_s_tokens_once(usage_based: _Rig) -> None:
    for _ in range(3):
        _generate_an_image(usage_based)
    held(lambda: _tokens_counted(usage_based), lambda tokens: tokens == 3 * _IMAGE_TOKENS, holding=3, seconds=30)
