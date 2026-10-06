import base64
import json
import re
import signal
import threading
import uuid
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_API_KEY: Final = "synthetic-provider-key"
_ENVIRONMENT_KEY: Final = "synthetic-environment-key"
_XAI_BACKEND: Final = "grok-test"
_GEMINI_BACKEND: Final = "gemini-2.5-flash-image"
_BEDROCK_BACKEND: Final = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
_OPENAI_BACKEND: Final = "gpt-4o-mini"
_XAI_MODEL: Final = "faults-xai-chat"
_GEMINI_MODEL: Final = "faults-gemini-image"
_BEDROCK_MODEL: Final = "faults-bedrock-invoke"
_OPENAI_MODEL: Final = "faults-openai"
_XAI_CHAT: Final = "/xai/v1/chat/completions"
_OPENAI_CHAT: Final = "/openai/v1/chat/completions"
_GENERATE: Final = f"/models/{_GEMINI_BACKEND}:generateContent"
_INVOKE: Final = f"/model/{_BEDROCK_BACKEND}/invoke"
_OPENAI_MODEL_LISTING: Final = "/openai/v1/models"
_CONTAINERS: Final = "/openai/v1/containers"
_CONTAINER: Final = "cntr_faults"
_CONTAINER_FILES: Final = f"{_CONTAINERS}/{_CONTAINER}/files"
_FILE: Final = "cfile_faults"
_BETAS: Final = ("context-1m-2025-08-07", "interleaved-thinking-2025-05-14")
_IMAGE_CONFIG: Final[dict[str, JsonValue]] = {"aspectRatio": "1:1"}
_SERVER_TOOL_USAGE: Final[dict[str, JsonValue]] = {"web_search_calls": 2, "x_search_calls": 1}
_PROMPT_TOKENS: Final = 1000
_COMPLETION_TOKENS: Final = 500
_BURST_ROUNDS: Final = 8
_GATE_SECONDS: Final = 60
_PNG: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_IMAGE_USAGE: Final[dict[str, JsonValue]] = {
    "total_tokens": 1553,
    "input_tokens": 263,
    "input_tokens_details": {"image_tokens": 258, "text_tokens": 5},
    "output_tokens": 1290,
    "output_tokens_details": {"image_tokens": 1290, "text_tokens": 0},
}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_PACKAGED_PRICES: Final = Path(__file__).resolve().parents[3] / "litellm" / "model_prices_and_context_window_backup.json"


def _chat_reply(model: str, usage: dict[str, JsonValue]) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-faults",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "faults control"}, "finish_reason": "stop"}
                ],
                "usage": usage,
            }
        ).encode()
    )


_XAI_REPLIES: Final = MappingProxyType(
    {
        "healthy": _chat_reply(
            _XAI_BACKEND,
            {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
                "server_side_tool_usage_details": _SERVER_TOOL_USAGE,
            },
        ),
        "dropped": Reply(drop_connection=True),
        "overloaded": Reply(
            status=503, body=json.dumps({"error": {"message": "overloaded", "type": "server_error"}}).encode()
        ),
    }
)
_REPLIES: Final = MappingProxyType(
    {
        ("POST", _OPENAI_CHAT): _chat_reply(
            _OPENAI_BACKEND,
            {
                "prompt_tokens": _PROMPT_TOKENS,
                "completion_tokens": _COMPLETION_TOKENS,
                "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
            },
        ),
        ("POST", _GENERATE): Reply(
            body=json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(_PNG).decode()}}
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 263,
                        "candidatesTokenCount": 1290,
                        "totalTokenCount": 1553,
                        "promptTokensDetails": [
                            {"modality": "TEXT", "tokenCount": 5},
                            {"modality": "IMAGE", "tokenCount": 258},
                        ],
                        "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 1290}],
                    },
                }
            ).encode()
        ),
        ("POST", _INVOKE): Reply(
            body=json.dumps(
                {
                    "id": "msg_faults",
                    "type": "message",
                    "role": "assistant",
                    "model": _BEDROCK_BACKEND,
                    "content": [{"type": "text", "text": "faults control"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 12, "output_tokens": 5},
                }
            ).encode()
        ),
        ("GET", _OPENAI_MODEL_LISTING): Reply(body=json.dumps({"object": "list", "data": []}).encode()),
        ("POST", _CONTAINERS): Reply(
            body=json.dumps(
                {"id": _CONTAINER, "object": "container", "created_at": 1, "status": "running", "name": "faults"}
            ).encode()
        ),
        ("GET", _CONTAINER_FILES): Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": _FILE,
                            "object": "container.file",
                            "container_id": _CONTAINER,
                            "created_at": 1,
                            "bytes": 5,
                            "path": "/mnt/data/notes.txt",
                            "source": "user",
                        }
                    ],
                    "first_id": _FILE,
                    "last_id": _FILE,
                    "has_more": False,
                }
            ).encode()
        ),
    }
)


def _path(request: Request) -> str:
    return request.target.split("?")[0]


def _marker(request: Request) -> str:
    messages: Final = _JSON_OBJECT.validate_json(request.body)["messages"]
    assert isinstance(messages, list), request.body
    return str(object_value(messages[-1])["content"])


def _peer(request: Request) -> Reply:
    path: Final = _path(request)
    if path != _GENERATE:
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", path
    if path == _XAI_CHAT:
        return _XAI_REPLIES[_marker(request).split("-")[0]]
    return _REPLIES[(request.method, path)]


@dataclass(frozen=True, slots=True)
class _Call:
    send: Callable[[Gateway], httpx.Response]
    check: Callable[[httpx.Response], None]


def _uncached(marker: str) -> str:
    return f"{marker}-{uuid.uuid4().hex}"


def _chat(model: str, marker: str, gateway: Gateway) -> httpx.Response:
    return gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": _uncached(marker)}]}
    )


def _beta_chat(gateway: Gateway) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": _BEDROCK_MODEL,
            "messages": [{"role": "user", "content": _uncached("synthetic beta request")}],
            "max_tokens": 16,
        },
        headers={"anthropic-beta": json.dumps(_BETAS)},
    )


def _edit(image_config: str, gateway: Gateway) -> httpx.Response:
    return gateway.request_multipart(
        "/v1/images/edits",
        {"model": _GEMINI_MODEL, "prompt": "synthetic edit request", "imageConfig": image_config},
        {"image": ("pixel.png", _PNG, "image/png")},
    )


def _list_files(container: str, gateway: Gateway) -> httpx.Response:
    return gateway.request("GET", f"/v1/containers/{container}/files")


def _error(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(_JSON_OBJECT.validate_json(response.content)["error"])


def _check_enriched(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    usage: Final = object_value(_JSON_OBJECT.validate_json(response.content)["usage"])
    assert usage["server_side_tool_usage_details"] == _SERVER_TOOL_USAGE, response.text
    assert object_value(usage["prompt_tokens_details"])["web_search_requests"] == 2, response.text


def _check_dropped(response: httpx.Response) -> None:
    assert response.status_code == 500, response.text
    error: Final = _error(response)
    assert error["type"] == "internal_server_error" and error["code"] == "500", response.text
    assert "XaiException - Server disconnected" in str(error["message"]), response.text


def _check_overloaded(response: httpx.Response) -> None:
    assert response.status_code == 503, response.text
    error: Final = _error(response)
    assert error["type"] == "internal_server_error" and error["code"] == "503", response.text
    assert str(error["message"]).startswith(
        "litellm.ServiceUnavailableError: ServiceUnavailableError: XaiException - "
    ), response.text


def _check_image(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    assert _JSON_OBJECT.validate_json(response.content)["usage"] == _IMAGE_USAGE, response.text


def _check_rejected_image_config(response: httpx.Response) -> None:
    assert response.status_code == 400, response.text
    error: Final = _error(response)
    assert error["type"] == "invalid_request_error" and error["code"] == "400", response.text
    assert str(error["message"]).startswith(
        "litellm.UnsupportedParamsError: `imageConfig` must be valid JSON when provided as a string."
    ), response.text


def _check_beta(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    usage: Final = object_value(_JSON_OBJECT.validate_json(response.content)["usage"])
    assert usage["prompt_tokens"] == 12 and usage["completion_tokens"] == 5, response.text


def _check_priced(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    packaged: Final = object_value(_JSON_OBJECT.validate_json(_PACKAGED_PRICES.read_bytes())[_OPENAI_BACKEND])
    input_rate: Final = packaged["input_cost_per_token"]
    output_rate: Final = packaged["output_cost_per_token"]
    assert isinstance(input_rate, float) and isinstance(output_rate, float), packaged
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(
        _PROMPT_TOKENS * input_rate + _COMPLETION_TOKENS * output_rate, rel=1e-9
    )


def _check_files(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    listed: Final = _JSON_OBJECT.validate_json(response.content)["data"]
    assert isinstance(listed, list) and [object_value(item)["id"] for item in listed] == [_FILE], response.text


_ENRICHED: Final = _Call(partial(_chat, _XAI_MODEL, "healthy"), _check_enriched)
_IMAGE: Final = _Call(partial(_edit, json.dumps(_IMAGE_CONFIG)), _check_image)
_REJECTED_IMAGE_CONFIG: Final = _Call(partial(_edit, "{not json"), _check_rejected_image_config)
_BETA: Final = _Call(_beta_chat, _check_beta)
_PRICED: Final = _Call(partial(_chat, _OPENAI_MODEL, "synthetic priced request"), _check_priced)


def _burst(gateway: Gateway, calls: tuple[_Call, ...]) -> None:
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures: Final = tuple(pool.submit(call.send, gateway) for call in calls)
        responses: Final = tuple(future.result(timeout=60) for future in futures)
    for call, response in zip(calls, responses, strict=True):
        call.check(response)


def _attempt(call: _Call, gateway: Gateway) -> httpx.Response | None:
    try:
        return call.send(gateway)
    except httpx.TransportError:
        return None


def _faults_config(wire: Wire, directory: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        {
            "model_name": _XAI_MODEL,
            "litellm_params": {"model": f"xai/{_XAI_BACKEND}", "api_base": f"{wire.url}/xai/v1", "api_key": _API_KEY},
        },
        {
            "model_name": _GEMINI_MODEL,
            "litellm_params": {"model": f"gemini/{_GEMINI_BACKEND}", "api_base": wire.url, "api_key": _API_KEY},
        },
        {
            "model_name": _BEDROCK_MODEL,
            "litellm_params": {
                "model": f"bedrock/invoke/{_BEDROCK_BACKEND}",
                "api_key": _API_KEY,
                "aws_region_name": "us-east-1",
                "api_base": wire.url,
                "aws_bedrock_runtime_endpoint": wire.url,
            },
        },
        {
            "model_name": _OPENAI_MODEL,
            "litellm_params": {
                "model": f"openai/{_OPENAI_BACKEND}",
                "api_base": f"{wire.url}/openai/v1",
                "api_key": _API_KEY,
            },
        },
    ]
    path: Final = directory / "validated-reply-boundaries-under-faults.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _environment(wire: Wire) -> dict[str, str]:
    return {
        "OPENAI_API_KEY": _ENVIRONMENT_KEY,
        "OPENAI_API_BASE": f"{wire.url}/environment/v1",
        "OPENAI_BASE_URL": f"{wire.url}/environment/v1",
    }


def _sent_betas(request: Request) -> tuple[str, ...]:
    betas: Final = _JSON_OBJECT.validate_json(request.body)["anthropic_beta"]
    assert isinstance(betas, list), request.body
    return tuple(sorted(str(beta) for beta in betas))


def _sent_image_config(request: Request) -> JsonValue:
    return object_value(_JSON_OBJECT.validate_json(request.body)["generationConfig"])["imageConfig"]


def _assert_outbound(received: tuple[Request, ...], expected: dict[tuple[str, str], int]) -> None:
    assert (
        Counter(
            (request.method, _path(request)) for request in received if _path(request) != _OPENAI_MODEL_LISTING
        )
        == expected
    )
    assert {_sent_betas(request) for request in received if _path(request) == _INVOKE} == {tuple(sorted(_BETAS))}
    assert all(_sent_image_config(request) == _IMAGE_CONFIG for request in received if _path(request) == _GENERATE)


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def test_xai_usage_enrichment_survives_a_burst_the_provider_partly_drops_and_overloads(gateway: Gateway) -> None:
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"xai/{_XAI_BACKEND}", api_base=f"{wire.url}/xai/v1", api_key=_API_KEY)
        kinds: Final = (("healthy", _check_enriched), ("dropped", _check_dropped), ("overloaded", _check_overloaded))
        calls: Final = tuple(
            _Call(partial(_chat, model, f"{kind}-{index}"), check) for index, (kind, check) in product(range(10), kinds)
        )
        _burst(gateway, calls)
        _check_enriched(_chat(model, "healthy-after-the-burst", gateway))
        received: Final = wire.drain()
        assert Counter(_marker(request).split("-")[0] for request in received) == {
            "healthy": 11,
            "dropped": 10,
            "overloaded": 10,
        }


@pytest.mark.timeout(180)
def test_a_worker_killed_mid_burst_leaves_the_sibling_validating_provider_payloads(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()

    def gated(request: Request) -> Reply:
        if _path(request) != _OPENAI_MODEL_LISTING:
            held.put(request.target)
            assert release.wait(timeout=_GATE_SECONDS), "The burst was never released"
        return _peer(request)

    calls: Final = (_ENRICHED, _IMAGE, _BETA) * _BURST_ROUNDS
    with wire_server(gated) as wire:
        config: Final = _faults_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, _environment(wire), config=config, workers=2) as owned:
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            with (
                httpx.Client(
                    base_url=owned.gateway.client.base_url, timeout=2 * _GATE_SECONDS, trust_env=False
                ) as outlasting_the_gate,
                ThreadPoolExecutor(max_workers=len(calls)) as pool,
            ):
                patient: Final = Gateway(outlasting_the_gate, owned.gateway.key, owned.gateway.upstream_url)
                futures: Final = tuple(pool.submit(_attempt, call, patient) for call in calls)
                eventually(held.qsize, lambda size: size == len(calls), seconds=_GATE_SECONDS)
                held_by: Final = {pid: _open_upstream_connections(pid, wire.url) for pid in workers}
                victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
                victim: Final = psutil.Process(victim_pid)
                victim.suspend()
                victim.send_signal(signal.SIGKILL)
                release.set()
                outcomes: Final = tuple(future.result(timeout=60) for future in futures)
            assert sum(held_by.values()) == len(calls), held_by
            served: Final = tuple(
                (call, response) for call, response in zip(calls, outcomes, strict=True) if response is not None
            )
            assert len(served) == held_by[survivor_pid] >= len(calls) // 2, (held_by, len(served))
            for call, response in served:
                call.check(response)
            _burst(owned.gateway, (_ENRICHED, _IMAGE, _BETA, _REJECTED_IMAGE_CONFIG, _PRICED))
            _assert_outbound(
                wire.drain(),
                {
                    ("POST", _XAI_CHAT): _BURST_ROUNDS + 1,
                    ("POST", _GENERATE): _BURST_ROUNDS + 1,
                    ("POST", _INVOKE): _BURST_ROUNDS + 1,
                    ("POST", _OPENAI_CHAT): 1,
                },
            )


@pytest.mark.timeout(240)
def test_a_restarted_proxy_answers_with_the_same_validated_shapes(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_peer) as wire:
        config: Final = _faults_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, _environment(wire), config=config, workers=2) as first:
            container: Final = str(first.gateway.post("/v1/containers", {"model": _OPENAI_MODEL, "name": "faults"})["id"])
            calls: Final = (
                _ENRICHED,
                _IMAGE,
                _REJECTED_IMAGE_CONFIG,
                _BETA,
                _PRICED,
                _Call(partial(_list_files, container), _check_files),
            ) * (_BURST_ROUNDS // 2)
            _burst(first.gateway, calls)
        with owned_proxy_process(gateway, tmp_path, _environment(wire), config=config, workers=2) as second:
            _burst(second.gateway, calls)
        _assert_outbound(
            wire.drain(),
            {
                ("POST", _XAI_CHAT): _BURST_ROUNDS,
                ("POST", _GENERATE): _BURST_ROUNDS,
                ("POST", _INVOKE): _BURST_ROUNDS,
                ("POST", _OPENAI_CHAT): _BURST_ROUNDS,
                ("POST", _CONTAINERS): 1,
                ("GET", _CONTAINER_FILES): _BURST_ROUNDS,
            },
        )
