import base64
import json
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, held
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server

pytestmark = pytest.mark.timeout(240)

_WAV_BYTES: Final = (
    b"RIFF\x30\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x08\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07"
)
_PNG_BYTES: Final = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff"
    b"\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)
_IMAGE_REPLY: Final = json.dumps({"created": 1700000000, "data": [{"b64_json": base64.b64encode(_PNG_BYTES).decode()}]})
_MESSAGE_REPLY: Final = json.dumps(
    {
        "id": "msg_hook_count",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
)
_RESPONSE_REPLY: Final = json.dumps(
    {
        "id": "resp_hook_count",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_hook_count",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    }
)
_SURFACE_CALL_TYPES: Final = {
    "chat": "acompletion",
    "embedding": "aembedding",
    "messages": "anthropic_messages",
    "responses": "aresponses",
    "speech": "aspeech",
    "transcription": "atranscription",
    "image_generation": "aimage_generation",
    "image_edit": "aimage_edit",
    "ocr": "aocr",
    "search": "asearch",
}
_IN_THE_ASYNC_FLAG_LIST: Final = frozenset({"chat", "embedding", "messages", "responses"})


def _provider(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    if request.target.startswith("/fail/"):
        return Reply(
            status=400,
            body=b'{"error":{"message":"scripted provider rejection","type":"invalid_request_error","code":null}}',
        )
    replies: Final[dict[str, Callable[[], Reply]]] = {
        "/v1/chat/completions": lambda: Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-hook-count",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                }
            ).encode()
        ),
        "/v1/embeddings": lambda: Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "model": "text-embedding-3-small",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5]}],
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                }
            ).encode()
        ),
        "/v1/messages": lambda: Reply(body=_MESSAGE_REPLY.encode()),
        "/v1/responses": lambda: Reply(body=_RESPONSE_REPLY.encode()),
        "/v1/audio/speech": lambda: Reply(body=_WAV_BYTES, content_type="audio/wav"),
        "/v1/audio/transcriptions": lambda: Reply(body=b'{"text":"hook count transcript"}'),
        "/v1/images/generations": lambda: Reply(body=_IMAGE_REPLY.encode()),
        "/v1/images/edits": lambda: Reply(body=_IMAGE_REPLY.encode()),
        "/v1/ocr": lambda: Reply(
            body=json.dumps(
                {
                    "pages": [
                        {
                            "index": 0,
                            "markdown": "page",
                            "images": [],
                            "dimensions": {"dpi": 200, "height": 1100, "width": 850},
                        }
                    ],
                    "model": "mistral-ocr-latest",
                    "usage_info": {"pages_processed": 1, "doc_size_bytes": 64},
                    "document_annotation": None,
                }
            ).encode()
        ),
        "/perplexity/search": lambda: Reply(
            body=json.dumps(
                {"results": [{"title": "hook count", "url": "https://result.invalid/a", "snippet": "hit"}]}
            ).encode()
        ),
    }
    return replies[request.target]()


@dataclass
class HookRecords:
    """Every hook record the probe reported, kept across drains so cells can read their own call id."""

    wire: Wire
    function_callbacks: bool
    seen: list[dict[str, str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def drained(self) -> tuple[dict[str, str], ...]:
        with self.lock:
            self.seen.extend(json.loads(request.body) for request in self.wire.drain())
            return tuple(self.seen)

    def for_call(self, call_id: str) -> tuple[dict[str, str], ...]:
        return tuple(record for record in self.drained() if record["litellm_call_id"] == call_id)


def _model(name: str, model: str, api_base: str, **extra: object) -> dict[str, object]:
    return {
        "model_name": name,
        "litellm_params": {"model": model, "api_base": api_base, "api_key": "synthetic-provider-key"},
        **extra,
    }


@pytest.fixture(scope="module", params=["cache", "cache_and_sync_callback"])
def hook_proxy(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[Gateway, HookRecords]]:
    tmp_path: Final = tmp_path_factory.mktemp("custom-logger-hook-counts")
    legacy_callbacks: Final = (
        {
            "success_callback": ["integration._support.hook_count_probe.sync_success_function"],
            "failure_callback": ["integration._support.hook_count_probe.sync_failure_function"],
        }
        if request.param == "cache_and_sync_callback"
        else {}
    )
    with (
        wire_server(_provider) as provider,
        wire_server(lambda request: Reply()) as sink,
        gateway_from_environment() as rig,
    ):
        models: Final = []
        tools: Final = []
        for outcome, prefix in (("success", ""), ("failure", "/fail")):
            base: str = provider.url + prefix
            models.extend(
                (
                    _model(f"hooks-chat-{outcome}", "openai/gpt-4o-mini", base + "/v1"),
                    _model(f"hooks-embedding-{outcome}", "openai/text-embedding-3-small", base + "/v1"),
                    _model(f"hooks-messages-{outcome}", "anthropic/claude-haiku-4-5", base),
                    _model(f"hooks-responses-{outcome}", "openai/gpt-4o-mini", base + "/v1"),
                    _model(f"hooks-speech-{outcome}", "openai/gpt-4o-mini-tts", base + "/v1"),
                    _model(f"hooks-transcription-{outcome}", "openai/whisper-1", base + "/v1"),
                    _model(f"hooks-image_generation-{outcome}", "openai/gpt-image-1", base + "/v1"),
                    _model(f"hooks-image_edit-{outcome}", "openai/gpt-image-1", base + "/v1"),
                    _model(
                        f"hooks-ocr-{outcome}", "mistral/mistral-ocr-latest", base + "/v1", model_info={"mode": "ocr"}
                    ),
                )
            )
            tools.append(
                {
                    "search_tool_name": f"hooks-search-{outcome}",
                    "litellm_params": {
                        "search_provider": "perplexity",
                        "api_key": "synthetic-provider-key",
                        "api_base": base + "/perplexity",
                    },
                }
            )
        config: Final = tmp_path / "hook-counts.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": models,
                    "search_tools": tools,
                    "litellm_settings": {
                        "callbacks": ["integration._support.hook_count_probe.probe"],
                        **legacy_callbacks,
                        "cache": True,
                        "cache_params": {
                            "type": "redis",
                            "host": "os.environ/REDIS_HOST",
                            "port": "os.environ/REDIS_PORT",
                        },
                    },
                    "general_settings": {
                        "master_key": "os.environ/LITELLM_MASTER_KEY",
                        "database_url": "os.environ/DATABASE_URL",
                        "disable_model_info_refresh": True,
                    },
                    "router_settings": {"num_retries": 0},
                }
            )
        )
        with owned_proxy(
            rig, tmp_path, {"HOOK_COUNT_PROBE_URL": sink.url, "STORE_MODEL_IN_DB": "False"}, config=config, workers=2
        ) as proxy:
            yield proxy, HookRecords(sink, function_callbacks=request.param == "cache_and_sync_callback")


def _send(proxy: Gateway, surface: str, outcome: str, call_id: str, key: str | None = None) -> httpx.Response:
    model: Final = f"hooks-{surface}-{outcome}"
    nonce: Final = f"hook-count-{call_id}"
    headers: Final = {"Authorization": f"Bearer {key or proxy.key}", "x-litellm-call-id": call_id}
    if surface == "transcription":
        return proxy.client.post(
            "/v1/audio/transcriptions",
            data={"model": model, "prompt": nonce},
            files={"file": ("a.wav", _WAV_BYTES + call_id.encode(), "audio/wav")},
            headers=headers,
        )
    if surface == "image_edit":
        return proxy.client.post(
            "/v1/images/edits",
            data={"model": model, "prompt": nonce},
            files={"image": ("a.png", _PNG_BYTES, "image/png")},
            headers=headers,
        )
    routes: Final[dict[str, tuple[str, dict[str, object]]]] = {
        "chat": ("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": nonce}]}),
        "embedding": ("/v1/embeddings", {"model": model, "input": nonce}),
        "messages": (
            "/v1/messages",
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": nonce}]},
        ),
        "responses": ("/v1/responses", {"model": model, "input": nonce}),
        "speech": ("/v1/audio/speech", {"model": model, "input": nonce, "voice": "alloy", "response_format": "wav"}),
        "image_generation": ("/v1/images/generations", {"model": model, "prompt": nonce, "n": 1}),
        "ocr": (
            "/v1/ocr",
            {
                "model": model,
                "document": {
                    "type": "document_url",
                    "document_url": "data:application/pdf;base64," + base64.b64encode(nonce.encode()).decode(),
                },
            },
        ),
        "search": ("/v1/search", {"search_tool_name": f"hooks-search-{outcome}", "query": nonce}),
    }
    path, body = routes[surface]
    return proxy.client.post(path, json=body, headers=headers)


def _assert_one_hook_per_request(records: HookRecords, surface: str, outcome: str, call_id: str) -> None:
    counts_function_hooks: Final = records.function_callbacks and surface not in _IN_THE_ASYNC_FLAG_LIST
    expected: Final = sorted((f"async_{outcome}", *((f"function_{outcome}",) if counts_function_hooks else ())))

    def hooks(found: tuple[dict[str, str], ...]) -> list[str]:
        return sorted(
            record["hook"] for record in found if counts_function_hooks or not record["hook"].startswith("function_")
        )

    settled: Final = held(
        lambda: records.for_call(call_id), lambda found: hooks(found) == expected, holding=3, seconds=20
    )
    assert {record["call_type"] for record in settled} == {_SURFACE_CALL_TYPES[surface]}, settled


@pytest.mark.parametrize("outcome", ["success", "failure"])
@pytest.mark.parametrize("surface", sorted(_SURFACE_CALL_TYPES))
def test_custom_logger_gets_one_async_hook_per_request_with_the_response_cache_on(
    hook_proxy: tuple[Gateway, HookRecords], surface: str, outcome: str
) -> None:
    proxy, records = hook_proxy
    call_id: Final = str(uuid.uuid4())
    response: Final = _send(proxy, surface, outcome, call_id)
    assert response.status_code == (200 if outcome == "success" else 400), response.text
    _assert_one_hook_per_request(records, surface, outcome, call_id)


@pytest.mark.parametrize("surface", sorted(_SURFACE_CALL_TYPES))
def test_custom_logger_gets_one_async_hook_per_request_rejected_by_key_auth(
    hook_proxy: tuple[Gateway, HookRecords], surface: str
) -> None:
    proxy, records = hook_proxy
    known_calls: Final = {record["litellm_call_id"] for record in records.drained()}
    response: Final = _send(proxy, surface, "success", str(uuid.uuid4()), key="sk-not-a-virtual-key")
    assert response.status_code == 401, response.text

    def new_calls(found: tuple[dict[str, str], ...]) -> set[str]:
        return {record["litellm_call_id"] for record in found if record["litellm_call_id"] not in known_calls}

    (rejected_call,) = eventually(lambda: new_calls(records.drained()), bool, seconds=20)
    _assert_one_hook_per_request(records, surface, "failure", rejected_call)
