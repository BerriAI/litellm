from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final

import litellm
import pytest
from integration._support.wire import Reply, Request, Wire, wire_server
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger

_MODEL: Final = "gpt-5.6"
_API_KEY: Final = "synthetic-sync-fallback-key"
_PROMPT: Final = "which deployment answers when the primary dies before its first chunk?"
_ERROR_FRAME: Final = (
    b"data: " + json.dumps({"error": {"message": "overloaded", "type": "server_error", "code": 500}}).encode() + b"\n\n"
)
_DONE: Final = b"data: [DONE]\n\n"
_BURST: Final = 6


def _delta(text: str, finish_reason: str | None) -> bytes:
    chunk: Final = {
        "id": "chatcmpl-sync-fallback-wire",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": _MODEL,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


def _serves(text: str) -> Reply:
    return Reply(content_type="text/event-stream", chunks=(_delta(text, None), _delta("", "stop"), _DONE))


def _dies_after(text: str) -> Reply:
    return Reply(content_type="text/event-stream", chunks=(_delta(text, None), _ERROR_FRAME, _DONE))


_DIES_BEFORE_CONTENT: Final = Reply(content_type="text/event-stream", chunks=(_ERROR_FRAME, _DONE))
_DROPS_BEFORE_CONTENT: Final = Reply(content_type="text/event-stream", chunks=(_DONE,), abort_after=0)


def _peer(replies: Mapping[str, Reply]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST", request.method
        deployment, _, route = request.target.lstrip("/").partition("/")
        assert route == "chat/completions", request.target
        return replies[deployment]

    return respond


def _deployments_hit(wire: Wire) -> tuple[str, ...]:
    return tuple(request.target.lstrip("/").partition("/")[0] for request in wire.drain())


def _router(wire: Wire, deployments: tuple[str, ...], **settings: object) -> Router:
    return Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {"model": f"openai/{_MODEL}", "api_base": f"{wire.url}/{name}", "api_key": _API_KEY},
            }
            for name in deployments
        ],
        num_retries=0,
        disable_cooldowns=True,
        **settings,
    )


class _FallbackRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.successes: tuple[str, ...] = ()
        self.failures: tuple[str, ...] = ()

    async def log_success_fallback_event(
        self, original_model_group: str, kwargs: dict, original_exception: Exception
    ) -> None:
        self.successes = (*self.successes, original_model_group)

    async def log_failure_fallback_event(
        self, original_model_group: str, kwargs: dict, original_exception: Exception
    ) -> None:
        self.failures = (*self.failures, original_model_group)


@dataclass(frozen=True, slots=True)
class _Streamed:
    text: str
    attempted_fallbacks: object


def _text_of(chunk: object) -> str:
    choices: Final = getattr(chunk, "choices", None) or ()
    return "".join(str(choice.delta.content or "") for choice in choices)


def _attempted_fallbacks(stream: object) -> object:
    hidden: Final = getattr(stream, "_hidden_params", None) or {}
    return (hidden.get("additional_headers") or {}).get("x-litellm-attempted-fallbacks")


def _stream_sync(router: Router, **request: object) -> _Streamed:
    stream: Final = router.completion(model="primary", messages=[{"role": "user", "content": _PROMPT}], stream=True, **request)
    text: Final = "".join(_text_of(chunk) for chunk in stream)
    return _Streamed(text=text, attempted_fallbacks=_attempted_fallbacks(stream))


async def _stream_async(router: Router, **request: object) -> _Streamed:
    stream: Final = await router.acompletion(
        model="primary", messages=[{"role": "user", "content": _PROMPT}], stream=True, **request
    )
    parts: Final = [_text_of(chunk) async for chunk in stream]
    return _Streamed(text="".join(parts), attempted_fallbacks=_attempted_fallbacks(stream))


def _stream(client: str, router: Router, **request: object) -> _Streamed:
    if client == "async":
        return asyncio.run(_stream_async(router, **request))
    return _stream_sync(router, **request)


_CLIENTS: Final = ("sync", "async")
_PRIMARY_DIES: Final = {"primary": _DIES_BEFORE_CONTENT, "backup": _serves("answered by the backup")}
_PRIMARY_AND_FB1_DIE: Final = {"primary": _DIES_BEFORE_CONTENT, "fb1": _DIES_BEFORE_CONTENT, "fb2": _serves("answered by fb2")}
_PRIMARY_TO_BACKUP: Final = [{"primary": ["backup"]}]


@pytest.mark.parametrize("client", _CLIENTS)
def test_primary_dies_before_content(client: str, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: Final = _FallbackRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    with wire_server(_peer(_PRIMARY_DIES)) as wire:
        router: Final = _router(wire, ("primary", "backup"), fallbacks=_PRIMARY_TO_BACKUP)
        streamed: Final = _stream(client, router)
        assert streamed.text == "answered by the backup", streamed
        assert streamed.attempted_fallbacks == 1, streamed
        assert _deployments_hit(wire) == ("primary", "backup")
    assert recorder.successes == ("primary",), recorder.successes
    assert recorder.failures == (), recorder.failures


@pytest.mark.parametrize("client", _CLIENTS)
def test_walks_every_configured_fallback(client: str) -> None:
    with wire_server(_peer(_PRIMARY_AND_FB1_DIE)) as wire:
        router: Final = _router(wire, ("primary", "fb1", "fb2"), fallbacks=[{"primary": ["fb1", "fb2"]}])
        streamed: Final = _stream(client, router)
        assert streamed.text == "answered by fb2", streamed
        assert streamed.attempted_fallbacks == 2, streamed
        assert _deployments_hit(wire) == ("primary", "fb1", "fb2")


@pytest.mark.parametrize("client", _CLIENTS)
def test_every_target_dies(client: str) -> None:
    with wire_server(_peer({"primary": _DIES_BEFORE_CONTENT, "backup": _DIES_BEFORE_CONTENT})) as wire:
        router: Final = _router(wire, ("primary", "backup"), fallbacks=_PRIMARY_TO_BACKUP)
        with pytest.raises(litellm.APIConnectionError, match="overloaded"):
            _stream(client, router)
        assert _deployments_hit(wire) == ("primary", "backup")


@pytest.mark.parametrize("client", _CLIENTS)
def test_fallbacks_disabled(client: str) -> None:
    with wire_server(_peer(_PRIMARY_DIES)) as wire:
        router: Final = _router(wire, ("primary", "backup"), fallbacks=_PRIMARY_TO_BACKUP)
        with pytest.raises(litellm.APIConnectionError, match="overloaded"):
            _stream(client, router, disable_fallbacks=True)
        assert _deployments_hit(wire) == ("primary",)


@pytest.mark.parametrize("client", _CLIENTS)
def test_dies_after_first_chunk(client: str) -> None:
    with wire_server(_peer({"primary": _dies_after("partial "), "backup": _serves("never asked")})) as wire:
        router: Final = _router(wire, ("primary", "backup"), fallbacks=_PRIMARY_TO_BACKUP)
        with pytest.raises(litellm.APIConnectionError, match="overloaded"):
            _stream(client, router)
        assert _deployments_hit(wire) == ("primary",)


@pytest.mark.parametrize("client", _CLIENTS)
def test_router_retries_configured(client: str) -> None:
    with wire_server(_peer(_PRIMARY_DIES)) as wire:
        router: Final = Router(
            model_list=_router(wire, ("primary", "backup")).model_list,
            fallbacks=_PRIMARY_TO_BACKUP,
            num_retries=2,
            disable_cooldowns=True,
        )
        streamed: Final = _stream(client, router)
        assert streamed.text == "answered by the backup", streamed
        assert _deployments_hit(wire) == ("primary", "backup")


@pytest.mark.parametrize("client", _CLIENTS)
def test_per_request_fallback_list(client: str) -> None:
    with wire_server(_peer(_PRIMARY_DIES)) as wire:
        router: Final = _router(wire, ("primary", "backup"))
        streamed: Final = _stream(client, router, fallbacks=_PRIMARY_TO_BACKUP)
        assert streamed.text == "answered by the backup", streamed
        assert streamed.attempted_fallbacks == 1, streamed
        assert _deployments_hit(wire) == ("primary", "backup")


@pytest.mark.parametrize("client", _CLIENTS)
def test_per_request_fallbacks_none_turns_the_router_list_off(client: str) -> None:
    with wire_server(_peer(_PRIMARY_DIES)) as wire:
        router: Final = _router(wire, ("primary", "backup"), fallbacks=_PRIMARY_TO_BACKUP)
        with pytest.raises(litellm.APIConnectionError, match="overloaded"):
            _stream(client, router, fallbacks=None)
        assert _deployments_hit(wire) == ("primary",)


@pytest.mark.parametrize("client", _CLIENTS)
def test_max_fallbacks_caps_the_walk(client: str) -> None:
    with wire_server(_peer(_PRIMARY_AND_FB1_DIE)) as wire:
        router: Final = _router(wire, ("primary", "fb1", "fb2"), fallbacks=[{"primary": ["fb1", "fb2"]}], max_fallbacks=1)
        with pytest.raises(litellm.APIConnectionError, match="overloaded"):
            _stream(client, router)
        assert _deployments_hit(wire) == ("primary", "fb1")


def test_called_inside_a_running_loop() -> None:
    with wire_server(_peer(_PRIMARY_DIES)) as wire:
        router: Final = _router(wire, ("primary", "backup"), fallbacks=_PRIMARY_TO_BACKUP)

        async def inside_a_loop() -> _Streamed:
            return _stream_sync(router)

        streamed: Final = asyncio.run(inside_a_loop())
        assert streamed.text == "answered by the backup", streamed
        assert _deployments_hit(wire) == ("primary", "backup")


@pytest.mark.parametrize("client", _CLIENTS)
def test_concurrent_burst(client: str) -> None:
    with wire_server(_peer(_PRIMARY_DIES)) as wire:
        router: Final = _router(wire, ("primary", "backup"), fallbacks=_PRIMARY_TO_BACKUP)
        if client == "async":

            async def burst() -> tuple[_Streamed, ...]:
                return tuple(await asyncio.gather(*(_stream_async(router) for _ in range(_BURST))))

            streamed: tuple[_Streamed, ...] = asyncio.run(burst())
        else:
            with ThreadPoolExecutor(max_workers=_BURST) as pool:
                streamed = tuple(pool.map(lambda _: _stream_sync(router), range(_BURST)))
        assert [item.text for item in streamed] == ["answered by the backup"] * _BURST, streamed
        hit: Final = _deployments_hit(wire)
        assert (hit.count("primary"), hit.count("backup"), len(hit)) == (_BURST, _BURST, 2 * _BURST), hit


@pytest.mark.parametrize("client", _CLIENTS)
def test_primary_drops_the_connection_before_content(client: str) -> None:
    with wire_server(_peer({"primary": _DROPS_BEFORE_CONTENT, "backup": _serves("answered by the backup")})) as wire:
        router: Final = _router(wire, ("primary", "backup"), fallbacks=_PRIMARY_TO_BACKUP)
        streamed: Final = _stream(client, router)
        assert streamed.text == "answered by the backup", streamed
        assert _deployments_hit(wire) == ("primary", "backup")
