from __future__ import annotations

import asyncio
import json
import linecache
import sys
import threading
import uuid
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from types import FrameType
from typing import Final, TypeVar

import pytest
from integration._support.client import object_value
from integration._support.openai_wire import chat_reply, openai_error, responses_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

import litellm
from litellm import Router

_DEPLOYMENTS: Final = 4
_TARGET: Final = _DEPLOYMENTS - 1
_SYNC_SECONDS: Final = 10
_RATE_LIMITED_KEY: Final = "key-rate-limited"
_BURST_THREADS: Final = 8
_BURST_CALLS: Final = 24
_CHURN_UPSERTS: Final = 120
_UNREPORTED_MODEL_ID: Final = frozenset({"aresponses-stream", "aanthropic-messages"})
_ROUTER_FILE: Final = Router.__init__.__code__.co_filename
_INDEX_LOCK_LINES: Final = frozenset(
    number
    for number, line in enumerate(linecache.getlines(_ROUTER_FILE), start=1)
    if "with self._deployment_index_lock" in line
)

T = TypeVar("T")


def _peer(request: Request) -> Reply:
    if request.headers["authorization"] == f"Bearer {_RATE_LIMITED_KEY}":
        return openai_error(429)
    body: Final = object_value(json.loads(request.body))
    stream: Final = body.get("stream") is True
    if request.target == "/v1/responses":
        return responses_reply(f"resp_{uuid.uuid4().hex}", "gpt-4o-mini", "ok", stream=stream)
    return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", "gpt-4o-mini", "ok", stream=stream)


def _deployment(
    index: int,
    api_base: str,
    team: bool,
    *,
    model: int | None = None,
    public_name: str | None = None,
    timeout: float = 10,
) -> dict[str, JsonValue]:
    team_info: Final[dict[str, JsonValue]] = (
        {"team_id": "team-1", "team_public_model_name": public_name or f"public-{index}"} if team else {}
    )
    return {
        "model_name": f"model-{index if model is None else model}",
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_base": api_base,
            "api_key": f"key-{index}",
            "timeout": timeout,
        },
        "model_info": {"id": f"deployment-{index}", **team_info},
    }


def _router(wire: Wire, team: bool = False) -> Router:
    return Router(model_list=[_deployment(index, f"{wire.url}/v1", team) for index in range(_DEPLOYMENTS)])


def _read_during_held_write(read: Callable[[], T], write: Callable[[], object]) -> T:
    """Run `read` on its own thread while `write` waits where the list has shrunk but the index maps have not caught
    up. The write resumes once `read` returns or reaches the router's index lock, so a read that skips the lock meets
    the half-updated list and a read that takes the lock waits for the write"""
    reindex: Final = Router._update_deployment_indices_after_removal.__code__
    held: Final = threading.Event()
    release: Final = threading.Event()
    settled: Final = threading.Event()
    released_in_time: Final[list[bool]] = []
    outcome: Final[list[T]] = []
    failure: Final[list[BaseException]] = []

    def hold_at_reindex(frame: FrameType, event: str, _arg: object) -> None:
        if event == "call" and frame.f_code is reindex and not held.is_set():
            held.set()
            released_in_time.append(release.wait(timeout=_SYNC_SECONDS))

    def run_write() -> None:
        previous: Final = sys.getprofile()
        sys.setprofile(hold_at_reindex)
        try:
            write()
        finally:
            sys.setprofile(previous)

    def watch_read(frame: FrameType, event: str, _arg: object) -> Callable[..., object] | None:
        if frame.f_code.co_filename != _ROUTER_FILE:
            return None
        if event == "line" and frame.f_lineno in _INDEX_LOCK_LINES:
            settled.set()
        return watch_read

    def run_read() -> None:
        previous: Final = sys.gettrace()
        sys.settrace(watch_read)
        try:
            outcome.append(read())
        except BaseException as error:
            failure.append(error)
        finally:
            sys.settrace(previous)
            settled.set()

    writer: Final = threading.Thread(target=run_write)
    reader: Final = threading.Thread(target=run_read)
    writer.start()
    try:
        assert held.wait(timeout=_SYNC_SECONDS), "the write never reached re-indexing"
        reader.start()
        assert settled.wait(timeout=_SYNC_SECONDS), "the read neither returned nor reached the index lock"
    finally:
        release.set()
        if reader.ident is not None:
            reader.join()
        writer.join()
    assert released_in_time == [True], "the write was not held for the whole read"
    if failure:
        raise failure[0]
    return outcome[0]


def _read_during_held_delete(router: Router, deployment_id: str, read: Callable[[], T]) -> T:
    return _read_during_held_write(read, lambda: router.delete_deployment(deployment_id))


def _keys(wire: Wire) -> tuple[str, ...]:
    return tuple(request.headers["authorization"] for request in wire.drain())


def _team_metadata(team: bool) -> dict[str, JsonValue]:
    return {"user_api_key_team_id": "team-1"} if team else {}


def _completion(router: Router, model: str, team: bool) -> str | None:
    response: Final = router.completion(
        model=model, messages=[{"role": "user", "content": "hi"}], metadata=_team_metadata(team)
    )
    return response._hidden_params["model_id"]


def _completion_stream(router: Router, model: str, team: bool) -> str | None:
    stream: Final = router.completion(
        model=model, messages=[{"role": "user", "content": "hi"}], metadata=_team_metadata(team), stream=True
    )
    assert isinstance(stream, Iterable), stream
    chunks: Final = list(stream)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "ok", chunks
    return stream._hidden_params["model_id"]


def _acompletion(router: Router, model: str, team: bool) -> str | None:
    async def call() -> str | None:
        response: Final = await router.acompletion(
            model=model, messages=[{"role": "user", "content": "hi"}], metadata=_team_metadata(team)
        )
        return response._hidden_params["model_id"]

    return asyncio.run(call())


def _acompletion_stream(router: Router, model: str, team: bool) -> str | None:
    async def call() -> str | None:
        stream: Final = await router.acompletion(
            model=model, messages=[{"role": "user", "content": "hi"}], metadata=_team_metadata(team), stream=True
        )
        assert isinstance(stream, AsyncIterable), stream
        chunks: Final = [chunk async for chunk in stream]
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "ok", chunks
        return stream._hidden_params["model_id"]

    return asyncio.run(call())


def _responses(router: Router, model: str, team: bool) -> str | None:
    response: Final = router.responses(model=model, input="hi", metadata=_team_metadata(team))
    assert response.output_text == "ok", response
    return response._hidden_params["model_id"]


def _aresponses(router: Router, model: str, team: bool) -> str | None:
    async def call() -> str | None:
        response: Final = await router.aresponses(model=model, input="hi", metadata=_team_metadata(team))
        assert response.output_text == "ok", response
        return response._hidden_params["model_id"]

    return asyncio.run(call())


def _aresponses_stream(router: Router, model: str, team: bool) -> str | None:
    async def call() -> None:
        stream: Final = await router.aresponses(model=model, input="hi", metadata=_team_metadata(team), stream=True)
        assert isinstance(stream, AsyncIterable), stream
        events: Final = [event.type async for event in stream]
        assert events[-1] == "response.completed", events

    asyncio.run(call())
    return None


def _aanthropic_messages(router: Router, model: str, team: bool) -> str | None:
    async def call() -> None:
        response: Final = await router.aanthropic_messages(
            model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=16, metadata=_team_metadata(team)
        )
        assert response["content"][0]["text"] == "ok", response

    asyncio.run(call())
    return None


_CALLS: Final[dict[str, Callable[[Router, str, bool], str | None]]] = {
    "completion": _completion,
    "completion-stream": _completion_stream,
    "acompletion": _acompletion,
    "acompletion-stream": _acompletion_stream,
    "responses": _responses,
    "aresponses": _aresponses,
    "aresponses-stream": _aresponses_stream,
    "aanthropic-messages": _aanthropic_messages,
}


@pytest.mark.parametrize("call", list(_CALLS))
@pytest.mark.parametrize("team", [False, True], ids=["model-name", "team-public-name"])
def test_call_reaches_its_deployment_while_another_thread_deletes_one(call: str, team: bool) -> None:
    with wire_server(_peer) as wire:
        router: Final = _router(wire, team)
        served_by: Final = _read_during_held_delete(
            router,
            "deployment-0",
            lambda: _CALLS[call](router, f"public-{_TARGET}" if team else f"model-{_TARGET}", team),
        )
        keys: Final = _keys(wire)

    assert served_by == (None if call in _UNREPORTED_MODEL_ID else f"deployment-{_TARGET}")
    assert keys == (f"Bearer key-{_TARGET}",)
    assert router.get_model_ids() == [f"deployment-{index}" for index in range(1, _DEPLOYMENTS)]


@pytest.mark.parametrize("deleted", [0, _TARGET - 1], ids=["first", "next-to-target"])
def test_completion_reaches_its_deployment_whichever_earlier_deployment_is_deleted(deleted: int) -> None:
    with wire_server(_peer) as wire:
        router: Final = _router(wire)
        served_by: Final = _read_during_held_delete(
            router, f"deployment-{deleted}", lambda: _completion(router, f"model-{_TARGET}", team=False)
        )
        keys: Final = _keys(wire)

    assert served_by == f"deployment-{_TARGET}"
    assert keys == (f"Bearer key-{_TARGET}",)


def test_team_completion_reaches_the_remaining_deployment_of_a_shared_public_name_while_its_sibling_is_deleted() -> (
    None
):
    with wire_server(_peer) as wire:
        api_base: Final = f"{wire.url}/v1"
        router: Final = Router(
            model_list=[
                _deployment(0, api_base, team=False),
                _deployment(1, api_base, team=False),
                _deployment(2, api_base, team=True, public_name="public-shared"),
                _deployment(3, api_base, team=True, public_name="public-shared"),
            ]
        )
        served_by: Final = _read_during_held_delete(
            router, "deployment-2", lambda: _completion(router, "public-shared", team=True)
        )
        keys: Final = _keys(wire)

    assert served_by == "deployment-3"
    assert keys == ("Bearer key-3",)


def test_completion_for_the_deployment_being_deleted_fails_cleanly_and_sends_nothing() -> None:
    with wire_server(_peer) as wire:
        router: Final = _router(wire)
        with pytest.raises(litellm.BadRequestError):
            _read_during_held_delete(
                router,
                "deployment-0",
                lambda: router.completion(model="model-0", messages=[{"role": "user", "content": "hi"}]),
            )
        keys: Final = _keys(wire)

    assert keys == ()
    assert router.get_model_ids() == [f"deployment-{index}" for index in range(1, _DEPLOYMENTS)]


def test_provider_error_during_a_concurrent_delete_comes_from_the_requested_deployment() -> None:
    with wire_server(_peer) as wire:
        api_base: Final = f"{wire.url}/v1"
        rate_limited: Final = _deployment(_TARGET, api_base, team=False)
        rate_limited["litellm_params"] = {**object_value(rate_limited["litellm_params"]), "api_key": _RATE_LIMITED_KEY}
        router: Final = Router(
            model_list=[*(_deployment(index, api_base, team=False) for index in range(_TARGET)), rate_limited],
            num_retries=0,
        )
        with pytest.raises(litellm.RateLimitError):
            _read_during_held_delete(
                router,
                "deployment-0",
                lambda: router.completion(model=f"model-{_TARGET}", messages=[{"role": "user", "content": "hi"}]),
            )
        served_by: Final = _completion(router, "model-1", team=False)
        keys: Final = _keys(wire)

    assert served_by == "deployment-1"
    assert keys == (f"Bearer {_RATE_LIMITED_KEY}", "Bearer key-1")


def test_two_threads_deleting_the_same_deployment_remove_it_once() -> None:
    with wire_server(_peer) as wire:
        router: Final = _router(wire)
        second: Final = _read_during_held_delete(
            router, "deployment-0", lambda: router.delete_deployment("deployment-0")
        )
        served_by: Final = _completion(router, f"model-{_TARGET}", team=False)
        keys: Final = _keys(wire)

    assert second is None
    assert served_by == f"deployment-{_TARGET}"
    assert keys == (f"Bearer key-{_TARGET}",)
    assert router.get_model_ids() == [f"deployment-{index}" for index in range(1, _DEPLOYMENTS)]


def test_completion_reaches_its_deployment_while_another_thread_upserts_an_earlier_one() -> None:
    with wire_server(_peer) as wire:
        router: Final = _router(wire)
        changed: Final = litellm.Deployment(**_deployment(0, f"{wire.url}/v1", team=False, timeout=11))
        served_by: Final = _read_during_held_write(
            lambda: _completion(router, f"model-{_TARGET}", team=False), lambda: router.upsert_deployment(changed)
        )
        keys: Final = _keys(wire)

    assert served_by == f"deployment-{_TARGET}"
    assert keys == (f"Bearer key-{_TARGET}",)
    assert sorted(router.get_model_ids()) == [f"deployment-{index}" for index in range(_DEPLOYMENTS)]


def test_deployment_added_during_a_concurrent_delete_serves_its_own_requests() -> None:
    with wire_server(_peer) as wire:
        router: Final = _router(wire)
        added: Final = litellm.Deployment(**_deployment(9, f"{wire.url}/v1", team=False))
        _read_during_held_delete(router, "deployment-0", lambda: router.add_deployment(added))
        served: Final = tuple(_completion(router, f"model-{index}", team=False) for index in (9, _TARGET))
        keys: Final = _keys(wire)

    assert served == ("deployment-9", f"deployment-{_TARGET}")
    assert keys == ("Bearer key-9", f"Bearer key-{_TARGET}")


def test_model_list_set_during_a_concurrent_delete_serves_its_own_requests() -> None:
    with wire_server(_peer) as wire:
        router: Final = _router(wire)
        replacement: Final = [_deployment(index, f"{wire.url}/v1", team=False) for index in range(10, 14)]
        _read_during_held_delete(router, "deployment-0", lambda: router.set_model_list(replacement))
        served: Final = tuple(_completion(router, f"model-{index}", team=False) for index in range(10, 14))
        keys: Final = _keys(wire)

    assert served == tuple(f"deployment-{index}" for index in range(10, 14))
    assert keys == tuple(f"Bearer key-{index}" for index in range(10, 14))
    assert router.get_model_ids() == [f"deployment-{index}" for index in range(10, 14)]


def _burst_router(api_base: str) -> Router:
    """Two deployments per model name, so an upsert of one leaves its model group served."""
    return Router(
        model_list=[
            _deployment(index, api_base, team=False, model=index % _DEPLOYMENTS) for index in range(2 * _DEPLOYMENTS)
        ]
    )


def _churn(router: Router, api_base: str, stop: threading.Event) -> int:
    """Upsert the deployments in turn with a changed timeout; each upsert moves one to the end of the list."""
    upserts = 0
    while upserts < _CHURN_UPSERTS and not stop.is_set():
        index, round_number = upserts % (2 * _DEPLOYMENTS), upserts // (2 * _DEPLOYMENTS)
        router.upsert_deployment(
            litellm.Deployment(
                **_deployment(index, api_base, team=False, model=index % _DEPLOYMENTS, timeout=11 + round_number % 2)
            )
        )
        upserts += 1
    return upserts


def _served_keys(wire: Wire) -> dict[str, str]:
    """Which key reached the upstream for each request marker."""
    return {
        object_value(json.loads(request.body))["messages"][0]["content"]: request.headers["authorization"]
        for request in wire.drain()
    }


def _assert_each_request_used_its_models_deployment(served: list[tuple[str, str | None]], keys: dict[str, str]) -> None:
    for marker, model_id in served:
        model: Final = int(marker.split(":")[0])
        assert model_id in (f"deployment-{model}", f"deployment-{model + _DEPLOYMENTS}"), (marker, model_id)
        assert keys[marker] == f"Bearer key-{str(model_id).removeprefix('deployment-')}", (marker, keys[marker])
    assert len(keys) == len(served)


def test_thread_burst_during_upsert_churn_sends_every_request_with_its_deployments_key() -> None:
    with wire_server(_peer) as wire:
        api_base: Final = f"{wire.url}/v1"
        router: Final = _burst_router(api_base)
        stop: Final = threading.Event()

        def call(number: int) -> tuple[str, str | None]:
            marker: Final = f"{number % _DEPLOYMENTS}:{number}"
            response: Final = router.completion(
                model=f"model-{number % _DEPLOYMENTS}", messages=[{"role": "user", "content": marker}]
            )
            return marker, response._hidden_params["model_id"]

        with ThreadPoolExecutor(max_workers=_BURST_THREADS + 1) as pool:
            churn: Final = pool.submit(_churn, router, api_base, stop)
            served: Final = list(pool.map(call, range(_BURST_THREADS * _BURST_CALLS)))
            stop.set()
            upserts: Final = churn.result()
        keys: Final = _served_keys(wire)

    assert upserts > 0
    _assert_each_request_used_its_models_deployment(served, keys)
    assert sorted(router.get_model_ids()) == sorted(f"deployment-{index}" for index in range(2 * _DEPLOYMENTS))


async def _gathered(calls: Iterable[Awaitable[tuple[str, str | None]]]) -> list[tuple[str, str | None]]:
    return list(await asyncio.gather(*calls))


def test_async_burst_during_upsert_churn_sends_every_request_with_its_deployments_key() -> None:
    with wire_server(_peer) as wire:
        api_base: Final = f"{wire.url}/v1"
        router: Final = _burst_router(api_base)
        stop: Final = threading.Event()

        async def call(number: int) -> tuple[str, str | None]:
            marker: Final = f"{number % _DEPLOYMENTS}:{number}"
            response: Final = await router.acompletion(
                model=f"model-{number % _DEPLOYMENTS}", messages=[{"role": "user", "content": marker}]
            )
            return marker, response._hidden_params["model_id"]

        with ThreadPoolExecutor(max_workers=1) as pool:
            churn: Final = pool.submit(_churn, router, api_base, stop)
            served: Final = asyncio.run(_gathered(call(number) for number in range(_BURST_THREADS * _BURST_CALLS)))
            stop.set()
            upserts: Final = churn.result()
        keys: Final = _served_keys(wire)

    assert upserts > 0
    _assert_each_request_used_its_models_deployment(served, keys)
    assert sorted(router.get_model_ids()) == sorted(f"deployment-{index}" for index in range(2 * _DEPLOYMENTS))
