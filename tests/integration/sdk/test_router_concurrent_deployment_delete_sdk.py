from __future__ import annotations

import sys
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType
from typing import Final

import pytest
from integration._support.openai_wire import chat_reply
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

from litellm import Router

_DEPLOYMENTS: Final = 4
_TARGET: Final = _DEPLOYMENTS - 1
_HOLD_SECONDS: Final = 1.0


def _peer(_request: Request) -> Reply:
    return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", "gpt-4o-mini", "ok", stream=False)


def _deployment(index: int, api_base: str, team: bool) -> dict[str, JsonValue]:
    team_info: Final[dict[str, JsonValue]] = (
        {"team_id": "team-1", "team_public_model_name": f"public-{index}"} if team else {}
    )
    return {
        "model_name": f"model-{index}",
        "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": api_base, "api_key": f"key-{index}"},
        "model_info": {"id": f"deployment-{index}", **team_info},
    }


@contextmanager
def _delete_held_before_reindex(router: Router, deployment_id: str) -> Iterator[None]:
    reindex: Final = Router._update_deployment_indices_after_removal.__code__
    held: Final = threading.Event()
    release: Final = threading.Event()

    def hold_at_reindex(frame: FrameType, event: str, _arg: object) -> None:
        if event == "call" and frame.f_code is reindex:
            held.set()
            release.wait(timeout=_HOLD_SECONDS)

    def delete() -> None:
        previous: Final = sys.getprofile()
        sys.setprofile(hold_at_reindex)
        try:
            router.delete_deployment(deployment_id)
        finally:
            sys.setprofile(previous)

    writer: Final = threading.Thread(target=delete)
    writer.start()
    try:
        assert held.wait(timeout=10), "delete_deployment never reached re-indexing"
        yield
    finally:
        release.set()
        writer.join()


@pytest.mark.parametrize("team", [False, True], ids=["model-name", "team-public-name"])
def test_completion_reaches_its_deployment_while_another_thread_deletes_one(team: bool) -> None:
    with wire_server(_peer) as wire:
        api_base: Final = f"{wire.url}/v1"
        router: Final = Router(model_list=[_deployment(index, api_base, team) for index in range(_DEPLOYMENTS)])
        with _delete_held_before_reindex(router, "deployment-0"):
            response: Final = router.completion(
                model=f"public-{_TARGET}" if team else f"model-{_TARGET}",
                messages=[{"role": "user", "content": "hi"}],
                metadata={"user_api_key_team_id": "team-1"} if team else {},
            )
        keys: Final = tuple(request.headers["authorization"] for request in wire.drain())

    assert response._hidden_params["model_id"] == f"deployment-{_TARGET}"
    assert keys == (f"Bearer key-{_TARGET}",)
    assert router.get_model_ids() == [f"deployment-{index}" for index in range(1, _DEPLOYMENTS)]
