from __future__ import annotations

import dataclasses
import threading
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpx
import pytest
from integration._support.client import JSON_OBJECT, gateway_from_environment
from integration._support.openai_wire import chat_reply
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import owned_config
from integration.routing.test_payment_required_mapping import OPENAI_MODEL, is_model_info_probe, model_list_reply

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

GROUPS: Final = ("otpm-fraction-first", "otpm-fraction-retry")
OTPM_LIMIT: Final = 1000
FRACTION: Final = 0.5
DEFAULT_RESERVATION: Final = int(OTPM_LIMIT * FRACTION)


@dataclass(frozen=True, slots=True)
class Rig:
    owned: OwnedProxy
    upstream: Wire
    gates: SimpleQueue[threading.Event]


@dataclass(frozen=True, slots=True)
class Admissions:
    held_forwarded_max_tokens: bool
    statuses: tuple[int, int, int]
    one_minute: bool


def _reply_for(gates: SimpleQueue[threading.Event]) -> Callable[[Request], Reply]:
    def reply(request: Request) -> Reply:
        if is_model_info_probe(request):
            return model_list_reply()
        if JSON_OBJECT.validate_json(request.body).get("stream") is True:
            return dataclasses.replace(
                chat_reply("chatcmpl-held", OPENAI_MODEL, "held", stream=True), gate_before_first=gates.get(timeout=60)
            )
        return chat_reply("chatcmpl-capped", OPENAI_MODEL, "capped", stream=False)

    return reply


def _config(directory: Path, upstream: Wire) -> Path:
    return owned_config(
        directory,
        [
            {
                "model_name": group,
                "litellm_params": {
                    "model": OPENAI_MODEL,
                    "api_base": upstream.url + "/v1",
                    "api_key": "synthetic-openai-key",
                    "otpm": OTPM_LIMIT,
                },
                "model_info": {"id": group},
            }
            for group in GROUPS
        ],
        litellm_settings={"cache": False},
        router_settings={"optional_pre_call_checks": ["enforce_model_rate_limits"], "num_retries": 0},
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("otpm-fraction")
    gates: Final[SimpleQueue[threading.Event]] = SimpleQueue()
    with (
        gateway_from_environment() as gateway,
        wire_server(_reply_for(gates)) as upstream,
        owned_proxy_process(
            gateway,
            directory,
            {"NO_MAX_TOKENS_TPM_RESERVATION_FRACTION": str(FRACTION)},
            config=_config(directory, upstream),
        ) as owned,
    ):
        yield Rig(owned, upstream, gates)


def _chat(rig: Rig, group: str, **params: int | bool) -> httpx.Response:
    return rig.owned.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": group, "messages": [{"role": "user", "content": f"fraction {uuid.uuid4().hex}"}], **params},
    )


def _await_forwarded_chat(upstream: Wire) -> Request:
    request: Final = upstream.received.get(timeout=60)
    return _await_forwarded_chat(upstream) if is_model_info_probe(request) else request


def _utc_minute() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M")


def _admissions_beside_a_held_stream(rig: Rig, group: str) -> Admissions:
    release: Final = threading.Event()
    rig.gates.put(release)
    started: Final = _utc_minute()
    with ThreadPoolExecutor(max_workers=1) as pool:
        held: Final = pool.submit(_chat, rig, group, stream=True)
        try:
            forwarded: Final = JSON_OBJECT.validate_json(_await_forwarded_chat(rig.upstream).body)
            over: Final = _chat(rig, group, max_tokens=OTPM_LIMIT - DEFAULT_RESERVATION + 1)
            fits: Final = _chat(rig, group, max_tokens=OTPM_LIMIT - DEFAULT_RESERVATION)
            ended: Final = _utc_minute()
        finally:
            release.set()
        return Admissions(
            held_forwarded_max_tokens="max_tokens" in forwarded,
            statuses=(over.status_code, fits.status_code, held.result(timeout=60).status_code),
            one_minute=started == ended,
        )


def test_reservation_fraction_env_var_sets_what_a_request_without_max_tokens_holds(rig: Rig) -> None:
    first: Final = _admissions_beside_a_held_stream(rig, GROUPS[0])
    admissions: Final = first if first.one_minute else _admissions_beside_a_held_stream(rig, GROUPS[1])

    assert admissions == Admissions(held_forwarded_max_tokens=False, statuses=(429, 200, 200), one_minute=True)
