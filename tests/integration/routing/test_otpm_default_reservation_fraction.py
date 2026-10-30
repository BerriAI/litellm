from __future__ import annotations

import dataclasses
import threading
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
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

GROUP: Final = "otpm-fraction-group"
OTPM_LIMIT: Final = 1000
FRACTION: Final = 0.5
DEFAULT_RESERVATION: Final = int(OTPM_LIMIT * FRACTION)


@dataclass(frozen=True, slots=True)
class Rig:
    owned: OwnedProxy
    upstream: Wire
    release: threading.Event


def _reply_for(release: threading.Event) -> Callable[[Request], Reply]:
    def reply(request: Request) -> Reply:
        if is_model_info_probe(request):
            return model_list_reply()
        if JSON_OBJECT.validate_json(request.body).get("stream") is True:
            return dataclasses.replace(
                chat_reply("chatcmpl-held", OPENAI_MODEL, "held", stream=True), gate_before_first=release
            )
        return chat_reply("chatcmpl-capped", OPENAI_MODEL, "capped", stream=False)

    return reply


def _config(directory: Path, upstream: Wire) -> Path:
    return owned_config(
        directory,
        [
            {
                "model_name": GROUP,
                "litellm_params": {
                    "model": OPENAI_MODEL,
                    "api_base": upstream.url + "/v1",
                    "api_key": "synthetic-openai-key",
                    "otpm": OTPM_LIMIT,
                },
                "model_info": {"id": "otpm-fraction"},
            }
        ],
        litellm_settings={"cache": False},
        router_settings={"optional_pre_call_checks": ["enforce_model_rate_limits"], "num_retries": 0},
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("otpm-fraction")
    release: Final = threading.Event()
    with (
        gateway_from_environment() as gateway,
        wire_server(_reply_for(release)) as upstream,
        owned_proxy_process(
            gateway,
            directory,
            {"NO_MAX_TOKENS_TPM_RESERVATION_FRACTION": str(FRACTION)},
            config=_config(directory, upstream),
        ) as owned,
    ):
        try:
            yield Rig(owned, upstream, release)
        finally:
            release.set()


def _chat(rig: Rig, **params: int | bool) -> httpx.Response:
    return rig.owned.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": GROUP, "messages": [{"role": "user", "content": f"fraction {uuid.uuid4().hex}"}], **params},
    )


def _await_forwarded_chat(upstream: Wire) -> Request:
    request: Final = upstream.received.get(timeout=60)
    return _await_forwarded_chat(upstream) if is_model_info_probe(request) else request


def test_reservation_fraction_env_var_sets_what_a_request_without_max_tokens_holds(rig: Rig) -> None:
    with ThreadPoolExecutor(max_workers=1) as pool:
        held: Final = pool.submit(_chat, rig, stream=True)
        try:
            assert "max_tokens" not in JSON_OBJECT.validate_json(_await_forwarded_chat(rig.upstream).body)
            over: Final = _chat(rig, max_tokens=OTPM_LIMIT - DEFAULT_RESERVATION + 1)
            fits: Final = _chat(rig, max_tokens=OTPM_LIMIT - DEFAULT_RESERVATION)
        finally:
            rig.release.set()

        assert (over.status_code, fits.status_code) == (429, 200), (over.text, fits.text)
        assert held.result(timeout=60).status_code == 200
