"""A CustomLogger with both hooks still gets one hook per request when the async success logging does not reach it.

When an async-only logger listed before it stalls past the logging worker's per-handler timeout, the worker cuts
the async success handler off before it reaches the logger, and the logger gets its sync success hook instead.

When a post-call guardrail blocks the response, the proxy drops the async success logging it parked for the
guardrails and logs the block as a failure. The logger gets one failure hook, plus, on the surfaces outside the
async flag list, the sync success hook it got before the sync run held it back for the dropped async success. Speech
and transcription responses never reach post-call guardrails, and image generation logs its success before they run,
so neither parks.

A legacy sync callback is configured, because that is what makes the sync handler run for an async call."""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment, held
from integration._support.hook_counts import (
    IN_THE_ASYNC_FLAG_LIST,
    SURFACE_CALL_TYPES,
    HookRecords,
    assert_hooks,
    probe_hooks,
    provider,
    send,
    write_hook_count_config,
)
from integration._support.process import owned_proxy
from integration._support.wire import Reply, wire_server

pytestmark = pytest.mark.timeout(240)

_PROBES: Final = "integration._support.hook_count_probe"
_WORKER_TIMEOUT_SECONDS: Final = 2
_OUTSIDE_THE_ASYNC_FLAG_LIST: Final = sorted(SURFACE_CALL_TYPES.keys() - IN_THE_ASYNC_FLAG_LIST)
_PARKED_FOR_POST_CALL_GUARDRAILS: Final = sorted(
    SURFACE_CALL_TYPES.keys() - {"speech", "transcription", "image_generation"}
)


@dataclass(frozen=True)
class _Rig:
    proxy: Gateway
    records: HookRecords


def _rig(
    tmp_path_factory: pytest.TempPathFactory,
    name: str,
    callbacks: list[str],
    guardrails: tuple[dict[str, object], ...],
    environment: dict[str, str],
) -> Iterator[_Rig]:
    tmp_path: Final = tmp_path_factory.mktemp(name)
    with (
        wire_server(provider) as upstream,
        wire_server(lambda request: Reply()) as sink,
        gateway_from_environment() as rig,
    ):
        config: Final = write_hook_count_config(
            tmp_path / f"{name}.yaml",
            upstream.url,
            {"callbacks": callbacks, "success_callback": [f"{_PROBES}.sync_success_function"]},
            guardrails,
        )
        overrides: Final = {"HOOK_COUNT_PROBE_URL": sink.url, "STORE_MODEL_IN_DB": "False", **environment}
        with owned_proxy(rig, tmp_path, overrides, config=config, workers=2) as proxy:
            yield _Rig(proxy, HookRecords(sink, function_callbacks=False))


@pytest.fixture(scope="module")
def stalled_sibling(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    yield from _rig(
        tmp_path_factory,
        "hook-count-stalled-sibling",
        [f"{_PROBES}.stalled_logger", f"{_PROBES}.probe"],
        (),
        {"LOGGING_WORKER_MAX_TIME_PER_COROUTINE": str(_WORKER_TIMEOUT_SECONDS)},
    )


@pytest.fixture(scope="module")
def post_call_block(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    guardrail: Final = {
        "guardrail_name": "hook-count-post-call-block",
        "litellm_params": {
            "guardrail": f"{_PROBES}.BlockingPostCallGuardrail",
            "mode": "post_call",
            "default_on": True,
        },
    }
    yield from _rig(tmp_path_factory, "hook-count-post-call-block", [f"{_PROBES}.probe"], (guardrail,), {})


@pytest.mark.parametrize("surface", _OUTSIDE_THE_ASYNC_FLAG_LIST)
def test_logger_the_async_success_never_reaches_behind_a_stalled_logger_gets_its_sync_hook(
    stalled_sibling: _Rig, surface: str
) -> None:
    call_id: Final = str(uuid.uuid4())
    response: Final = send(stalled_sibling.proxy, surface, "success", call_id)
    assert response.status_code == 200, response.text
    assert_hooks(stalled_sibling.records, surface, call_id, ["sync_success"])


@pytest.mark.parametrize("surface", _PARKED_FOR_POST_CALL_GUARDRAILS)
def test_response_a_post_call_guardrail_blocks_gives_the_logger_one_failure_hook(
    post_call_block: _Rig, surface: str
) -> None:
    call_id: Final = str(uuid.uuid4())
    expected: Final = ["async_failure"] if surface in IN_THE_ASYNC_FLAG_LIST else ["async_failure", "sync_success"]
    response: Final = send(post_call_block.proxy, surface, "success", call_id)
    assert response.status_code == 400, response.text
    held(
        lambda: probe_hooks(post_call_block.records, surface, post_call_block.records.for_call(call_id)),
        lambda hooks: hooks == expected,
        holding=3,
        seconds=20,
    )
