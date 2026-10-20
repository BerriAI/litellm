"""A CustomLogger with both hooks gets one failure hook when a post-call guardrail blocks the response.

The proxy drops the async success logging it parked for the guardrails and logs the block as a failure through both
the async and the sync failure handlers. The logger gets only its async failure hook from them. Speech and
transcription responses never reach post-call guardrails, and image generation logs its success before they run, so
neither parks.

A legacy sync callback is configured, because that is what makes the sync handler run for an async call."""

import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment, held
from integration._support.hook_counts import (
    SURFACE_CALL_TYPES,
    HookRecords,
    probe_hooks,
    provider,
    send,
    write_hook_count_config,
)
from integration._support.process import owned_proxy
from integration._support.wire import Reply, wire_server

pytestmark = pytest.mark.timeout(240)

_PROBES: Final = "integration._support.hook_count_probe"
_PARKED_FOR_POST_CALL_GUARDRAILS: Final = sorted(
    SURFACE_CALL_TYPES.keys() - {"speech", "transcription", "image_generation"}
)


@pytest.fixture(scope="module")
def post_call_block(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, HookRecords]]:
    tmp_path: Final = tmp_path_factory.mktemp("hook-count-post-call-block")
    guardrail: Final = {
        "guardrail_name": "hook-count-post-call-block",
        "litellm_params": {
            "guardrail": f"{_PROBES}.BlockingPostCallGuardrail",
            "mode": "post_call",
            "default_on": True,
        },
    }
    with (
        wire_server(provider) as upstream,
        wire_server(lambda request: Reply()) as sink,
        gateway_from_environment() as rig,
    ):
        config: Final = write_hook_count_config(
            tmp_path / "hook-count-post-call-block.yaml",
            upstream.url,
            {"callbacks": [f"{_PROBES}.probe"], "success_callback": [f"{_PROBES}.sync_success_function"]},
            (guardrail,),
        )
        overrides: Final = {"HOOK_COUNT_PROBE_URL": sink.url, "STORE_MODEL_IN_DB": "False"}
        with owned_proxy(rig, tmp_path, overrides, config=config, workers=2) as proxy:
            yield proxy, HookRecords(sink, function_callbacks=False)


@pytest.mark.parametrize("surface", _PARKED_FOR_POST_CALL_GUARDRAILS)
def test_response_a_post_call_guardrail_blocks_gives_the_logger_one_failure_hook(
    post_call_block: tuple[Gateway, HookRecords], surface: str
) -> None:
    proxy, records = post_call_block
    call_id: Final = str(uuid.uuid4())
    response: Final = send(proxy, surface, "success", call_id)
    assert response.status_code == 400, response.text
    held(
        lambda: probe_hooks(records, surface, records.for_call(call_id)),
        lambda hooks: hooks == ["async_failure"],
        holding=3,
        seconds=20,
    )
