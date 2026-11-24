"""A logger the async handler never reaches keeps its sync hook on async calls outside the async flag list.

A logger with both hooks registered only under `success_callback` and `failure_callback` is never reached by the async
failure handler, so it keeps its sync failure hook. The async success handler does reach it, and gives it its async
success hook instead of the sync one.

The sync success handler only runs for an async call when a legacy sync callback is configured, so every
registration here carries the legacy function callbacks too."""

import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.hook_counts import (
    IN_THE_ASYNC_FLAG_LIST,
    SURFACE_CALL_TYPES,
    HookRecords,
    assert_hooks,
    provider,
    send,
    write_hook_count_config,
)
from integration._support.process import owned_proxy
from integration._support.wire import Reply, wire_server

pytestmark = pytest.mark.timeout(240)

_PROBES: Final = "integration._support.hook_count_probe"
_REGISTRATIONS: Final[dict[str, dict[str, object]]] = {
    "sync_only_logger": {
        "callbacks": [f"{_PROBES}.sync_only_probe"],
        "success_callback": [f"{_PROBES}.sync_success_function"],
        "failure_callback": [f"{_PROBES}.sync_failure_function"],
    },
    "sync_event_lists": {
        "success_callback": [f"{_PROBES}.probe", f"{_PROBES}.sync_success_function"],
        "failure_callback": [f"{_PROBES}.probe", f"{_PROBES}.sync_failure_function"],
    },
}
_OUTSIDE_THE_ASYNC_FLAG_LIST: Final = sorted(SURFACE_CALL_TYPES.keys() - IN_THE_ASYNC_FLAG_LIST)


@pytest.fixture(scope="module", params=sorted(_REGISTRATIONS))
def sync_hook_proxy(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[Gateway, HookRecords, str]]:
    tmp_path: Final = tmp_path_factory.mktemp("custom-logger-hook-count-modes")
    with (
        wire_server(provider) as upstream,
        wire_server(lambda request: Reply()) as sink,
        gateway_from_environment() as rig,
    ):
        config: Final = write_hook_count_config(
            tmp_path / "hook-count-modes.yaml", upstream.url, _REGISTRATIONS[request.param]
        )
        with owned_proxy(
            rig, tmp_path, {"HOOK_COUNT_PROBE_URL": sink.url, "STORE_MODEL_IN_DB": "False"}, config=config, workers=2
        ) as proxy:
            yield proxy, HookRecords(sink, function_callbacks=True), request.param


@pytest.mark.parametrize("outcome", ["success", "failure"])
@pytest.mark.parametrize("surface", _OUTSIDE_THE_ASYNC_FLAG_LIST)
def test_logger_the_async_handler_skips_gets_one_sync_hook_per_request(
    sync_hook_proxy: tuple[Gateway, HookRecords, str], surface: str, outcome: str
) -> None:
    proxy, records, registration = sync_hook_proxy
    reached_by_the_async_success: Final = registration == "sync_event_lists" and outcome == "success"
    call_id: Final = str(uuid.uuid4())
    response: Final = send(proxy, surface, outcome, call_id)
    assert response.status_code == (200 if outcome == "success" else 400), response.text
    assert_hooks(
        records,
        surface,
        call_id,
        sorted([f"function_{outcome}", f"async_{outcome}" if reached_by_the_async_success else f"sync_{outcome}"]),
    )


@pytest.mark.parametrize("surface", _OUTSIDE_THE_ASYNC_FLAG_LIST)
def test_logger_the_async_handler_skips_gets_one_sync_hook_per_request_rejected_by_key_auth(
    sync_hook_proxy: tuple[Gateway, HookRecords, str], surface: str
) -> None:
    proxy, records, _ = sync_hook_proxy
    known_calls: Final = {record["litellm_call_id"] for record in records.drained()}
    response: Final = send(proxy, surface, "success", str(uuid.uuid4()), key="sk-not-a-virtual-key")
    assert response.status_code == 401, response.text

    def new_calls(found: tuple[dict[str, str], ...]) -> set[str]:
        return {record["litellm_call_id"] for record in found if record["litellm_call_id"] not in known_calls}

    (rejected_call,) = eventually(lambda: new_calls(records.drained()), bool, seconds=20)
    assert_hooks(records, surface, rejected_call, ["function_failure", "sync_failure"])
