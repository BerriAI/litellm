import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.hook_counts import (
    STREAMING_SURFACES,
    SURFACE_CALL_TYPES,
    HookRecords,
    assert_hooks,
    one_async_hook,
    provider,
    send,
    write_hook_count_config,
)
from integration._support.process import owned_proxy
from integration._support.wire import Reply, wire_server

pytestmark = pytest.mark.timeout(240)


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
        wire_server(provider) as upstream,
        wire_server(lambda request: Reply()) as sink,
        gateway_from_environment() as rig,
    ):
        config: Final = write_hook_count_config(
            tmp_path / "hook-counts.yaml",
            upstream.url,
            {
                "callbacks": ["integration._support.hook_count_probe.probe"],
                **legacy_callbacks,
                "cache": True,
                "cache_params": {"type": "redis", "host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
            },
        )
        with owned_proxy(
            rig, tmp_path, {"HOOK_COUNT_PROBE_URL": sink.url, "STORE_MODEL_IN_DB": "False"}, config=config, workers=2
        ) as proxy:
            yield proxy, HookRecords(sink, function_callbacks=request.param == "cache_and_sync_callback")


@pytest.mark.parametrize("outcome", ["success", "failure"])
@pytest.mark.parametrize("surface", sorted(SURFACE_CALL_TYPES))
def test_custom_logger_gets_one_async_hook_per_request_with_the_response_cache_on(
    hook_proxy: tuple[Gateway, HookRecords], surface: str, outcome: str
) -> None:
    proxy, records = hook_proxy
    call_id: Final = str(uuid.uuid4())
    response: Final = send(proxy, surface, outcome, call_id)
    assert response.status_code == (200 if outcome == "success" else 400), response.text
    assert_hooks(records, surface, call_id, one_async_hook(records, surface, outcome))


@pytest.mark.parametrize("outcome", ["success", "failure"])
@pytest.mark.parametrize("surface", STREAMING_SURFACES)
def test_custom_logger_gets_one_async_hook_per_streamed_request(
    hook_proxy: tuple[Gateway, HookRecords], surface: str, outcome: str
) -> None:
    proxy, records = hook_proxy
    call_id: Final = str(uuid.uuid4())
    response: Final = send(proxy, surface, outcome, call_id, stream=True)
    assert response.status_code == (200 if outcome == "success" else 400), response.text
    if outcome == "success":
        assert "ok" in response.text, response.text
    assert_hooks(records, surface, call_id, one_async_hook(records, surface, outcome))


@pytest.mark.parametrize("surface", sorted(SURFACE_CALL_TYPES))
def test_custom_logger_gets_one_async_hook_per_request_rejected_by_key_auth(
    hook_proxy: tuple[Gateway, HookRecords], surface: str
) -> None:
    proxy, records = hook_proxy
    known_calls: Final = {record["litellm_call_id"] for record in records.drained()}
    response: Final = send(proxy, surface, "success", str(uuid.uuid4()), key="sk-not-a-virtual-key")
    assert response.status_code == 401, response.text

    def new_calls(found: tuple[dict[str, str], ...]) -> set[str]:
        return {record["litellm_call_id"] for record in found if record["litellm_call_id"] not in known_calls}

    (rejected_call,) = eventually(lambda: new_calls(records.drained()), bool, seconds=20)
    assert_hooks(records, surface, rejected_call, one_async_hook(records, surface, "failure"))
