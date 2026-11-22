from collections.abc import Iterator
from types import SimpleNamespace
from typing import Final

import pytest

import litellm.proxy.proxy_server as ps
from litellm.integrations.lens.proxy.ledger import ledger_request_id
from litellm.proxy.spend_tracking.spend_logs_ledger_setting import capture_spend_logs_ledger_setting

_CHAT: Final = {"call_type": "acompletion", "litellm_call_id": "call-1"}
_UNSECURED: Final = {"disable_responses_id_security": True}


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    capture_spend_logs_ledger_setting({})()
    monkeypatch.setattr(ps, "prisma_client", object())
    monkeypatch.setattr(ps, "general_settings", {})
    try:
        yield
    finally:
        capture_spend_logs_ledger_setting({})()


@pytest.mark.usefixtures("proxy")
@pytest.mark.parametrize(
    ("settings", "connected", "startup", "expected"),
    (
        pytest.param({}, True, None, "chatcmpl-1", id="ledger-on-by-default"),
        pytest.param({}, False, None, None, id="no-database"),
        pytest.param({"use_spend_logs_ledger": False}, True, None, None, id="ledger-disabled"),
        pytest.param({"disable_spend_updates": True}, True, None, None, id="spend-updates-disabled"),
        pytest.param({}, True, {"use_spend_logs_ledger": False}, None, id="startup-capture-wins"),
    ),
)
def test_only_a_connected_enabled_ledger_names_a_commit_to_wait_for(
    monkeypatch: pytest.MonkeyPatch,
    settings: dict[str, object],
    connected: bool,
    startup: dict[str, object] | None,
    expected: str | None,
) -> None:
    monkeypatch.setattr(ps, "general_settings", settings)
    if not connected:
        monkeypatch.setattr(ps, "prisma_client", None)
    forget: Final = capture_spend_logs_ledger_setting(startup) if startup is not None else None
    try:
        assert ledger_request_id(_CHAT, {"id": "chatcmpl-1"}) == expected
    finally:
        if forget is not None:
            forget()


@pytest.mark.usefixtures("proxy")
@pytest.mark.parametrize(
    ("settings", "kwargs", "response", "expected"),
    (
        pytest.param({}, _CHAT, SimpleNamespace(id="chatcmpl-1"), "chatcmpl-1", id="response-object"),
        pytest.param({}, _CHAT, None, "call-1", id="call-id-fallback"),
        pytest.param({}, {"call_type": "acompletion"}, None, None, id="no-id"),
        pytest.param(_UNSECURED, {"call_type": "aget_responses"}, {"id": "resp-1"}, None, id="unsecured-free-read"),
        pytest.param(
            _UNSECURED,
            {"call_type": "acancel_responses", "response_cost": 0.1},
            {"id": "resp-1"},
            "resp-1",
            id="unsecured-charged-cancel",
        ),
        pytest.param({}, {"call_type": "aget_responses"}, {"id": "resp-1"}, "resp-1", id="secured-free-read"),
        pytest.param(
            {},
            {"call_type": "aretrieve_batch", "response_cost": 0.5},
            {"id": "batch-1"},
            "batch-1_batch_cost",
            id="charged-batch",
        ),
        pytest.param(
            {}, {"call_type": "aretrieve_batch", "response_cost": 0.0}, {"id": "batch-1"}, None, id="free-batch"
        ),
        pytest.param(
            {},
            {"call_type": "avideo_status", "polled_video_status": "completed", "response_cost": 0.5},
            {"id": "vid-1"},
            "vid-1_video_cost",
            id="charged-video",
        ),
        pytest.param(
            {},
            {"call_type": "avideo_status", "polled_video_status": "completed", "response_cost": 0.0},
            {"id": "vid-1"},
            "vid-1",
            id="free-video-poll",
        ),
    ),
)
def test_the_commit_id_is_the_spend_log_id_and_free_settle_once_polls_wait_for_none(
    monkeypatch: pytest.MonkeyPatch,
    settings: dict[str, object],
    kwargs: dict[str, object],
    response: object,
    expected: str | None,
) -> None:
    monkeypatch.setattr(ps, "general_settings", settings)
    assert ledger_request_id(kwargs, response) == expected
