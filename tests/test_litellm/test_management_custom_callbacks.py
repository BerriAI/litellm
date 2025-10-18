from typing import Any, Dict, List, Optional

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.management_helpers.utils import send_management_endpoint_alert


class RecordingAsyncLogger(CustomLogger):
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    async def async_log_management_event(
        self,
        event_name: str,
        event_payload: dict,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    ) -> None:
        self.events.append(
            {
                "event_name": event_name,
                "payload": event_payload,
                "user_id": getattr(user_api_key_dict, "user_id", None),
            }
        )


class RecordingSyncLogger(CustomLogger):
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_management_event(
        self,
        event_name: str,
        event_payload: dict,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    ) -> None:
        self.events.append(
            {
                "event_name": event_name,
                "payload": event_payload,
                "user_id": getattr(user_api_key_dict, "user_id", None),
            }
        )


@pytest.fixture
def reset_callbacks():
    original_callbacks = list(litellm.callbacks)
    original_success = list(litellm.success_callback)
    original_failure = list(litellm.failure_callback)
    original_async_success = list(litellm._async_success_callback)
    original_async_failure = list(litellm._async_failure_callback)
    yield
    litellm.callbacks = original_callbacks  # type: ignore[assignment]
    litellm.success_callback = original_success  # type: ignore[assignment]
    litellm.failure_callback = original_failure  # type: ignore[assignment]
    litellm._async_success_callback = original_async_success  # type: ignore[assignment]
    litellm._async_failure_callback = original_async_failure  # type: ignore[assignment]


def _set_proxy_logging_obj(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    class _ProxyLoggingStub:
        def __init__(self) -> None:
            self.slack_alerting_instance = None

    monkeypatch.setattr(proxy_server, "proxy_logging_obj", _ProxyLoggingStub())


@pytest.mark.asyncio
async def test_management_event_triggers_async_custom_logger(
    monkeypatch: pytest.MonkeyPatch, reset_callbacks
) -> None:
    _set_proxy_logging_obj(monkeypatch)

    logger = RecordingAsyncLogger()
    litellm.callbacks = [logger]  # type: ignore[assignment]
    litellm.success_callback = []  # type: ignore[assignment]
    litellm.failure_callback = []  # type: ignore[assignment]
    litellm._async_success_callback = []  # type: ignore[assignment]
    litellm._async_failure_callback = []  # type: ignore[assignment]

    user_api_key = UserAPIKeyAuth(user_id="user-123", user_role="proxy_admin")
    request_kwargs = {
        "data": {"key": "value"},
        "http_request": object(),
    }

    await send_management_endpoint_alert(
        request_kwargs=request_kwargs,
        user_api_key_dict=user_api_key,
        function_name="update_key_fn",
    )

    assert len(logger.events) == 1
    event = logger.events[0]
    assert event["event_name"] == "Virtual Key Updated"
    payload = event["payload"]
    assert payload["function_name"] == "update_key_fn"
    assert payload["alert_type"] == "virtual_key_updated"
    assert "triggered_at" in payload
    assert "http_request" not in payload["request"]
    assert event["user_id"] == "user-123"


@pytest.mark.asyncio
async def test_management_event_triggers_sync_custom_logger(
    monkeypatch: pytest.MonkeyPatch, reset_callbacks
) -> None:
    _set_proxy_logging_obj(monkeypatch)

    logger = RecordingSyncLogger()
    litellm.callbacks = [logger]  # type: ignore[assignment]
    litellm.success_callback = []  # type: ignore[assignment]
    litellm.failure_callback = []  # type: ignore[assignment]
    litellm._async_success_callback = []  # type: ignore[assignment]
    litellm._async_failure_callback = []  # type: ignore[assignment]

    user_api_key = UserAPIKeyAuth(user_id="sync-user", user_role="internal_user")

    await send_management_endpoint_alert(
        request_kwargs={"data": {"foo": "bar"}},
        user_api_key_dict=user_api_key,
        function_name="delete_key_fn",
    )

    assert len(logger.events) == 1
    event = logger.events[0]
    assert event["event_name"] == "Virtual Key Deleted"
    payload = event["payload"]
    assert payload["function_name"] == "delete_key_fn"
    assert payload["alert_type"] == "virtual_key_deleted"
    assert event["user_id"] == "sync-user"
