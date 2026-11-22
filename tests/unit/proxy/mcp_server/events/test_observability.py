from collections.abc import Iterator, Mapping
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.events.observability import EventActivity, record, write_log_row

from .fixtures import CALLER


def _activity(caller: UserAPIKeyAuth | None = CALLER, **fields: object) -> EventActivity:  # noqa: ANN003
    base: Final = {
        "call_type": "mcp_event_poll",
        "mode": "poll",
        "server_id": "srv-ci",
        "event_name": "ci-build.failed",
        "outcome": "delivered",
        "caller": caller,
        "event_ids": ("evt_1",),
    }
    return EventActivity(**{**base, **fields})  # pyright: ignore[reportArgumentType]  # test builder


class _Rows:
    def __init__(self) -> None:
        self.rows: list[tuple[EventActivity, UserAPIKeyAuth]] = []  # mutable-ok: captured log rows

    async def write(self, activity: EventActivity, caller: UserAPIKeyAuth) -> None:
        self.rows.append((activity, caller))


@pytest.mark.asyncio
async def test_an_attributed_outcome_writes_one_log_row() -> None:
    rows: Final = _Rows()
    await record(_activity(), write=rows.write)
    assert [(item.outcome, caller.user_id) for item, caller in rows.rows] == [("delivered", "user-1")]


@pytest.mark.asyncio
async def test_an_outcome_without_a_caller_writes_no_log_row() -> None:
    rows: Final = _Rows()
    await record(_activity(caller=None), write=rows.write)
    assert rows.rows == []


@pytest.mark.asyncio
async def test_a_failing_log_writer_never_reaches_the_client() -> None:
    attempts: Final[list[str]] = []  # mutable-ok: records the write attempt

    async def broken(activity: EventActivity, _caller: UserAPIKeyAuth) -> None:
        attempts.append(activity.outcome)
        raise ConnectionError("spend log queue down")

    await record(_activity(), write=broken)
    assert attempts == ["delivered"]


_LOGGED: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


class _Captured(CustomLogger):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger.__init__ is untyped
        self.metadata: list[JsonValue] = []  # mutable-ok: captured success payloads

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        params: Final = _LOGGED.validate_python(kwargs.get("litellm_params"), from_attributes=False)
        self.metadata.append(params.get("metadata"))


@pytest.fixture
def captured() -> Iterator[_Captured]:
    logger: Final = _Captured()
    litellm.logging_callback_manager.add_litellm_callback(logger)
    try:
        yield logger
    finally:
        litellm.callbacks.remove(logger)


@pytest.mark.asyncio
async def test_the_log_row_carries_the_event_details_and_the_caller(captured: _Captured) -> None:
    await write_log_row(_activity(count=3, subscription_id="sub_1"), CALLER)
    (metadata,) = captured.metadata
    assert isinstance(metadata, dict)
    assert metadata["spend_logs_metadata"]["mcp_event"] == {
        "call_type": "mcp_event_poll",
        "mode": "poll",
        "server_id": "srv-ci",
        "event_name": "ci-build.failed",
        "outcome": "delivered",
        "subscription_id": "sub_1",
        "event_ids": ["evt_1"],
        "attempts": None,
        "count": 3,
    }
    assert metadata["user_api_key_user_id"] == "user-1"
