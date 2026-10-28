import json
import os
from collections.abc import Mapping
from datetime import datetime
from typing import Final

import httpx
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger


def _record(hook: str, kwargs: Mapping[str, object]) -> bytes:
    return json.dumps(
        {"hook": hook, "litellm_call_id": kwargs.get("litellm_call_id"), "call_type": kwargs.get("call_type")}
    ).encode()


def _send(hook: str, kwargs: Mapping[str, object]) -> None:
    with httpx.Client(trust_env=False) as client:
        client.post(os.environ["HOOK_COUNT_PROBE_URL"], content=_record(hook, kwargs)).raise_for_status()


async def _send_async(hook: str, kwargs: Mapping[str, object]) -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        response: Final = await client.post(os.environ["HOOK_COUNT_PROBE_URL"], content=_record(hook, kwargs))
        response.raise_for_status()


class HookCountProbe(CustomLogger):
    """Reports every sync and async success and failure hook it receives, one record per call."""

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        _send("sync_success", kwargs)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await _send_async("async_success", kwargs)

    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        _send("sync_failure", kwargs)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await _send_async("async_failure", kwargs)


class SyncOnlyHookCountProbe(CustomLogger):
    """Overrides only the sync success and failure hooks."""

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        _send("sync_success", kwargs)

    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        _send("sync_failure", kwargs)


class BlockingPostCallGuardrail(CustomGuardrail):
    """Rejects every response with a 400 after the provider answered."""

    async def async_post_call_success_hook(
        self, data: Mapping[str, object], user_api_key_dict: object, response: object
    ) -> None:
        raise HTTPException(status_code=400, detail="blocked by the hook-count guardrail")


probe: Final = HookCountProbe()
sync_only_probe: Final = SyncOnlyHookCountProbe()


def sync_success_function(
    kwargs: Mapping[str, object], completion_response: object, start_time: datetime, end_time: datetime
) -> None:
    _send("function_success", kwargs)


def sync_failure_function(
    kwargs: Mapping[str, object], completion_response: object, start_time: datetime, end_time: datetime
) -> None:
    _send("function_failure", kwargs)
