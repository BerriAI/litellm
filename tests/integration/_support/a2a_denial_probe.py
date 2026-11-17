import json
import os
from collections.abc import Mapping
from datetime import datetime
from typing import Final

import httpx
from fastapi import HTTPException

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _logged(hook: str, kwargs: Mapping[str, object]) -> Mapping[str, object]:
    logged: Final = _mapping(kwargs.get("standard_logging_object"))
    metadata: Final = _mapping(logged.get("metadata"))
    return {
        "hook": hook,
        "call_type": kwargs.get("call_type"),
        "litellm_call_id": kwargs.get("litellm_call_id"),
        "model": logged.get("model"),
        "status": logged.get("status"),
        "key_alias": metadata.get("user_api_key_alias"),
        "key_hash": metadata.get("user_api_key_hash"),
        "user_id": metadata.get("user_api_key_user_id"),
        "team_id": metadata.get("user_api_key_team_id"),
    }


def _status(exception: Exception) -> int | None:
    status: Final = getattr(exception, "status_code", None)
    if isinstance(status, int):
        return status
    code: Final = getattr(exception, "code", None)
    return int(code) if isinstance(code, str) and code.isdigit() else None


def _hooked(
    request_data: Mapping[str, object], user_api_key_dict: UserAPIKeyAuth, original_exception: Exception
) -> Mapping[str, object]:
    return {
        "hook": "post_call_failure",
        "call_type": request_data.get("call_type"),
        "litellm_call_id": request_data.get("litellm_call_id"),
        "key_alias": user_api_key_dict.key_alias,
        "key_hash": user_api_key_dict.api_key,
        "user_id": user_api_key_dict.user_id,
        "team_id": user_api_key_dict.team_id,
        "agent_id": request_data.get("agent_id"),
        "metadata_agent_id": _mapping(request_data.get("metadata")).get("agent_id"),
        "status_code": _status(original_exception),
    }


def _send(record: Mapping[str, object]) -> None:
    with httpx.Client(trust_env=False) as client:
        client.post(os.environ["A2A_DENIAL_PROBE_URL"], content=json.dumps(record, default=str)).raise_for_status()


async def _send_async(record: Mapping[str, object]) -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        response: Final = await client.post(os.environ["A2A_DENIAL_PROBE_URL"], content=json.dumps(record, default=str))
        response.raise_for_status()


class A2ADenialProbe(CustomLogger):
    """Reports the identity, target and outcome every logging and failure hook receives, one record per call."""

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await _send_async(_logged("async_success", kwargs))

    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        _send(_logged("sync_failure", kwargs))

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await _send_async(_logged("async_failure", kwargs))

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        await _send_async(_hooked(request_data, user_api_key_dict, original_exception))


class RaisingFailureLogger(CustomLogger):
    """Fails inside both failure hooks."""

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        raise RuntimeError("failure logger is broken")

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        raise RuntimeError("failure hook is broken")


class ReturningTeapotLogger(CustomLogger):
    """Hands a replacement 418 back from the failure hook."""

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> HTTPException:
        return HTTPException(status_code=418, detail="transformed")


class RaisingTeapotLogger(CustomLogger):
    """Raises a replacement 418 from the failure hook."""

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        raise HTTPException(status_code=418, detail="transformed")


probe: Final = A2ADenialProbe()
raising_failure_logger: Final = RaisingFailureLogger()
returning_teapot_logger: Final = ReturningTeapotLogger()
raising_teapot_logger: Final = RaisingTeapotLogger()
