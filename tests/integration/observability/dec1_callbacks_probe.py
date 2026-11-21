"""Callback module the dec1 callback contract tests load into the proxy as `module.instance`.

`recorder` overrides every proxy and logging hook a CustomLogger can override and reports each call to the wire
server named by DEC1_CALLBACKS_PROBE_URL, with a per-process sequence number so tests can read the order. A hook
changes behaviour only when the request text carries one of the `dec1-...` directives below, so one proxy serves
every scenario. `success_only` and `failure_only` are plain recording loggers for the success and failure lists,
the two functions are legacy function callbacks, and `BlockingGuardrail` is a pre-call guardrail that blocks on a
directive.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
from collections.abc import AsyncGenerator, Mapping
from datetime import datetime
from typing import Final

import httpx
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information
from litellm.integrations.custom_logger import CustomLogger

PRE_MODIFY: Final = "dec1-pre-modify"
PRE_REJECT_STRING: Final = "dec1-pre-reject-string"
PRE_RAISE_HTTP: Final = "dec1-pre-raise-http"
PRE_RAISE_PLAIN: Final = "dec1-pre-raise-plain"
MODERATION_RAISE: Final = "dec1-moderation-raise"
POST_MODIFY: Final = "dec1-post-modify"
POST_RAISE_HTTP: Final = "dec1-post-raise-http"
FAILURE_TRANSFORM: Final = "dec1-failure-transform"
FAILURE_RAISE_PLAIN: Final = "dec1-failure-raise-plain"
LOG_RAISE: Final = "dec1-log-raise"
LOG_SLOW: Final = "dec1-log-slow"
DEPLOYMENT_MODIFY: Final = "dec1-deployment-modify"
GUARDRAIL_BLOCK: Final = "dec1-guardrail-block"

REWRITTEN_REQUEST: Final = "rewritten by the dec1 pre-call hook"
REWRITTEN_RESPONSE: Final = "rewritten by the dec1 post-call hook"
REJECTION: Final = "rejected by the dec1 pre-call hook"
SLOW_LOG_SECONDS: Final = 6
OUTCOME_HEADER: Final = "x-dec1-callback-outcome"

_SEQUENCE: Final = itertools.count()


def _text(container: object) -> str:
    if not isinstance(container, Mapping):
        return ""
    return json.dumps(
        {name: container.get(name) for name in ("messages", "input", "prompt")}, default=str, ensure_ascii=False
    )


def _wants(container: object, directive: str) -> bool:
    return directive in _text(container)


def _record(logger: str, hook: str, source: object, **extra: object) -> bytes:
    fields: Final = source if isinstance(source, Mapping) else {}
    return json.dumps(
        {
            "logger": logger,
            "hook": hook,
            "seq": next(_SEQUENCE),
            "pid": os.getpid(),
            "litellm_call_id": fields.get("litellm_call_id"),
            "call_type": fields.get("call_type"),
            "text": _text(fields),
            **extra,
        },
        default=str,
    ).encode()


def _send(body: bytes) -> None:
    with httpx.Client(trust_env=False) as client:
        client.post(os.environ["DEC1_CALLBACKS_PROBE_URL"], content=body).raise_for_status()


async def _send_async(body: bytes) -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        (await client.post(os.environ["DEC1_CALLBACKS_PROBE_URL"], content=body)).raise_for_status()


def _log_extra(kwargs: Mapping[str, object], response_obj: object) -> dict[str, object]:
    return {
        "standard": kwargs.get("standard_logging_object"),
        "kwargs_keys": sorted(str(key) for key in kwargs),
        "response_type": type(response_obj).__name__,
    }


class Recorder(CustomLogger):
    """Reports every hook it receives and acts on the dec1 directives found in the request text."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def log_pre_api_call(self, model: str, messages: object, kwargs: Mapping[str, object]) -> None:
        _send(_record(self.name, "log_pre_api_call", kwargs, model=model))

    async def async_log_pre_api_call(self, model: str, messages: object, kwargs: Mapping[str, object]) -> None:
        await _send_async(_record(self.name, "async_log_pre_api_call", kwargs, model=model))

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        _send(_record(self.name, "log_success_event", kwargs))

    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        _send(_record(self.name, "log_failure_event", kwargs))

    def log_stream_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        _send(_record(self.name, "log_stream_event", kwargs))

    async def async_log_stream_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await _send_async(_record(self.name, "async_log_stream_event", kwargs))

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        if _wants(kwargs, LOG_SLOW):
            await asyncio.sleep(SLOW_LOG_SECONDS)
        await _send_async(_record(self.name, "async_log_success_event", kwargs, **_log_extra(kwargs, response_obj)))
        if _wants(kwargs, LOG_RAISE):
            raise RuntimeError("the dec1 success logger crashed")

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await _send_async(_record(self.name, "async_log_failure_event", kwargs, **_log_extra(kwargs, response_obj)))
        if _wants(kwargs, LOG_RAISE):
            raise RuntimeError("the dec1 failure logger crashed")

    async def async_pre_call_hook(
        self, user_api_key_dict: object, cache: object, data: dict[str, object], call_type: str
    ) -> dict[str, object] | str | None:
        await _send_async(
            _record(
                self.name,
                "async_pre_call_hook",
                data,
                hook_call_type=call_type,
                key_alias=getattr(user_api_key_dict, "key_alias", None),
                has_logging_obj="litellm_logging_obj" in data,
            )
        )
        if _wants(data, PRE_REJECT_STRING):
            return REJECTION
        if _wants(data, PRE_RAISE_HTTP):
            raise HTTPException(status_code=403, detail={"error": "blocked by the dec1 pre-call hook"})
        if _wants(data, PRE_RAISE_PLAIN):
            raise ValueError("the dec1 pre-call hook crashed")
        if _wants(data, PRE_MODIFY):
            return {**data, "messages": [{"role": "user", "content": REWRITTEN_REQUEST}]}
        return None

    async def async_moderation_hook(self, data: dict[str, object], user_api_key_dict: object, call_type: str) -> None:
        await _send_async(_record(self.name, "async_moderation_hook", data, hook_call_type=call_type))
        if _wants(data, MODERATION_RAISE):
            raise HTTPException(status_code=400, detail={"error": "blocked by the dec1 moderation hook"})

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, object], call_type: object
    ) -> dict[str, object] | None:
        await _send_async(_record(self.name, "async_pre_call_deployment_hook", kwargs, hook_call_type=str(call_type)))
        if _wants(kwargs, DEPLOYMENT_MODIFY):
            return {**kwargs, "max_tokens": 7}
        return None

    async def async_post_call_success_deployment_hook(
        self, request_data: dict[str, object], response: object, call_type: object
    ) -> None:
        await _send_async(_record(self.name, "async_post_call_success_deployment_hook", request_data))

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: object, response: object
    ) -> object:
        await _send_async(
            _record(self.name, "async_post_call_success_hook", data, response_type=type(response).__name__)
        )
        if _wants(data, POST_RAISE_HTTP):
            raise HTTPException(status_code=422, detail={"error": "blocked by the dec1 post-call hook"})
        if _wants(data, POST_MODIFY):
            choices: Final = getattr(response, "choices", None)
            if choices:
                choices[0].message.content = REWRITTEN_RESPONSE
            return response
        return None

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: object,
        traceback_str: str | None = None,
    ) -> HTTPException | None:
        await _send_async(
            _record(
                self.name,
                "async_post_call_failure_hook",
                request_data,
                exception_class=type(original_exception).__name__,
                exception_status=getattr(original_exception, "status_code", None),
                has_logging_obj="litellm_logging_obj" in request_data,
            )
        )
        if _wants(request_data, FAILURE_RAISE_PLAIN):
            raise RuntimeError("the dec1 failure hook crashed")
        if _wants(request_data, FAILURE_TRANSFORM):
            return HTTPException(status_code=503, detail="transformed by the dec1 failure hook")
        return None

    async def async_post_call_streaming_hook(self, user_api_key_dict: object, response: str) -> None:
        await _send_async(
            _record(
                self.name,
                "async_post_call_streaming_hook",
                {},
                stream_text=response,
                key_alias=getattr(user_api_key_dict, "key_alias", None),
            )
        )

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: object, response: object, request_data: dict[str, object]
    ) -> AsyncGenerator[object, None]:
        await _send_async(_record(self.name, "async_post_call_streaming_iterator_hook", request_data))
        async for item in response:  # pyright: ignore[reportGeneralTypeIssues]  # the proxy hands an async iterator
            yield item

    async def async_post_call_response_headers_hook(
        self,
        data: dict[str, object],
        user_api_key_dict: object,
        response: object,
        request_headers: dict[str, str] | None = None,
        litellm_call_info: dict[str, object] | None = None,
    ) -> dict[str, str]:
        await _send_async(
            _record(
                self.name,
                "async_post_call_response_headers_hook",
                data,
                call_info=litellm_call_info,
                saw_request_headers=sorted(request_headers or {}),
            )
        )
        return {OUTCOME_HEADER: "failure" if response is None else "success"}


class LogOnly(CustomLogger):
    """Reports only the async success and failure log events, for the success_callback and failure_callback lists."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await _send_async(_record(self.name, "async_log_success_event", kwargs, **_log_extra(kwargs, response_obj)))

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await _send_async(_record(self.name, "async_log_failure_event", kwargs, **_log_extra(kwargs, response_obj)))


class BlockingGuardrail(CustomGuardrail):
    """Pre-call guardrail that answers 400 when the request text carries the dec1-guardrail-block directive."""

    @log_guardrail_information
    async def async_pre_call_hook(
        self, user_api_key_dict: object, cache: object, data: dict[str, object], call_type: str
    ) -> dict[str, object]:
        if _wants(data, GUARDRAIL_BLOCK):
            raise HTTPException(status_code=400, detail={"error": "blocked by the dec1 guardrail"})
        return data


def success_function(
    kwargs: Mapping[str, object], completion_response: object, start_time: datetime, end_time: datetime
) -> None:
    _send(_record("success_function", "function_success", kwargs))


def failure_function(
    kwargs: Mapping[str, object], completion_response: object, start_time: datetime, end_time: datetime
) -> None:
    _send(_record("failure_function", "function_failure", kwargs))


recorder: Final = Recorder("recorder")
second_recorder: Final = Recorder("second_recorder")
success_only: Final = LogOnly("success_only")
failure_only: Final = LogOnly("failure_only")
key_scoped: Final = LogOnly("key_scoped")
