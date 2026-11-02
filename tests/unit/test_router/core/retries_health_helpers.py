"""
Helpers for the retries / deployment-health characterization tests (``test_retries_health_characterization*.py``).

Backoff values carry litellm's random jitter (up to ``JITTER`` seconds), so sleeps are asserted inside
``[base, base + JITTER)``.
"""

import asyncio
import copy
from collections.abc import Callable, Mapping
from typing import Final

import httpx

import litellm
from litellm.router_core import RouterCore
from tests.unit.test_router.core.scripted_host import Call, ScriptedHost

MESSAGES: Final = [{"role": "user", "content": "hi"}]

MakeCore = Callable[..., RouterCore]
ErrorBuilder = Callable[[], Exception]

_REQUEST: Final = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def fresh(builder: ErrorBuilder) -> Callable[..., Exception]:
    """A scripted outcome that raises a new error per call."""
    return lambda **_: builder()


def http_response(status: int, headers: Mapping[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, headers=dict(headers or {}), request=_REQUEST)


def not_found() -> litellm.NotFoundError:
    return litellm.NotFoundError(message="no such model", llm_provider="openai", model="gpt-4o")


def permission_denied() -> litellm.PermissionDeniedError:
    return litellm.PermissionDeniedError(
        message="forbidden", llm_provider="openai", model="gpt-4o", response=http_response(403)
    )


def unprocessable() -> litellm.UnprocessableEntityError:
    return litellm.UnprocessableEntityError(
        message="unprocessable", llm_provider="openai", model="gpt-4o", response=http_response(422)
    )


def service_unavailable() -> litellm.ServiceUnavailableError:
    return litellm.ServiceUnavailableError(message="unavailable", llm_provider="openai", model="gpt-4o")


def bad_gateway() -> litellm.BadGatewayError:
    return litellm.BadGatewayError(message="bad gateway", llm_provider="openai", model="gpt-4o")


def api_connection() -> litellm.APIConnectionError:
    return litellm.APIConnectionError(message="connection reset", llm_provider="openai", model="gpt-4o")


def api_status(status: int) -> ErrorBuilder:
    return lambda: litellm.APIError(
        status_code=status, message=f"status {status}", llm_provider="openai", model="gpt-4o"
    )


def plain_error() -> ValueError:
    return ValueError("not a provider error")


def rate_limited(retry_after: str | None = None) -> litellm.RateLimitError:
    headers: Final = {"retry-after": retry_after} if retry_after is not None else {}
    return litellm.RateLimitError(
        message="slow down", llm_provider="openai", model="gpt-4o", response=httpx.Response(429, headers=headers)
    )


async def attempt(core: RouterCore, model: str = "g", **kwargs: object) -> object:  # kwargs-ok: request kwargs
    """The response, or the error the core raised. A request the Rust core does not serve yet propagates, so the
    test is skipped rather than failed."""
    from litellm.router_rust_core import RustCoreUnsupported

    try:
        return await core.acompletion(model=model, messages=MESSAGES, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]  # every core serves acompletion
    except RustCoreUnsupported:
        raise
    except Exception as error:
        return error


def attempt_sync(core: RouterCore, model: str = "g", **kwargs: object) -> object:  # kwargs-ok: request kwargs
    from litellm.router_rust_core import RustCoreUnsupported

    try:
        return core.completion(model=model, messages=MESSAGES, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]  # every core serves completion
    except RustCoreUnsupported:
        raise
    except Exception as error:
        return error


def cooled_down(core: RouterCore, ids: list[str]) -> dict[str, Mapping[str, object]]:  # mutable-ok: a fresh dict
    return dict(core.health().cooldowns.get_active_cooldowns(ids, None))


def headers_of(response: object) -> Mapping[str, object]:
    hidden: Final = getattr(response, "_hidden_params", None) or {}
    return hidden.get("additional_headers") or {}


class MetadataRecorder:
    """A scripted outcome factory that snapshots the request metadata each call carried, then answers."""

    KEYS: Final = (
        "attempted_retries",
        "max_retries",
        "model_group",
        "model_group_size",
        "request_retry_count",
        "previous_models",
    )

    def __init__(self) -> None:
        self.snapshots: Final[list[dict[str, object]]] = []

    def then(self, builder: ErrorBuilder | None) -> Callable[..., object]:
        def outcome(**kwargs: object) -> object:  # kwargs-ok: the request
            metadata: Final = kwargs.get("metadata") or kwargs.get("litellm_metadata") or {}
            assert isinstance(metadata, Mapping)
            self.snapshots.append({key: copy.deepcopy(metadata.get(key)) for key in self.KEYS})
            return builder() if builder is not None else None

        return outcome


class BackgroundSuccessLoggingHost(ScriptedHost):
    """``ScriptedHost`` whose async success callbacks run in a background task after the response is returned,
    as litellm's logging worker runs them (``ScriptedHost`` runs them before returning). Await ``drain()``
    before reading what they wrote."""

    __slots__ = ("_pending",)

    def __init__(self) -> None:
        super().__init__()
        self._pending: list[asyncio.Task[None]] = []

    async def _answer_async(self, op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> object:
        call, outcome = self._outcome(op, args, kwargs)
        error, response = self._finish(op, call, outcome)
        if error is not None:
            logged_failure: Final = self._logging_kwargs(call, error, response)
            for callback in list(self.lists.get("failure_callback", ())):
                if callable(callback):
                    callback(logged_failure, None, 0.0, 0.0)
            for callback in list(self.lists.get("_async_failure_callback", ())):
                if callable(callback):
                    await callback(logged_failure, None, 0.0, 0.0)
            raise error

        async def log_success(logged_call: Call) -> None:
            logged: Final = self._logging_kwargs(logged_call, None, response)
            for callback in list(self.lists.get("_async_success_callback", ())):
                if callable(callback):
                    await callback(logged, response, 0.0, 0.0)

        self._pending.append(asyncio.ensure_future(log_success(call)))
        return response

    async def drain(self) -> None:
        while self._pending:
            await self._pending.pop()


__all__ = [
    "MESSAGES",
    "BackgroundSuccessLoggingHost",
    "ErrorBuilder",
    "MakeCore",
    "MetadataRecorder",
    "api_connection",
    "api_status",
    "attempt",
    "attempt_sync",
    "bad_gateway",
    "cooled_down",
    "fresh",
    "headers_of",
    "http_response",
    "not_found",
    "permission_denied",
    "plain_error",
    "rate_limited",
    "service_unavailable",
    "unprocessable",
]
