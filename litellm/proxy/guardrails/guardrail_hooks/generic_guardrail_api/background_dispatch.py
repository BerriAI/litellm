import asyncio
import contextvars
from collections.abc import Awaitable, Callable
from typing import Annotated, Final

from pydantic import Field, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger

DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT: Final = 100

FIRE_AND_FORGET_POST_TIMEOUT_SECONDS: Final = 30.0

FIRE_AND_FORGET_DISPATCHED_REASON: Final = "fire_and_forget dispatched, verdict not read"

FIRE_AND_FORGET_DROPPED_REASON: Final = "fire_and_forget_max_inflight reached, call dropped"

FIRE_AND_FORGET_NOT_DISPATCHED_REASON: Final = "fire_and_forget payload could not be built, call not dispatched"

_DROP_LOG_INTERVAL: Final = 100
_FIRE_AND_FORGET_ADAPTER: Final[TypeAdapter[bool]] = TypeAdapter(bool)
_MAX_INFLIGHT_ADAPTER: Final[TypeAdapter[int]] = TypeAdapter(Annotated[int, Field(ge=1)])


def fire_and_forget_from_config(value: object) -> bool:
    if value is None:
        return False
    try:
        return _FIRE_AND_FORGET_ADAPTER.validate_python(value)
    except ValidationError:
        verbose_proxy_logger.warning(
            "Ignoring fire_and_forget=%r, expected true or false. Awaiting every guardrail call", value
        )
        return False


def _parsed_max_inflight(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return _MAX_INFLIGHT_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def max_inflight_from_config(value: object) -> int:
    if value is None:
        return DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT
    parsed: Final = _parsed_max_inflight(value)
    if parsed is None:
        verbose_proxy_logger.warning(
            "Ignoring fire_and_forget_max_inflight=%r, expected an integer of at least 1. Using %d",
            value,
            DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT,
        )
        return DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT
    return parsed


class BackgroundDispatcher:
    def __init__(self, *, guardrail_name: str | None, max_inflight: int) -> None:
        if max_inflight < 1:
            raise ValueError(f"fire_and_forget_max_inflight must be >= 1 (got {max_inflight})")
        self._guardrail_name: Final = guardrail_name
        self._max_inflight: Final = max_inflight
        self._pending: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: strong refs, asyncio keeps only weak ones
        self._dropped: int = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def dropped_count(self) -> int:
        return self._dropped

    def dispatch(self, run: Callable[[], Awaitable[None]], *, context: str) -> bool:
        if len(self._pending) >= self._max_inflight:
            self._dropped += 1
            if self._dropped % _DROP_LOG_INTERVAL == 1:
                verbose_proxy_logger.warning(
                    "Generic Guardrail API (%s, fire_and_forget): dropped %d call(s) so far, "
                    "%d already in flight (fire_and_forget_max_inflight=%d). %s",
                    self._guardrail_name,
                    self._dropped,
                    len(self._pending),
                    self._max_inflight,
                    context,
                )
            return False

        task: Final = contextvars.Context().run(asyncio.create_task, self._run_logging_failures(run, context=context))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return True

    async def _run_logging_failures(self, run: Callable[[], Awaitable[None]], *, context: str) -> None:
        try:
            await run()
        except Exception as e:  # noqa: BLE001  # a detached task has no caller to raise into
            verbose_proxy_logger.warning(
                "Generic Guardrail API (%s, fire_and_forget) call failed. %s: %s",
                self._guardrail_name,
                context,
                e,
            )

    async def wait_for_pending(self) -> None:
        pending: Final = tuple(self._pending)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
