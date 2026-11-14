"""
Helpers for the selection characterization tests.

``TrafficHost`` is a ``ScriptedHost`` whose async inference calls also feed the routing strategy
loggers the core registered on the host, the way litellm's logging does around a real call: input
callbacks see the call start (``log_pre_api_call``), and the ``CustomLogger`` instances in the
``callbacks`` list see it succeed or fail with real start and end times. A call can be held open
on a gate (an ``asyncio.Event``) or delayed for real, so in-flight counts and latencies are real
for any core.
"""

import asyncio
import datetime
from collections.abc import Callable, Mapping
from typing import Final

from litellm.integrations.custom_logger import CustomLogger
from litellm.router_core import RouterCore
from tests.unit.test_router.core.scripted_host import Call, ScriptedHost

MakeCore = Callable[..., RouterCore]

MESSAGES: Final = [{"role": "user", "content": "hi"}]


class TrafficHost(ScriptedHost):
    """``ScriptedHost`` plus the strategy-logger traffic litellm's logging produces around a call."""

    def __init__(self) -> None:
        super().__init__()
        self.gates: dict[str, asyncio.Event] = {}
        self.delays: dict[str, float] = {}
        self.started: list[str] = []  # mutable-ok: the models whose calls have started, in order

    def _loggers(self, name: str) -> tuple[CustomLogger, ...]:
        return tuple(callback for callback in self.lists.get(name, ()) if isinstance(callback, CustomLogger))

    async def _answer_async(self, op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> object:
        model: Final = str(kwargs.get("model") or (args[0] if args else ""))
        logged: Final = self._logging_kwargs(Call(op=op, model=model, kwargs=dict(kwargs)), None, None)
        for logger in self._loggers("input_callback"):
            logger.log_pre_api_call(model=model, messages=kwargs.get("messages"), kwargs=logged)
        self.started.append(model)
        start: Final = datetime.datetime.now()
        gate: Final = self.gates.get(model)
        if gate is not None:
            await gate.wait()
        delay: Final = self.delays.get(model)
        if delay:
            await asyncio.sleep(delay)
        try:
            response = await super()._answer_async(op, args, kwargs)
        except Exception as error:
            failed = {**logged, "exception": error}
            for logger in self._loggers("callbacks"):
                await logger.async_log_failure_event(failed, None, start, datetime.datetime.now())
            raise
        succeeded: Final = self._logging_kwargs(Call(op=op, model=model, kwargs=dict(kwargs)), None, response)
        for logger in self._loggers("callbacks"):
            await logger.async_log_success_event(succeeded, response, start, datetime.datetime.now())
        return response


async def picked_ids(
    core: RouterCore,
    model: str,
    times: int = 20,
    request_kwargs: Mapping[str, object] | None = None,
    **select_arguments: object,  # kwargs-ok: select's keyword options
) -> set[str]:
    """The deployment ids ``select`` picks over ``times`` picks, each with a fresh copy of ``request_kwargs``."""
    picked: Final[set[str]] = set()
    for _ in range(times):
        chosen = await core.select(model, dict(request_kwargs or {}), **select_arguments)  # pyright: ignore[reportCallIssue]  # PythonCore's select takes keyword options
        assert isinstance(chosen, Mapping)
        picked.add(str(chosen["model_info"]["id"]))
    return picked


def ids_of(deployments: object) -> list[str]:  # mutable-ok: a fresh list
    """The ids of a ``select(kind="healthy")`` result, in order (a single dict gives one id)."""
    if isinstance(deployments, Mapping):
        return [str(deployments["model_info"]["id"])]
    assert isinstance(deployments, list)
    return [str(deployment["model_info"]["id"]) for deployment in deployments]


__all__ = ["MESSAGES", "MakeCore", "TrafficHost", "ids_of", "picked_ids"]
