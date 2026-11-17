import asyncio
import sys
import types
from pathlib import Path

repo = Path(globals()["repo"])
rust = globals()["rust"]

for name, path in (("litellm", repo / "litellm"), ("litellm.integrations", repo / "litellm" / "integrations")):
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.__path__ = [str(path)]
        sys.modules[name] = stub

from litellm.integrations.custom_logger_v2 import (
    Abort,
    Activation,
    CallbackManifest,
    CallbackRegistry,
    CallFacts,
    CallIds,
    CallKind,
    CallTarget,
    CallTiming,
    Dispatcher,
    Field,
    Hook,
    InProcessSink,
    Patch,
    RequestSelection,
    Selection,
    Subscription,
    build_manifest,
    plan_hook,
)
from litellm.integrations.custom_logger_v2.rust import rust_logger
from litellm.integrations.custom_logger_v2.planner import Entry

seen = []
example = build_manifest(rust_logger(rust))


class Legacy:
    pass


def ignore(logger, kwargs):
    return None


async def done(result):
    return result


def legacy(name, hook, patch=None):
    def call(binding, fields):
        seen.append(f"{name} {hook.value} {fields.get(Field.model)}")
        return done(patch)

    subscription = Subscription(
        hook=hook,
        handler="call",
        reads=frozenset({Field.deprecated_kwargs}),
        writes=frozenset({Field.deprecated_kwargs}) if hook is Hook.pre_call else frozenset(),
        may_abort=hook is Hook.pre_call,
        legacy=True,
    )
    return CallbackManifest(
        logger=Legacy(),
        selection=Selection(name=name, activation=Activation.required),
        subscriptions={hook: subscription},
        request_state=None,
        legacy_call=call,
    )


def line(hook, first_patch=None):
    manifests = (legacy("first", hook, first_patch), example, legacy("last", hook))
    return plan_hook(
        tuple(Entry(position, manifest, manifest.subscriptions[hook]) for position, manifest in enumerate(manifests))
    )


def facts(call_id):
    return CallFacts(
        ids=CallIds(call_id=call_id),
        kind=CallKind(call_type="acompletion"),
        target=CallTarget(model="gpt"),
        timing=CallTiming(start=1.0),
    )


async def request(model, call_id):
    dispatcher = Dispatcher(CallbackRegistry(), sink=InProcessSink())
    scope = dispatcher.open_scope(RequestSelection())
    fields = {Field.deprecated_kwargs: {}, Field.model: "client"}
    pre_call = await scope.run_plan(
        Hook.pre_call, line(Hook.pre_call, Patch().set(Field.model, model)), fields, facts(call_id), ignore
    )
    if pre_call.outcome is not None:
        return pre_call
    await scope.run_plan(Hook.attempt_ended, line(Hook.attempt_ended), pre_call.fields, facts(call_id), ignore)
    await dispatcher.sink.drain()
    return pre_call


async def main():
    served = await request("gpt", "call-1")
    blocked = await request("blocked", "call-2")
    return served, blocked


served, blocked = asyncio.run(main())
served_model = served.fields[Field.model]
outcome = blocked.outcome
blocked_abort = (
    (outcome.status, outcome.message, outcome.code, dict(outcome.headers)) if isinstance(outcome, Abort) else None
)
