from dataclasses import replace
from typing import Final, cast

import pytest

from litellm._auth.decision import (
    Allow,
    AuthCore,
    Deny,
    KeyBlocked,
    NoMasterKey,
    RequestFacts,
    ResolveEndUser,
    Step,
    WithinModelBudgets,
)
from litellm._auth.driver import drive
from litellm._auth.rust_core import RustCore
from litellm._auth.shadow import ShadowCore, ShadowMismatch
from litellm.rust_bridge import _native
from tests.unit._auth.test_python_core import NOW, PLAIN_KEY, SETTINGS, RecordingHandler


class ScriptedCore:
    def __init__(self, step: Step) -> None:
        self.step: Final = step
        self.calls = 0

    def check_model_budgets(self) -> Step:
        self.calls += 1
        return self.step


FACTS: Final = RequestFacts("/chat/completions", "hash", True)


def test_native_continuations_are_one_shot_and_own_their_pending_decision():
    payload, continuation = _native.request_auth_budgets("Key")
    assert payload == {"kind": "ReadModelBudgetScope"}
    assert isinstance(continuation, _native.NativeAuthContinuation)
    outcome, done = continuation.resume(
        {
            "kind": "ModelScope",
            "value": {
                "models": [],
                "rewritable_model": None,
                "key_has_model_budgets": False,
                "key_budget_fallbacks": {},
                "user_has_model_budgets": False,
            },
        }
    )
    assert outcome == {"kind": "Allow", "grant": {"kind": "WithinModelBudgets"}}
    assert done is None
    with pytest.raises(RuntimeError, match="already consumed"):
        continuation.resume({"kind": "Unit"})


def test_invalid_native_answer_consumes_the_continuation_without_fallback():
    _, continuation = _native.request_auth_budgets("Key")
    assert continuation is not None
    with pytest.raises(RuntimeError, match="invalid auth effect answer"):
        continuation.resume({"kind": "Bool", "value": False})
    with pytest.raises(RuntimeError, match="already consumed"):
        continuation.resume({"kind": "Unit"})


@pytest.mark.asyncio
async def test_shadow_replays_only_after_the_primary_finishes_and_uses_one_io_trace():
    primary: Final = ScriptedCore(ResolveEndUser(False, lambda: Allow(WithinModelBudgets())))
    reference: Final = ScriptedCore(ResolveEndUser(False, lambda: Allow(WithinModelBudgets())))
    step: Final = ShadowCore(cast(AuthCore, primary), cast(AuthCore, reference)).check_model_budgets()
    assert reference.calls == 0
    handler: Final = RecordingHandler()
    assert await drive(step, handler) == Allow(WithinModelBudgets())
    assert handler.calls == (("resolve_end_user", False),)
    assert (primary.calls, reference.calls) == (1, 1)


@pytest.mark.asyncio
async def test_shadow_catches_effect_payload_mismatch_without_a_second_io():
    primary: Final = ScriptedCore(ResolveEndUser(True, lambda: Allow(WithinModelBudgets())))
    reference: Final = ScriptedCore(ResolveEndUser(False, lambda: Allow(WithinModelBudgets())))
    handler: Final = RecordingHandler()
    with pytest.raises(ShadowMismatch, match="effect mismatch"):
        await drive(ShadowCore(cast(AuthCore, primary), cast(AuthCore, reference)).check_model_budgets(), handler)
    assert handler.calls == (("resolve_end_user", True),)


@pytest.mark.asyncio
async def test_shadow_catches_outcome_mismatch_without_exposing_credentials():
    with pytest.raises(ShadowMismatch) as error:
        ShadowCore(
            cast(AuthCore, ScriptedCore(Allow(NoMasterKey("private-credential")))),
            cast(AuthCore, ScriptedCore(Deny(KeyBlocked()))),
        ).check_model_budgets()
    assert str(error.value) == "auth shadow mismatch at Allow"


@pytest.mark.asyncio
async def test_native_requests_keep_independent_continuations():
    core: Final = RustCore()
    first: Final = core.start_resolved_key(replace(FACTS, token="one"), SETTINGS, NOW)
    second: Final = core.start_resolved_key(replace(FACTS, token="two"), SETTINGS, NOW)
    blocked: Final = RecordingHandler(key=replace(PLAIN_KEY, blocked=True))
    plain: Final = RecordingHandler()
    assert await drive(second, blocked) == Deny(KeyBlocked())
    assert isinstance(await drive(first, plain), Allow)
    assert blocked.calls == (("inspect_key", None),)
    assert plain.calls[-1] == ("load_key_context", None)


@pytest.mark.asyncio
async def test_effect_failure_after_native_entry_is_never_replayed(core: AuthCore):
    failure: Final = RuntimeError("effect failed")

    class FailingHandler(RecordingHandler):
        async def enforce_model_access(self) -> None:
            self._record("enforce_model_access")
            raise failure

    handler: Final = FailingHandler()
    with pytest.raises(RuntimeError) as error:
        await drive(core.start_resolved_key(FACTS, SETTINGS, NOW), handler)
    assert error.value is failure
    assert handler.calls == (("inspect_key", None), ("enforce_model_access", None))
