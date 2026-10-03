"""OracleRouter: sticky program bindings, program id resolution, delayed feedback, eviction."""

import asyncio

import pytest

from litellm.router_strategy.oracle_router.config import (
    CHOSEN_MODEL_METADATA_KEY,
    PROGRAM_ID_HEADER,
    PROGRAM_ID_METADATA_KEY,
)
from litellm.router_strategy.oracle_router.decision import ProgramContext
from litellm.router_strategy.oracle_router.oracle_router import OracleRouter, first_user_text, resolve_program_id
from litellm.router_strategy.oracle_router.verifier import ProgramOutcome, ReportedVerifier
from litellm.types.router import OracleRouterConfig

MODELS = ("smart", "fast")


class _RecordingMaker:
    """Alternates models per bind and records every update in arrival order."""

    def __init__(self) -> None:
        self.selections = 0
        self.updates: list[tuple[str, str, float]] = []

    @property
    def models(self):
        return MODELS

    async def select(self, context: ProgramContext) -> str:
        self.selections += 1
        return MODELS[(self.selections - 1) % 2]

    def update(self, context: ProgramContext, model: str, score: float) -> None:
        self.updates.append((context.program_id, model, score))

    def snapshot(self):
        return {"type": "recording"}


class _SlowVerifier:
    def __init__(self, delays: dict[str, float]) -> None:
        self.delays = delays

    async def verify(self, outcome: ProgramOutcome) -> float:
        await asyncio.sleep(self.delays.get(outcome.program_id, 0.0))
        return outcome.score if outcome.score is not None else 0.0


def _router(verifier=None, maker=None, clock=None, **config) -> tuple[OracleRouter, _RecordingMaker]:
    maker = maker or _RecordingMaker()
    router = OracleRouter(
        router_name="oracle",
        config=OracleRouterConfig(available_models=list(MODELS), **config),
        decision_maker=maker,
        verifier=verifier or ReportedVerifier(),
        **({"clock": clock} if clock else {}),
    )
    return router, maker


def _messages(text: str = "fix the failing test in src/parser.py"):
    return [{"role": "system", "content": "be terse"}, {"role": "user", "content": text}]


def test_resolve_program_id_precedence_and_header():
    assert resolve_program_id({"metadata": {"program_id": "a", "litellm_session_id": "b"}}, "program_id") == "a"
    assert resolve_program_id({"metadata": {"litellm_session_id": "b"}}, "program_id") == "b"
    assert resolve_program_id({"metadata": {"session_id": 7}}, "program_id") == "7"
    assert resolve_program_id({"litellm_session_id": "lp"}, "program_id") == "lp"
    assert resolve_program_id({"headers": {PROGRAM_ID_HEADER.upper(): "h"}}, "program_id") == "h"
    assert resolve_program_id({"metadata": {"task": "t"}}, "task") == "t"
    assert resolve_program_id({"metadata": {"program_id": ""}}, "program_id") is None


def test_first_user_text_reads_the_initial_request_including_text_parts():
    assert first_user_text(_messages("hello")) == "hello"
    parts = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "a"}, {"type": "image_url"}, {"type": "text", "text": "b"}],
        }
    ]
    assert first_user_text(parts) == "a\nb"
    assert first_user_text([{"role": "assistant", "content": "x"}]) == ""
    assert first_user_text(None) == ""


@pytest.mark.asyncio
async def test_first_request_binds_and_later_requests_reuse_the_binding():
    router, maker = _router()
    first = {"metadata": {"program_id": "t1", "user_api_key_hash": "key-1"}}
    response = await router.async_pre_routing_hook(model="oracle", request_kwargs=first, messages=_messages())
    assert response is not None and response.model == "smart"
    assert (
        first["metadata"][CHOSEN_MODEL_METADATA_KEY] == "smart" and first["metadata"][PROGRAM_ID_METADATA_KEY] == "t1"
    )
    assert response.routing_decision["router_type"] == "oracle" and response.routing_decision["routed_model"] == "smart"

    later = {"metadata": {"program_id": "t1", "user_api_key_hash": "key-1"}}
    again = await router.async_pre_routing_hook(
        model="oracle", request_kwargs=later, messages=_messages("now run the tests")
    )
    assert again is not None and again.model == "smart"
    assert maker.selections == 1
    assert router.binding("t1", "key-1").api_key_hash == "key-1"

    other = await router.async_pre_routing_hook(
        model="oracle", request_kwargs={"metadata": {"program_id": "t2"}}, messages=_messages()
    )
    assert other.model == "fast" and router.programs_bound == 2


@pytest.mark.asyncio
async def test_concurrent_first_requests_of_one_program_share_a_single_binding():
    class _SlowMaker(_RecordingMaker):
        async def select(self, context: ProgramContext) -> str:
            await asyncio.sleep(0.01)  # a decision maker that does I/O (pre_routing, custom)
            return await super().select(context)

    router, maker = _router(maker=_SlowMaker())
    responses = await asyncio.gather(
        *(
            router.async_pre_routing_hook(
                model="oracle", request_kwargs={"metadata": {"program_id": "t1"}}, messages=_messages()
            )
            for _ in range(3)
        )
    )
    assert {response.model for response in responses} == {"smart"}
    assert maker.selections == 1 and router.programs_bound == 1 and router.binding("t1").model == "smart"


@pytest.mark.asyncio
async def test_a_non_finite_verifier_score_is_a_failed_verification():
    class _NaN:
        async def verify(self, outcome: ProgramOutcome) -> float:
            return float("nan")

    router, maker = _router(verifier=_NaN())
    await router.async_pre_routing_hook(
        model="oracle", request_kwargs={"metadata": {"program_id": "t"}}, messages=_messages()
    )
    router.observe_request("t", cost=0.1, response_text="answer")
    result = await router.complete("t")
    assert result != result and router.verifications_failed == 1 and maker.updates == []
    assert router.binding("t") is None  # the slot was released all the same


@pytest.mark.asyncio
async def test_requests_without_a_program_id_route_statelessly_and_never_learn():
    router, maker = _router()
    kwargs = {"metadata": {}}
    response = await router.async_pre_routing_hook(model="oracle", request_kwargs=kwargs, messages=_messages())
    assert response.model == "smart" and PROGRAM_ID_METADATA_KEY not in kwargs["metadata"]
    assert router.stateless_requests == 1 and router.binding("") is None
    assert router.complete("", score=1.0) is None


@pytest.mark.asyncio
async def test_complete_releases_the_slot_now_and_learns_in_verification_order():
    router, maker = _router(verifier=_SlowVerifier({"t1": 0.05, "t2": 0.0}))
    for program_id in ("t1", "t2"):
        await router.async_pre_routing_hook(
            model="oracle", request_kwargs={"metadata": {"program_id": program_id}}, messages=_messages()
        )
    router.observe_request("t1", cost=0.4, response_text="patch")
    router.observe_request("t1", cost=0.2, response_text="tests pass")
    router.observe_request("t2", cost=0.1, response_text="wrong")
    t1 = router.complete("t1", score=1.0)
    t2 = router.complete("t2", score=0.0)
    assert router.binding("t1") is None and router.binding("t2") is None
    assert router.pending_verifications == 2
    assert await t2 == 0.0
    assert await t1 == 1.0
    assert [u[0] for u in maker.updates] == ["t2", "t1"]
    assert maker.updates[1] == ("t1", "smart", 1.0)
    snapshot = await router.get_state_snapshot()
    assert snapshot["programs_completed"] == 2 and snapshot["recent_accuracy"] == {"smart": 1.0, "fast": 0.0}
    assert snapshot["recent_feedback"][-1]["cost"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_complete_cost_override_and_payload_reach_the_verifier():
    seen: list[ProgramOutcome] = []

    class _Capture:
        async def verify(self, outcome: ProgramOutcome) -> float:
            seen.append(outcome)
            return 1.0

    router, maker = _router(verifier=_Capture())
    await router.async_pre_routing_hook(
        model="oracle", request_kwargs={"metadata": {"program_id": "t"}}, messages=_messages("do x")
    )
    transcript = [{"role": "user", "content": "do x"}, {"role": "assistant", "content": "answer"}]
    router.observe_request("t", cost=0.3, response_text="answer", messages=transcript)
    await router.complete("t", cost=2.5, payload={"container": "c1"})
    assert seen[0].cost == 2.5 and seen[0].payload == {"container": "c1"} and seen[0].response_text == "answer"
    assert list(seen[0].messages) == transcript
    assert seen[0].prompt == "do x" and maker.updates == [("t", "smart", 1.0)]


@pytest.mark.asyncio
async def test_failed_verification_is_counted_and_does_not_update():
    class _Broken:
        async def verify(self, outcome):
            raise RuntimeError("sandbox down")

    router, maker = _router(verifier=_Broken())
    await router.async_pre_routing_hook(
        model="oracle", request_kwargs={"metadata": {"program_id": "t"}}, messages=_messages()
    )
    router.observe_request("t", cost=0.1, response_text="answer")
    result = await router.complete("t")
    assert result != result and router.verifications_failed == 1 and maker.updates == []
    snapshot = await router.get_state_snapshot()
    assert snapshot["last_failure"] == "verification: RuntimeError: sandbox down"  # kept for the state endpoint, not logged


@pytest.mark.asyncio
async def test_expired_programs_are_evicted_when_the_table_is_full():
    now = [1000.0]
    router, maker = _router(clock=lambda: now[0], program_ttl_seconds=10, max_programs=3)
    for program_id in ("a", "b", "c"):
        await router.async_pre_routing_hook(
            model="oracle", request_kwargs={"metadata": {"program_id": program_id}}, messages=_messages()
        )
        now[0] += 4.0
    await router.async_pre_routing_hook(
        model="oracle", request_kwargs={"metadata": {"program_id": "d"}}, messages=_messages()
    )
    assert router.binding("a") is None and router.binding("b") is not None and router.binding("d") is not None
    assert router.programs_evicted == 1


@pytest.mark.asyncio
async def test_decision_is_stamped_in_the_proxy_internal_bucket_when_one_exists():
    router, _ = _router()
    kwargs = {"metadata": {"program_id": "t1"}, "litellm_metadata": {"user_api_key_hash": "k"}}
    response = await router.async_pre_routing_hook(model="oracle", request_kwargs=kwargs, messages=_messages())
    assert response.model == "smart"
    assert kwargs["litellm_metadata"][CHOSEN_MODEL_METADATA_KEY] == "smart"
    assert CHOSEN_MODEL_METADATA_KEY not in kwargs["metadata"]
    assert router.binding("t1", "k").api_key_hash == "k"


@pytest.mark.asyncio
async def test_the_same_program_id_under_another_key_is_another_program():
    router, maker = _router()
    for key in ("key-a", "key-b"):
        await router.async_pre_routing_hook(
            model="oracle",
            request_kwargs={"metadata": {"program_id": "t1", "user_api_key_hash": key}},
            messages=_messages(),
        )
    assert router.programs_bound == 2 and maker.selections == 2
    assert router.binding("t1") is None  # no key: not the same program either
    assert [binding.api_key_hash for binding in router.bindings_named("t1")] == ["key-a", "key-b"]
    router.observe_request("t1", cost=0.1, response_text="answer", owner="key-b")
    router.complete("t1", score=1.0, owner="key-b")
    assert router.binding("t1", "key-a") is not None and router.binding("t1", "key-b") is None
    await router.drain()


@pytest.mark.asyncio
async def test_completion_before_any_observed_response_is_deferred_to_the_next_one():
    router, maker = _router()
    await router.async_pre_routing_hook(
        model="oracle", request_kwargs={"metadata": {"program_id": "t"}}, messages=_messages()
    )
    assert router.complete("t", score=1.0, cost=2.0, payload={"run": "r1"}) is None  # nothing answered yet
    binding = router.binding("t")
    assert binding is not None and binding.deferred is not None and router.programs_completed == 0
    assert maker.updates == []
    router.observe_request("t", cost=0.3, response_text="answer")  # the response the feedback raced
    assert router.binding("t") is None and router.programs_completed == 1
    await router.drain()
    assert maker.updates == [("t", "smart", 1.0)]


@pytest.mark.asyncio
async def test_pre_routing_hook_replaces_caller_supplied_internal_keys():
    router, _ = _router()
    kwargs = {
        "metadata": {"program_id": "t1", PROGRAM_ID_METADATA_KEY: "victim", CHOSEN_MODEL_METADATA_KEY: "fast"},
        "litellm_metadata": {PROGRAM_ID_METADATA_KEY: "victim"},
    }
    response = await router.async_pre_routing_hook(model="oracle", request_kwargs=kwargs, messages=_messages())
    assert PROGRAM_ID_METADATA_KEY not in kwargs["metadata"] and CHOSEN_MODEL_METADATA_KEY not in kwargs["metadata"]
    assert kwargs["litellm_metadata"][PROGRAM_ID_METADATA_KEY] == "t1"
    assert kwargs["litellm_metadata"][CHOSEN_MODEL_METADATA_KEY] == response.model
    stateless = {"metadata": {PROGRAM_ID_METADATA_KEY: "victim", CHOSEN_MODEL_METADATA_KEY: "fast"}}
    await router.async_pre_routing_hook(model="oracle", request_kwargs=stateless, messages=_messages())
    assert PROGRAM_ID_METADATA_KEY not in stateless["metadata"]
    assert stateless["metadata"][CHOSEN_MODEL_METADATA_KEY] in ("smart", "fast")
