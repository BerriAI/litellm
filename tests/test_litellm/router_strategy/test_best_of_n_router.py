"""
Tests for the best-of-n router.

Covers the workflows an operator depends on, driven through the public
Router.acompletion entry point against mock deployments:
- synthesize mode: parallel fan-out, synthesizer answer returned, decision annotated.
- pick mode (tools present, or a candidate answered with tool_calls): judged
  candidate returned verbatim; judge faults fall back to the priority arm.
- degradation: a failed arm is dropped and recorded; all arms failing re-raises.
- synthesizer failure falls back to the highest-priority candidate.
- streaming for both modes.
- init validation: unresolvable arms and best_of_n cycles are rejected.
- internal-call metadata: children stamped with best-of-n origins; nested
  best-of-n calls refused.
"""

import asyncio

import pytest

import litellm
from litellm import Router
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_llm import CustomLLM as CustomLLMBase
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices
from litellm.utils import custom_llm_setup


@pytest.fixture(autouse=True)
def _isolate_custom_provider_globals(monkeypatch):
    monkeypatch.setattr(litellm, "custom_provider_map", list(litellm.custom_provider_map))
    monkeypatch.setattr(litellm, "provider_list", list(litellm.provider_list))
    monkeypatch.setattr(litellm, "_custom_providers", list(litellm._custom_providers))
    monkeypatch.setattr(litellm, "callbacks", list(litellm.callbacks))


def _mock_deployment(name: str, mock_response: str | Exception | ModelResponse) -> dict[str, object]:
    return {
        "model_name": name,
        "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test", "mock_response": mock_response},
    }


def _best_of_n_deployment(
    name: str, models: list[str | dict[str, object]], synthesizer: str | dict[str, object]
) -> dict[str, object]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"best_of_n/{name}",
            "best_of_n_config": {"models": models, "synthesizer": synthesizer},
        },
    }


def _tool_call_response() -> ModelResponse:
    return ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "sf"}'},
                        }
                    ],
                },
            }
        ]
    )


def _router(*extra_deployments: dict, judge_response: str = "Synthesized best answer") -> Router:
    return Router(
        model_list=[
            _mock_deployment("arm-a", "Answer from arm A"),
            _mock_deployment("arm-b", "Answer from arm B"),
            _mock_deployment("synth", judge_response),
            _best_of_n_deployment(
                "max-quality",
                [{"model_name": "arm-a", "litellm_params": {"reasoning_effort": "high"}}, "arm-b"],
                "synth",
            ),
            *extra_deployments,
        ]
    )


TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]


def test_synthesize_mode_returns_synthesizer_answer_with_decision_metadata():
    router = _router()
    response = asyncio.run(router.acompletion(model="max-quality", messages=[{"role": "user", "content": "hi"}]))
    assert response.choices[0].message.content == "Synthesized best answer"
    decision = response._hidden_params["best_of_n"]
    assert decision["mode"] == "synthesize"
    assert [c["model"] for c in decision["candidates"]] == ["arm-a", "arm-b"]
    assert decision["failed_arms"] == []
    assert response._hidden_params["response_cost"] == 0.0


def test_pick_mode_with_tools_returns_judged_candidate_verbatim():
    router = _router(judge_response='{"best": 2, "reason": "b wins"}')
    response = asyncio.run(
        router.acompletion(model="max-quality", messages=[{"role": "user", "content": "hi"}], tools=TOOLS)
    )
    assert response.choices[0].message.content == "Answer from arm B"
    decision = response._hidden_params["best_of_n"]
    assert decision["mode"] == "pick"
    assert decision["picked"] == 2
    assert "fallback_reason" not in decision


def test_candidate_tool_calls_force_pick_mode_and_survive_verbatim():
    router = Router(
        model_list=[
            _mock_deployment("arm-tools", _tool_call_response()),
            _mock_deployment("arm-b", "plain text answer"),
            _mock_deployment("synth", '{"best": 1, "reason": "tool call is right"}'),
            _best_of_n_deployment("mq", ["arm-tools", "arm-b"], "synth"),
        ]
    )
    response = asyncio.run(router.acompletion(model="mq", messages=[{"role": "user", "content": "hi"}]))
    assert response._hidden_params["best_of_n"]["mode"] == "pick"
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None and tool_calls[0].function.name == "get_weather"


def test_legacy_function_call_candidate_forces_pick_and_survives_verbatim():
    legacy = ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "function_call",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "get_weather", "arguments": '{"city": "sf"}'},
                },
            }
        ]
    )
    router = Router(
        model_list=[
            _mock_deployment("arm-legacy", legacy),
            _mock_deployment("arm-b", "plain text answer"),
            _mock_deployment("synth", '{"best": 1, "reason": "function call is right"}'),
            _best_of_n_deployment("mq", ["arm-legacy", "arm-b"], "synth"),
        ]
    )
    response = asyncio.run(router.acompletion(model="mq", messages=[{"role": "user", "content": "hi"}]))
    decision = response._hidden_params["best_of_n"]
    assert decision["mode"] == "pick"
    assert [c["model"] for c in decision["candidates"]] == ["arm-legacy", "arm-b"]
    assert response.choices[0].message.function_call.name == "get_weather"


def test_legacy_functions_request_param_forces_pick_mode():
    router = _router(judge_response='{"best": 2, "reason": "b wins"}')
    response = asyncio.run(
        router.acompletion(
            model="max-quality",
            messages=[{"role": "user", "content": "hi"}],
            functions=[{"name": "get_weather", "parameters": {}}],
        )
    )
    assert response._hidden_params["best_of_n"]["mode"] == "pick"


def test_parent_metadata_merges_identity_across_both_buckets():
    """Proxy identity keys can live in either metadata bucket; the resolver must merge the
    user_api_key* keys so children are never forwarded without the caller identity."""
    from litellm.router_strategy.best_of_n_router.best_of_n_router import _parent_metadata

    merged = _parent_metadata(
        {"litellm_metadata": {"model_group": "mq"}, "metadata": {"user_api_key": "hash-identity"}}
    )
    assert merged.get("user_api_key") == "hash-identity"
    assert merged.get("model_group") == "mq"


def test_arm_returning_no_choices_is_dropped_not_fatal():
    router = Router(
        model_list=[
            _mock_deployment("arm-a", "Answer from arm A"),
            _mock_deployment("arm-hollow", ModelResponse(choices=[])),
            _mock_deployment("synth", "Synthesized best answer"),
            _best_of_n_deployment("mq", ["arm-a", "arm-hollow"], "synth"),
        ]
    )
    response = asyncio.run(router.acompletion(model="mq", messages=[{"role": "user", "content": "hi"}]))
    decision = response._hidden_params["best_of_n"]
    assert [c["model"] for c in decision["candidates"]] == ["arm-a"]
    assert decision["failed_arms"] == [{"model": "arm-hollow", "error": "empty answer (finish_reason=no choices)"}]


def test_arm_level_timeout_and_retry_overrides_do_not_collide_with_call_keywords():
    router = Router(
        model_list=[
            _mock_deployment("arm-a", "Answer from arm A"),
            _mock_deployment("arm-b", "Answer from arm B"),
            _mock_deployment("synth", "Synthesized best answer"),
            _best_of_n_deployment(
                "mq",
                [{"model_name": "arm-a", "litellm_params": {"timeout": 30, "num_retries": 1}}, "arm-b"],
                {"model_name": "synth", "litellm_params": {"timeout": 45}},
            ),
        ]
    )
    response = asyncio.run(router.acompletion(model="mq", messages=[{"role": "user", "content": "hi"}]))
    assert response.choices[0].message.content == "Synthesized best answer"
    assert response._hidden_params["best_of_n"]["failed_arms"] == []


def test_judge_float_verdict_still_picks():
    router = _router(judge_response='{"best": 2.0, "reason": "b wins"}')
    response = asyncio.run(
        router.acompletion(model="max-quality", messages=[{"role": "user", "content": "hi"}], tools=TOOLS)
    )
    assert response.choices[0].message.content == "Answer from arm B"
    assert response._hidden_params["best_of_n"]["picked"] == 2


@pytest.mark.parametrize("judge_response", ["this is not json", '{"best": 99, "reason": "missing"}'])
def test_judge_fault_falls_back_to_highest_priority_arm(judge_response):
    router = _router(judge_response=judge_response)
    response = asyncio.run(
        router.acompletion(model="max-quality", messages=[{"role": "user", "content": "hi"}], tools=TOOLS)
    )
    assert response.choices[0].message.content == "Answer from arm A"
    assert "fallback_reason" in response._hidden_params["best_of_n"]


def test_failed_arm_is_dropped_and_recorded():
    router = Router(
        model_list=[
            _mock_deployment("arm-a", "Answer from arm A"),
            _mock_deployment("arm-dead", Exception("arm exploded")),
            _mock_deployment("synth", "Synthesized best answer"),
            _best_of_n_deployment("mq", ["arm-a", "arm-dead"], "synth"),
        ]
    )
    response = asyncio.run(router.acompletion(model="mq", messages=[{"role": "user", "content": "hi"}]))
    decision = response._hidden_params["best_of_n"]
    assert [c["model"] for c in decision["candidates"]] == ["arm-a"]
    assert [f["model"] for f in decision["failed_arms"]] == ["arm-dead"]


def test_every_arm_failing_reraises_the_arm_error():
    router = Router(
        model_list=[
            _mock_deployment("arm-dead", Exception("arm exploded")),
            _mock_deployment("arm-dead-2", Exception("arm exploded")),
            _mock_deployment("synth", "never reached"),
            _best_of_n_deployment("mq", ["arm-dead", "arm-dead-2"], "synth"),
        ]
    )
    with pytest.raises(Exception, match="arm exploded"):
        asyncio.run(router.acompletion(model="mq", messages=[{"role": "user", "content": "hi"}]))


def test_synthesizer_failure_falls_back_to_highest_priority_candidate():
    router = Router(
        model_list=[
            _mock_deployment("arm-a", "Answer from arm A"),
            _mock_deployment("arm-b", "Answer from arm B"),
            _mock_deployment("synth-dead", Exception("synthesizer exploded")),
            _best_of_n_deployment("mq", ["arm-a", "arm-b"], "synth-dead"),
        ]
    )
    response = asyncio.run(router.acompletion(model="mq", messages=[{"role": "user", "content": "hi"}]))
    assert response.choices[0].message.content == "Answer from arm A"
    assert "synthesizer failed" in response._hidden_params["best_of_n"]["fallback_reason"]


def test_streaming_synthesize_yields_the_synthesizer_stream():
    router = _router()

    async def _collect():
        stream = await router.acompletion(model="max-quality", messages=[{"role": "user", "content": "hi"}], stream=True)
        return [chunk async for chunk in stream]

    chunks = asyncio.run(_collect())
    text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert text == "Synthesized best answer"


def test_streaming_pick_replays_the_picked_candidate():
    router = _router(judge_response='{"best": 2, "reason": "b wins"}')

    async def _collect():
        stream = await router.acompletion(
            model="max-quality", messages=[{"role": "user", "content": "hi"}], stream=True, tools=TOOLS
        )
        return [chunk async for chunk in stream]

    chunks = asyncio.run(_collect())
    text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert text == "Answer from arm B"
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_streaming_pick_tool_calls_carry_stream_indexes():
    router = Router(
        model_list=[
            _mock_deployment("arm-tools", _tool_call_response()),
            _mock_deployment("arm-b", "plain text answer"),
            _mock_deployment("synth", '{"best": 1, "reason": "tool call is right"}'),
            _best_of_n_deployment("mq", ["arm-tools", "arm-b"], "synth"),
        ]
    )

    async def _collect():
        stream = await router.acompletion(model="mq", messages=[{"role": "user", "content": "hi"}], stream=True)
        return [chunk async for chunk in stream]

    chunks = asyncio.run(_collect())
    streamed_calls = [
        tc for chunk in chunks if chunk.choices for tc in (chunk.choices[0].delta.tool_calls or [])
    ]
    assert streamed_calls
    assert all(isinstance(tc.index, int) for tc in streamed_calls)


@pytest.mark.parametrize(
    "arms,expected",
    [
        (["mq", "arm-a"], "resolves to a best_of_n deployment"),
        (["ghost-model", "arm-a"], "does not resolve to any deployment"),
    ],
)
def test_invalid_configs_are_rejected_at_init(arms, expected):
    with pytest.raises(ValueError, match=expected):
        Router(
            model_list=[
                _mock_deployment("arm-a", "a"),
                _mock_deployment("arm-b", "b"),
                _mock_deployment("synth", "s"),
                _best_of_n_deployment("mq", arms, "synth"),
            ],
            ignore_invalid_deployments=False,
        )


def test_too_few_arms_rejected_at_init():
    with pytest.raises(ValueError, match="between 2 and 8 arms"):
        Router(
            model_list=[
                _mock_deployment("arm-a", "a"),
                _mock_deployment("synth", "s"),
                _best_of_n_deployment("mq", ["arm-a"], "synth"),
            ],
            ignore_invalid_deployments=False,
        )


class _MetadataRecorder(CustomLogger):
    def __init__(self):
        super().__init__()
        self.origins_by_group = {}  # mutable-ok: test capture buffer
        self.child_costs_by_group = {}  # mutable-ok: test capture buffer

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        metadata = (kwargs.get("litellm_params") or {}).get("metadata") or {}
        self.origins_by_group[metadata.get("model_group")] = metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY)
        self.child_costs_by_group[metadata.get("model_group")] = (
            getattr(response_obj, "_hidden_params", {}).get("response_cost") or kwargs.get("response_cost")
        )
        self.ids_by_group = {**getattr(self, "ids_by_group", {}), metadata.get("model_group"): getattr(response_obj, "id", None)}


def test_child_calls_carry_best_of_n_internal_origins(monkeypatch):
    recorder = _MetadataRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router = _router()
    asyncio.run(
        router.acompletion(
            model="max-quality",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"user_api_key": "hash-abc"},
        )
    )
    assert recorder.origins_by_group.get("arm-a") == "best_of_n_candidate"
    assert recorder.origins_by_group.get("arm-b") == "best_of_n_candidate"
    assert recorder.origins_by_group.get("synth") == "best_of_n_synthesizer"


def test_parent_zero_cost_never_reaches_the_childs_own_spend_row(monkeypatch):
    """The parent response is a zero-cost copy: zeroing the child's own object in place
    races the child's async cost callback and wipes its real spend (observed live)."""
    recorder = _MetadataRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router = _router()
    response = asyncio.run(router.acompletion(model="max-quality", messages=[{"role": "user", "content": "hi"}]))
    assert response._hidden_params["response_cost"] == 0.0
    assert recorder.child_costs_by_group.get("synth") not in (0.0, None)
    assert response.id != recorder.ids_by_group.get("synth")


def _empty_answer_response() -> ModelResponse:
    return ModelResponse(
        choices=[{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": None}}]
    )


def test_empty_candidate_is_dropped_like_a_failed_arm():
    router = Router(
        model_list=[
            _mock_deployment("arm-a", "Answer from arm A"),
            _mock_deployment("arm-empty", _empty_answer_response()),
            _mock_deployment("synth", "Synthesized best answer"),
            _best_of_n_deployment("mq", ["arm-a", "arm-empty"], "synth"),
        ]
    )
    response = asyncio.run(router.acompletion(model="mq", messages=[{"role": "user", "content": "hi"}]))
    decision = response._hidden_params["best_of_n"]
    assert [c["model"] for c in decision["candidates"]] == ["arm-a"]
    assert decision["failed_arms"] == [{"model": "arm-empty", "error": "empty answer (finish_reason=length)"}]


def test_empty_synthesizer_answer_falls_back_to_highest_priority_candidate():
    router = Router(
        model_list=[
            _mock_deployment("arm-a", "Answer from arm A"),
            _mock_deployment("arm-b", "Answer from arm B"),
            _mock_deployment("synth-empty", _empty_answer_response()),
            _best_of_n_deployment("mq", ["arm-a", "arm-b"], "synth-empty"),
        ]
    )
    response = asyncio.run(router.acompletion(model="mq", messages=[{"role": "user", "content": "hi"}]))
    assert response.choices[0].message.content == "Answer from arm A"
    assert "empty answer" in response._hidden_params["best_of_n"]["fallback_reason"]


def test_every_arm_empty_raises_instead_of_returning_nothing():
    router = Router(
        model_list=[
            _mock_deployment("arm-empty", _empty_answer_response()),
            _mock_deployment("arm-empty-2", _empty_answer_response()),
            _mock_deployment("synth", "never reached"),
            _best_of_n_deployment("mq", ["arm-empty", "arm-empty-2"], "synth"),
        ]
    )
    with pytest.raises(litellm.InternalServerError, match="empty answer"):
        asyncio.run(router.acompletion(model="mq", messages=[{"role": "user", "content": "hi"}]))


def test_second_router_does_not_hijack_the_first_routers_dispatch():
    """The provider map is process-global and holds only the newest handler, so dispatch
    resolves the owning handler from the call's deployment id instead of the map entry."""
    router_a = Router(
        model_list=[
            _mock_deployment("arm-a", "arm answer A"),
            _mock_deployment("arm-b", "arm answer A2"),
            _mock_deployment("synth", "synthesized by router A"),
            _best_of_n_deployment("mq", ["arm-a", "arm-b"], "synth"),
        ]
    )
    Router(
        model_list=[
            _mock_deployment("arm-a", "arm answer B"),
            _mock_deployment("arm-b", "arm answer B2"),
            _mock_deployment("synth", "synthesized by router B"),
            _best_of_n_deployment("mq", ["arm-a", "arm-b"], "synth"),
        ]
    )
    response = asyncio.run(router_a.acompletion(model="mq", messages=[{"role": "user", "content": "hi"}]))
    assert response.choices[0].message.content == "synthesized by router A"


class _LazyDeathStreamLLM(CustomLLMBase):
    async def astreaming(self, *args, **kwargs):
        raise RuntimeError("stream died on first pull")
        yield


class _NoOutputStreamLLM(CustomLLMBase):
    async def astreaming(self, model, *args, **kwargs):
        yield ModelResponseStream(model=model, choices=[StreamingChoices(index=0, delta=Delta(content=""))])
        yield ModelResponseStream(
            model=model, choices=[StreamingChoices(index=0, delta=Delta(), finish_reason="length")]
        )


def _router_with_stream_synthesizer(provider_name: str, handler: CustomLLMBase) -> Router:
    litellm.custom_provider_map.append({"provider": provider_name, "custom_handler": handler})
    custom_llm_setup()
    return Router(
        model_list=[
            _mock_deployment("arm-a", "Answer from arm A"),
            _mock_deployment("arm-b", "Answer from arm B"),
            {"model_name": "synth-stream", "litellm_params": {"model": f"{provider_name}/synth", "api_key": "sk-x"}},
            _best_of_n_deployment("mq", ["arm-a", "arm-b"], "synth-stream"),
        ]
    )


@pytest.mark.parametrize(
    "provider_name,handler",
    [("lazy_death_stream", _LazyDeathStreamLLM()), ("no_output_stream", _NoOutputStreamLLM())],
)
def test_streaming_falls_back_to_a_candidate_when_the_synthesizer_stream_yields_nothing_usable(
    provider_name, handler
):
    """Stream failures resolve lazily on iteration and an always-thinking model can stream
    only thinking chunks; both must replay the highest-priority candidate, matching the
    non-stream fallback."""
    router = _router_with_stream_synthesizer(provider_name, handler)

    async def _collect():
        stream = await router.acompletion(model="mq", messages=[{"role": "user", "content": "hi"}], stream=True)
        return [chunk async for chunk in stream]

    chunks = asyncio.run(_collect())
    text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert text == "Answer from arm A"


def test_router_helper_classification_and_entry_faults():
    from litellm.types.router import LiteLLM_Params

    router = _router()
    assert router._is_best_of_n_deployment(litellm_params=LiteLLM_Params(model="best_of_n/max-quality"))
    assert not router._is_best_of_n_deployment(litellm_params=LiteLLM_Params(model="openai/gpt-4o-mini"))
    assert router._best_of_n_entry_fault("ghost-model") == "does not resolve to any deployment on this router"
    assert router._best_of_n_entry_fault("max-quality") == "resolves to a best_of_n deployment, which would recurse"
    assert router._best_of_n_entry_fault("arm-a") is None


def test_reregistering_the_provider_binds_the_newest_handler():
    router = _router()
    replacement = router._register_best_of_n_provider()
    entries = [item["custom_handler"] for item in litellm.custom_provider_map if item["provider"] == "best_of_n"]
    assert entries == [replacement]


def test_init_rejects_a_duplicate_marker_name():
    from litellm.types.router import Deployment, LiteLLM_Params

    router = _router()
    duplicate = Deployment(
        model_name="max-quality-second",
        litellm_params=LiteLLM_Params(
            model="best_of_n/max-quality",
            best_of_n_config={"models": ["arm-a", "arm-b"], "synthesizer": "synth"},
        ),
    )
    with pytest.raises(ValueError, match="already registered"):
        router.init_best_of_n_deployment(deployment=duplicate)


def test_finalize_drops_a_faulty_config_when_invalid_deployments_are_ignored():
    from litellm.router_strategy.best_of_n_router.config import BestOfNRouterConfig

    router = Router(
        model_list=[
            _mock_deployment("arm-a", "a"),
            _mock_deployment("arm-b", "b"),
            _mock_deployment("synth", "s"),
            _best_of_n_deployment("mq", ["arm-a", "arm-b"], "synth"),
        ],
        ignore_invalid_deployments=True,
    )
    router.best_of_n_router.register(
        "broken", BestOfNRouterConfig.model_validate({"models": ["ghost-model", "arm-a"], "synthesizer": "synth"})
    )
    router._finalize_best_of_n_routers_if_configured()
    assert "broken" not in router.best_of_n_router.configs
    assert "mq" in router.best_of_n_router.configs


def test_streaming_parent_never_carries_positive_cost(monkeypatch):
    """The children carry the spend; the assembled parent stream must price to nothing.
    The marker's litellm model string matches no public cost-map key and custom pricing on
    strategy markers is stripped at cost-map registration, so a positive parent cost here
    would mean the caller is billed twice."""
    recorder = _MetadataRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router = _router()

    async def _collect():
        stream = await router.acompletion(model="max-quality", messages=[{"role": "user", "content": "hi"}], stream=True)
        return [chunk async for chunk in stream]

    asyncio.run(_collect())
    assert "max-quality" in recorder.child_costs_by_group
    assert not recorder.child_costs_by_group.get("max-quality")


def test_nested_best_of_n_call_is_refused():
    router = _router()
    with pytest.raises(litellm.BadRequestError, match="inside another best-of-n request"):
        asyncio.run(
            router.acompletion(
                model="max-quality",
                messages=[{"role": "user", "content": "hi"}],
                metadata={INTERNAL_CALL_ORIGIN_METADATA_KEY: "best_of_n_candidate"},
            )
        )
