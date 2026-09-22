from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, cast  # noqa: TID251  # public SDK callable is narrowed to its protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from litellm.fusion_router import FusionCompletionCaller, build_fusion_router
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

DEFAULT_FUSION_PANEL_MODELS: Final = (
    "openai/gpt-5.6-terra",
    "anthropic/claude-sonnet-5",
)
DEFAULT_FUSION_JUDGE_MODEL: Final = "openai/gpt-5.6-sol"
FUSION_SDK_MODEL: Final = "litellm/fusion-1"

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class FusionJudgeConfig(BaseModel):
    model: str = Field(default=DEFAULT_FUSION_JUDGE_MODEL, min_length=1)
    criteria: str | None = Field(default=None, min_length=1)

    model_config = ConfigDict(extra="forbid", frozen=True)


class FusionSDKConfig(BaseModel):
    models: tuple[str, ...] = Field(default=DEFAULT_FUSION_PANEL_MODELS, min_length=1, max_length=8)
    judge: FusionJudgeConfig = Field(default_factory=FusionJudgeConfig)
    max_tool_calls: int = Field(default=4, ge=1, le=16)
    max_completion_tokens: int = Field(default=16000, ge=1, le=128000)
    reasoning: ReasoningEffort | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_models(self) -> FusionSDKConfig:
        configured_models: Final = (*self.models, self.judge.model)
        if any(not model.strip() for model in configured_models):
            raise ValueError("Fusion model names must not be empty")
        if any(model == FUSION_SDK_MODEL for model in configured_models):
            raise ValueError(f"{FUSION_SDK_MODEL} cannot be a panel or judge model")
        return self

    def router_config(self) -> Mapping[str, object]:
        return {  # mutable-ok: FusionRouter accepts a mapping and validates it immediately
            "outer_model": self.judge.model,
            "panel_models": self.models,
            "analyst_model": self.judge.model,
            "analyst_criteria": self.judge.criteria,
            "invocation": "required",
            "max_tool_calls": self.max_tool_calls,
            "max_completion_tokens": self.max_completion_tokens,
            "reasoning_effort": self.reasoning,
            "temperature": self.temperature,
        }


async def _call_completion(
    *,
    model: str,
    messages: list[AllMessageValues],  # mutable-ok: public SDK boundary
    stream: bool,
    **kwargs: object,  # kwargs-ok: preserves the public completion parameter surface
) -> ModelResponse | CustomStreamWrapper:
    import litellm

    completion: Final = cast(  # cast-ok: public acompletion matches FusionCompletionCaller
        FusionCompletionCaller, litellm.acompletion
    )
    forwarded_kwargs: Final = {key: value for key, value in kwargs.items() if key != "_fusion_depth"}
    return await completion(
        model=model,
        messages=messages,
        stream=stream,
        **forwarded_kwargs,
    )


class FusionLiteLLMModel:
    def __init__(self, completion: FusionCompletionCaller = _call_completion) -> None:
        self._completion: Final = completion

    async def acompletion(
        self,
        *,
        messages: list[AllMessageValues],  # mutable-ok: public SDK boundary
        stream: bool,
        request_kwargs: Mapping[str, object],
    ) -> ModelResponse | CustomStreamWrapper:
        if request_kwargs.get("_fusion_depth"):
            raise ValueError(f"{FUSION_SDK_MODEL} cannot recursively invoke itself")
        config: Final = FusionSDKConfig.model_validate(
            request_kwargs.get("fusion") or {}  # mutable-ok: pydantic validates and freezes this input
        )
        router: Final = build_fusion_router(
            model_name=FUSION_SDK_MODEL,
            raw_config=config.router_config(),
            completion=self._completion,
        )
        forwarded_kwargs: Final = {  # mutable-ok: FusionRouter accepts an isolated request mapping
            key: value
            for key, value in request_kwargs.items()
            if value is not None
            and key not in frozenset(("fusion", "messages", "model", "stream", "acompletion", "aresponses"))
        }
        return await router.acompletion(
            messages=messages,
            stream=stream,
            request_kwargs=forwarded_kwargs,
        )
