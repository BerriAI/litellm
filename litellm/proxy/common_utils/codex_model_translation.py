"""Codex's `ModelInfo` entry for a model Codex does not know, translated from what LiteLLM knows about the
deployments behind it.

Field names, types and enum values follow `codex-rs/protocol/src/openai_models.rs` at rust-v0.166.0, and every
field a release back to 0.140.0 decodes without a default is spelled out, so one body parses on all of them.
Codex rejects the whole catalog when one entry fails to decode, so only enum values every one of those
releases knows are sent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from types import NoneType
from typing import Annotated, Final, Literal, TypeAlias

from packaging.version import Version
from pydantic import ConfigDict, Field, StringConstraints, field_serializer

from litellm.litellm_core_utils.reasoning_effort_strength import nearest_declared_reasoning_effort
from litellm.types.llms.base_model import LiteLLMBaseModel

TierId: TypeAlias = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CodexInputModality: TypeAlias = Literal["text", "image"]

_TEXT_ONLY: Final[tuple[CodexInputModality, ...]] = ("text",)
_TEXT_AND_IMAGE: Final[tuple[CodexInputModality, ...]] = ("text", "image")
TOKEN_PRICE_KEY: Final = "input_cost_per_token"
_FIRST_CODEX_WITH_CUSTOM_EFFORTS: Final = Version("0.138.0")
_EFFORTS_EVERY_CODEX_PARSES: Final = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
_PREFERRED_DEFAULT_EFFORT: Final = "medium"
_CODEX_UNSET_EFFORT: Final = "none"


def _unset(value: object) -> bool:
    return value is None


class CodexServiceTier(LiteLLMBaseModel):
    """A `ModelServiceTier` as Codex reads it: the slash command is `name` lowercased, and toggling
    it sends `id` as the request's `service_tier`."""

    model_config = ConfigDict(frozen=True)

    id: TierId
    name: str
    description: str


class CodexFallbackReasoningLevel(LiteLLMBaseModel):
    """A `ReasoningEffortPreset` as Codex reads it: picking it sends `effort` as the request's reasoning effort."""

    model_config = ConfigDict(frozen=True)

    effort: str
    description: str


class CodexTruncationPolicy(LiteLLMBaseModel):
    """How Codex truncates a long tool output before sending it back, the policy its own entries use."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["tokens"] = "tokens"
    limit: int = 10_000


class CodexFallbackModel(LiteLLMBaseModel):
    """One `ModelInfo` entry of a Codex model catalog for a model Codex does not know.

    `supports_reasoning_summaries` and `supports_parallel_tool_calls` are read only by Codex before 0.150.0,
    which requires them, and `base_instructions` is the prompt field every release still decodes.
    """

    model_config = ConfigDict(frozen=True)

    slug: str
    display_name: str
    description: NoneType = None
    default_reasoning_level: str | None = Field(default=None, exclude_if=_unset)
    supported_reasoning_levels: tuple[CodexFallbackReasoningLevel, ...] = ()
    shell_type: Literal["unified_exec"] = "unified_exec"
    visibility: Literal["list"] = "list"
    supported_in_api: Literal[True] = True
    priority: int
    service_tiers: tuple[CodexServiceTier, ...] = ()
    availability_nux: NoneType = None
    upgrade: NoneType = None
    support_verbosity: Literal[False] = False
    supports_reasoning_summary_parameter: bool | None = Field(default=None, exclude_if=_unset)
    supports_reasoning_summaries: bool = False
    supports_parallel_tool_calls: bool = False
    default_verbosity: NoneType = None
    apply_patch_tool_type: Literal["freeform"] | None = None
    truncation_policy: CodexTruncationPolicy = CodexTruncationPolicy()
    experimental_supported_tools: tuple[()] = ()
    input_modalities: tuple[CodexInputModality, ...] = _TEXT_AND_IMAGE
    context_window: int | None
    base_instructions: str
    supports_search_tool: bool = False
    use_responses_lite: Literal[False] = False


class CodexDeploymentCapabilities(LiteLLMBaseModel):
    """What one deployment's model info says about the capabilities a Codex entry describes, None where
    it says nothing, whether it has a token price at all, and the input price keys of the service tiers it
    has a price for."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    max_input_tokens: int | None = None
    supports_vision: bool | None = None
    supports_function_calling: bool | None = None
    supports_parallel_function_calling: bool | None = None
    supports_reasoning: bool | None = None
    prices_tokens: bool = False
    priced_service_tier_keys: frozenset[str] = frozenset()

    @field_serializer("priced_service_tier_keys")
    def _sorted_priced_service_tier_keys(self, keys: frozenset[str]) -> tuple[str, ...]:
        return tuple(sorted(keys))


@cache
def tier_price_keys() -> tuple[str, ...]:
    """The input price key of every service tier cost tracking prices. Cost tracking is imported on first use
    because `lite codex` imports this module before the SDK is initialized, and cost tracking imports the rest
    of the SDK."""
    from litellm.litellm_core_utils.llm_cost_calc.utils import SERVICE_TIER_COST_KEY_SUFFIXES

    return tuple(f"{TOKEN_PRICE_KEY}{suffix}" for suffix in SERVICE_TIER_COST_KEY_SUFFIXES)


def no_priced_deployment_lacks_tier(tier_id: str, deployments: Sequence[CodexDeploymentCapabilities]) -> bool:
    """Whether every deployment that has a token price also has one for `tier_id`, the way cost tracking
    prices a request sent at that tier. A deployment with no token price at all (a subscription upstream,
    a deployment name the cost map does not know) says nothing about tiers, so it does not vote, and a tier
    cost tracking has no price key for is lacked by every deployment that prices tokens."""
    from litellm.litellm_core_utils.llm_cost_calc.utils import get_service_tier_cost_key

    price_key: Final = get_service_tier_cost_key(TOKEN_PRICE_KEY, tier_id)
    return all(
        price_key in deployment.priced_service_tier_keys for deployment in deployments if deployment.prices_tokens
    )


@dataclass(frozen=True, slots=True)
class CodexCapabilities:
    """What a request to a listed name can count on whichever deployment serves it: a capability is False
    when one deployment lacks it, True when one has it and none lacks it, None when none says, and the
    context window is the smallest one any deployment declares."""

    context_window: int | None = None
    supports_vision: bool | None = None
    supports_function_calling: bool | None = None
    supports_parallel_function_calling: bool | None = None
    supports_reasoning: bool | None = None


def _combined(flags: Iterable[bool | None]) -> bool | None:
    said: Final = frozenset(flag for flag in flags if flag is not None)
    return None if not said else False not in said


def combined_capabilities(
    deployments: Sequence[CodexDeploymentCapabilities], listed_context_window: int | None = None
) -> CodexCapabilities:
    """The capabilities every one of `deployments` offers, the window falling back to
    `listed_context_window` when no deployment declares one."""
    windows: Final = tuple(
        deployment.max_input_tokens
        for deployment in deployments
        if deployment.max_input_tokens is not None and deployment.max_input_tokens > 0
    )
    return CodexCapabilities(
        context_window=min(windows) if windows else listed_context_window,
        supports_vision=_combined(deployment.supports_vision for deployment in deployments),
        supports_function_calling=_combined(deployment.supports_function_calling for deployment in deployments),
        supports_parallel_function_calling=_combined(
            deployment.supports_parallel_function_calling for deployment in deployments
        ),
        supports_reasoning=_combined(deployment.supports_reasoning for deployment in deployments),
    )


def _parses_custom_efforts(client_version: str | None) -> bool:
    try:
        return client_version is not None and Version(client_version) >= _FIRST_CODEX_WITH_CUSTOM_EFFORTS
    except ValueError:
        return False


def codex_reasoning_levels(
    efforts: Sequence[str], client_version: str | None
) -> tuple[tuple[CodexFallbackReasoningLevel, ...], str | None]:
    """The reasoning levels a fallback entry offers for a model group's efforts, and its default level.

    Codex before 0.138.0 fails the whole catalog on an effort it has no name for, so for it, and for
    a version that does not parse, only the efforts every release knows are offered. The default is
    set whenever a level is offered, because Codex reads a missing one as `none`: `none` itself when the
    group accepts it, so a user who picks no level sends what Codex sent before, else medium, else the
    nearest weaker declared level, so picking no level never costs more reasoning than Codex's usual medium,
    else the weakest declared level, else the first one."""
    parses_any: Final = _parses_custom_efforts(client_version)
    offered: Final = tuple(effort for effort in efforts if parses_any or effort in _EFFORTS_EVERY_CODEX_PARSES)
    if not offered:
        return (), None
    nearest: Final = nearest_declared_reasoning_effort(_PREFERRED_DEFAULT_EFFORT, offered, "down")
    levels: Final = tuple(
        CodexFallbackReasoningLevel(effort=effort, description=f"Sends reasoning_effort={effort} upstream")
        for effort in offered
    )
    if _CODEX_UNSET_EFFORT in offered:
        return levels, _CODEX_UNSET_EFFORT
    return levels, nearest if nearest in offered else offered[0]


def codex_fallback_entry(
    *,
    slug: str,
    display_name: str,
    priority: int,
    capabilities: CodexCapabilities,
    instructions: str,
    service_tiers: tuple[CodexServiceTier, ...] = (),
    supports_search_tool: bool = False,
    reasoning_efforts: Sequence[str] = (),
    client_version: str | None = None,
) -> CodexFallbackModel:
    """Codex's entry for a model it does not know, from the capabilities its deployments share.

    A capability no deployment declares keeps Codex's own default for an unknown model, except the freeform
    `apply_patch` tool, which goes out only to a model that calls tools: `supports_reasoning_summary_parameter`
    is left out, which Codex reads as true, and the fields only Codex before 0.150.0 reads keep the false
    it requires.
    """
    levels, default_level = codex_reasoning_levels(reasoning_efforts, client_version)
    return CodexFallbackModel(
        slug=slug,
        display_name=display_name,
        default_reasoning_level=default_level,
        supported_reasoning_levels=levels,
        priority=priority,
        service_tiers=service_tiers,
        supports_reasoning_summary_parameter=capabilities.supports_reasoning,
        supports_reasoning_summaries=capabilities.supports_reasoning is True,
        supports_parallel_tool_calls=capabilities.supports_parallel_function_calling is True,
        apply_patch_tool_type="freeform" if capabilities.supports_function_calling is True else None,
        input_modalities=_TEXT_ONLY if capabilities.supports_vision is False else _TEXT_AND_IMAGE,
        context_window=capabilities.context_window,
        base_instructions=instructions,
        supports_search_tool=supports_search_tool,
    )
