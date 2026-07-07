from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from litellm.constants import (
    ANTHROPIC_MIN_THINKING_BUDGET_TOKENS,
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
)

OMIT = object()


@dataclass(frozen=True)
class CellExpectation:
    status: int
    thinking_type: object
    output_config_effort: object = OMIT
    thinking_budget_tokens: object = OMIT
    max_tokens: object = OMIT


@dataclass(frozen=True)
class ModelEntry:
    alias: str
    model: str
    mode: str
    extra_params: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    required_env: FrozenSet[str] = field(default_factory=frozenset)
    caps: FrozenSet[str] = field(default_factory=frozenset)
    unavailable_error: Optional[str] = None
    fail_reason: Optional[str] = None
    bedrock_effort_ceiling: Optional[str] = None

    def params(self) -> Dict[str, str]:
        return dict(self.extra_params)


EFFORTS: Tuple[str, ...] = (
    "__omit__",
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "disabled",
    "invalid",
    "",
)

_BUDGET_TOKENS: Dict[str, int] = {
    "minimal": max(
        DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
        ANTHROPIC_MIN_THINKING_BUDGET_TOKENS,
    ),
    "low": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    "medium": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    "high": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    "xhigh": DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
    "max": DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
}

_ADAPTIVE_EFFORT_LABEL: Dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

_EFFORT_RANK: Dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "max": 3,
    "xhigh": 4,
}

_BAD_REQUEST_EFFORTS: FrozenSet[str] = frozenset({"disabled", "invalid", ""})

# Live providers reject ``max_tokens <= thinking.budget_tokens``, so a budget-mode
# request must leave room above the largest 200-expected tier (``high``).
BUDGET_MODE_MAX_TOKENS: int = DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET * 2


def _bedrock_clamps_effort(model: "ModelEntry", effort: str) -> bool:
    """Whether Bedrock will clamp ``effort`` down to ``bedrock_effort_ceiling``.

    Bedrock chat/messages paths clamp unsupported high tiers (e.g. ``xhigh``
    on Opus 4.6) to the model's ceiling rather than rejecting them, so the
    missing native capability is OK — the wire effort just degrades.
    """
    if model.bedrock_effort_ceiling is None:
        return False
    if effort not in _EFFORT_RANK or model.bedrock_effort_ceiling not in _EFFORT_RANK:
        return False
    return _EFFORT_RANK[effort] > _EFFORT_RANK[model.bedrock_effort_ceiling]


def expected(model: ModelEntry, effort: str) -> CellExpectation:
    if effort in ("__omit__", "none"):
        if model.mode == "budget":
            return CellExpectation(
                status=200, thinking_type=OMIT, max_tokens=BUDGET_MODE_MAX_TOKENS
            )
        return CellExpectation(status=200, thinking_type=OMIT)

    if effort in _BAD_REQUEST_EFFORTS:
        return CellExpectation(status=400, thinking_type=OMIT)

    if effort in ("xhigh", "max"):
        cap = f"supports_{effort}_reasoning_effort"
        if cap not in model.caps and not _bedrock_clamps_effort(model, effort):
            return CellExpectation(status=400, thinking_type=OMIT)

    if model.mode == "adaptive":
        wire_effort = _ADAPTIVE_EFFORT_LABEL[effort]
        if model.bedrock_effort_ceiling is not None:
            wire_rank = _EFFORT_RANK[wire_effort]
            ceiling_rank = _EFFORT_RANK[model.bedrock_effort_ceiling]
            if wire_rank > ceiling_rank:
                wire_effort = model.bedrock_effort_ceiling
        return CellExpectation(
            status=200,
            thinking_type="adaptive",
            output_config_effort=wire_effort,
        )

    return CellExpectation(
        status=200,
        thinking_type="enabled",
        thinking_budget_tokens=_BUDGET_TOKENS[effort],
        max_tokens=BUDGET_MODE_MAX_TOKENS,
    )


_ANTHROPIC_REQ = frozenset({"ANTHROPIC_API_KEY"})
_AZURE_FOUNDRY_REQ = frozenset({"AZURE_FOUNDRY_API_BASE", "AZURE_FOUNDRY_API_KEY"})
_VERTEX_REQ = frozenset({"VERTEX_PROJECT"})
_BEDROCK_REQ = frozenset({"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"})


_CAPS_XHIGH_MAX: FrozenSet[str] = frozenset(
    {"supports_xhigh_reasoning_effort", "supports_max_reasoning_effort"}
)
_CAPS_4_6: FrozenSet[str] = frozenset({"supports_max_reasoning_effort"})
_CAPS_NONE: FrozenSet[str] = frozenset()


ANTHROPIC_DIRECT_MODELS: Tuple[ModelEntry, ...] = (
    ModelEntry(
        alias="claude-fable-5",
        model="anthropic/claude-fable-5",
        mode="adaptive",
        required_env=_ANTHROPIC_REQ,
        caps=_CAPS_XHIGH_MAX,
        fail_reason=(
            "claude-fable-5 is not yet released on the Anthropic API for the CI "
            "account; Anthropic returns not_found_error until the model is "
            "available, so this cell stays loud in CI. Remove this fail_reason "
            "once the model is available."
        ),
    ),
    ModelEntry(
        alias="claude-opus-4-8",
        model="anthropic/claude-opus-4-8",
        mode="adaptive",
        required_env=_ANTHROPIC_REQ,
        caps=_CAPS_XHIGH_MAX,
    ),
    ModelEntry(
        alias="claude-opus-4-7",
        model="anthropic/claude-opus-4-7",
        mode="adaptive",
        required_env=_ANTHROPIC_REQ,
        caps=_CAPS_XHIGH_MAX,
    ),
    ModelEntry(
        alias="claude-sonnet-5",
        model="anthropic/claude-sonnet-5",
        mode="adaptive",
        required_env=_ANTHROPIC_REQ,
        caps=_CAPS_XHIGH_MAX,
    ),
    ModelEntry(
        alias="claude-sonnet-4-6",
        model="anthropic/claude-sonnet-4-6",
        mode="adaptive",
        required_env=_ANTHROPIC_REQ,
        caps=_CAPS_4_6,
    ),
    ModelEntry(
        alias="claude-haiku-4-5",
        model="anthropic/claude-haiku-4-5",
        mode="budget",
        required_env=_ANTHROPIC_REQ,
        caps=_CAPS_NONE,
    ),
)


AZURE_AI_MODELS: Tuple[ModelEntry, ...] = (
    ModelEntry(
        alias="azure-claude-fable-5",
        model="azure_ai/claude-fable-5",
        mode="adaptive",
        required_env=_AZURE_FOUNDRY_REQ,
        caps=_CAPS_XHIGH_MAX,
        fail_reason=(
            "claude-fable-5 has no deployment on the CI Microsoft Foundry "
            "resource yet; Foundry returns DeploymentNotFound until someone "
            "creates the fable-5 deployment, so this cell stays loud in CI. "
            "Remove this fail_reason once the deployment exists."
        ),
    ),
    ModelEntry(
        alias="azure-claude-opus-4-8",
        model="azure_ai/claude-opus-4-8",
        mode="adaptive",
        required_env=_AZURE_FOUNDRY_REQ,
        caps=_CAPS_XHIGH_MAX,
        fail_reason=(
            "claude-opus-4-8 has no deployment on the CI Microsoft Foundry "
            "resource yet; Foundry returns DeploymentNotFound until someone "
            "creates the opus-4-8 deployment, so this cell stays loud in CI. "
            "Remove this fail_reason once the deployment exists."
        ),
    ),
    ModelEntry(
        alias="azure-claude-opus-4-7",
        model="azure_ai/claude-opus-4-7",
        mode="adaptive",
        required_env=_AZURE_FOUNDRY_REQ,
        caps=_CAPS_XHIGH_MAX,
    ),
    ModelEntry(
        alias="azure-claude-opus-4-6",
        model="azure_ai/claude-opus-4-6",
        mode="adaptive",
        required_env=_AZURE_FOUNDRY_REQ,
        caps=_CAPS_4_6,
    ),
    ModelEntry(
        alias="azure-claude-sonnet-4-6",
        model="azure_ai/claude-sonnet-4-6",
        mode="adaptive",
        required_env=_AZURE_FOUNDRY_REQ,
        caps=_CAPS_4_6,
    ),
    ModelEntry(
        alias="azure-claude-haiku-4-5",
        model="azure_ai/claude-haiku-4-5",
        mode="budget",
        required_env=_AZURE_FOUNDRY_REQ,
        caps=_CAPS_NONE,
    ),
)


VERTEX_AI_MODELS: Tuple[ModelEntry, ...] = (
    ModelEntry(
        alias="vertex-claude-fable-5",
        model="vertex_ai/claude-fable-5",
        mode="adaptive",
        extra_params=(("vertex_location", "global"),),
        required_env=_VERTEX_REQ,
        caps=_CAPS_XHIGH_MAX,
        fail_reason=(
            "claude-fable-5 availability on the CI Vertex project is not yet "
            "confirmed for this brand-new release, so this cell stays loud in "
            "CI until verified. Remove this fail_reason once the model is "
            "confirmed available on the global Vertex endpoint."
        ),
    ),
    ModelEntry(
        alias="vertex-claude-opus-4-8",
        model="vertex_ai/claude-opus-4-8",
        mode="adaptive",
        extra_params=(("vertex_location", "global"),),
        required_env=_VERTEX_REQ,
        caps=_CAPS_XHIGH_MAX,
        fail_reason=(
            "claude-opus-4-8 availability on the CI Vertex project is not yet "
            "confirmed for this brand-new release, so this cell stays loud in "
            "CI until verified. Remove this fail_reason once the model is "
            "confirmed available on the global Vertex endpoint."
        ),
    ),
    ModelEntry(
        alias="vertex-claude-opus-4-7",
        model="vertex_ai/claude-opus-4-7",
        mode="adaptive",
        extra_params=(("vertex_location", "global"),),
        required_env=_VERTEX_REQ,
        caps=_CAPS_XHIGH_MAX,
    ),
    ModelEntry(
        alias="vertex-claude-opus-4-6",
        model="vertex_ai/claude-opus-4-6",
        mode="adaptive",
        extra_params=(("vertex_location", "us-east5"),),
        required_env=_VERTEX_REQ,
        caps=_CAPS_4_6,
    ),
    ModelEntry(
        alias="vertex-claude-sonnet-4-6",
        model="vertex_ai/claude-sonnet-4-6",
        mode="adaptive",
        extra_params=(("vertex_location", "us-east5"),),
        required_env=_VERTEX_REQ,
        caps=_CAPS_4_6,
    ),
    ModelEntry(
        alias="vertex-claude-haiku-4-5",
        model="vertex_ai/claude-haiku-4-5",
        mode="budget",
        extra_params=(("vertex_location", "us-east5"),),
        required_env=_VERTEX_REQ,
        caps=_CAPS_NONE,
    ),
)


BEDROCK_CONVERSE_MODELS: Tuple[ModelEntry, ...] = (
    ModelEntry(
        alias="bedrock-claude-fable-5",
        model="bedrock/converse/us.anthropic.claude-fable-5",
        mode="adaptive",
        extra_params=(("aws_region_name", "us-east-1"),),
        required_env=_BEDROCK_REQ,
        caps=_CAPS_XHIGH_MAX,
        bedrock_effort_ceiling="xhigh",
        unavailable_error="is not available for this account",
        fail_reason=(
            "claude-fable-5 on Bedrock requires the account to opt in to "
            "provider data sharing (data retention mode "
            "'provider_data_sharing' via the Data Retention API); the CI "
            "account has not opted in yet, so this cell stays loud in CI. "
            "Remove this fail_reason once the opt-in is done."
        ),
    ),
    ModelEntry(
        alias="bedrock-claude-opus-4-8",
        model="bedrock/converse/us.anthropic.claude-opus-4-8",
        mode="adaptive",
        extra_params=(("aws_region_name", "us-east-1"),),
        required_env=_BEDROCK_REQ,
        caps=_CAPS_XHIGH_MAX,
        bedrock_effort_ceiling="xhigh",
        unavailable_error="is not available for this account",
    ),
    ModelEntry(
        alias="bedrock-claude-opus-4-7",
        model="bedrock/converse/us.anthropic.claude-opus-4-7",
        mode="adaptive",
        extra_params=(("aws_region_name", "us-east-1"),),
        required_env=_BEDROCK_REQ,
        caps=_CAPS_XHIGH_MAX,
        bedrock_effort_ceiling="xhigh",
        unavailable_error="is not available for this account",
    ),
    ModelEntry(
        alias="bedrock-claude-opus-4-6",
        model="bedrock/converse/us.anthropic.claude-opus-4-6-v1",
        mode="adaptive",
        extra_params=(("aws_region_name", "us-east-1"),),
        required_env=_BEDROCK_REQ,
        caps=_CAPS_4_6,
        bedrock_effort_ceiling="max",
    ),
    ModelEntry(
        alias="bedrock-claude-sonnet-4-6",
        model="bedrock/converse/us.anthropic.claude-sonnet-4-6",
        mode="adaptive",
        extra_params=(("aws_region_name", "us-east-1"),),
        required_env=_BEDROCK_REQ,
        caps=_CAPS_4_6,
    ),
    ModelEntry(
        alias="bedrock-claude-sonnet-4-5",
        model="bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        mode="budget",
        extra_params=(("aws_region_name", "us-east-1"),),
        required_env=_BEDROCK_REQ,
        caps=_CAPS_NONE,
    ),
)


BEDROCK_INVOKE_CHAT_MODELS: Tuple[ModelEntry, ...] = (
    ModelEntry(
        alias="bedrock-invoke-claude-opus-4-6",
        model="bedrock/invoke/us.anthropic.claude-opus-4-6-v1",
        mode="adaptive",
        extra_params=(("aws_region_name", "us-east-1"),),
        required_env=_BEDROCK_REQ,
        caps=_CAPS_4_6,
        bedrock_effort_ceiling="max",
    ),
    ModelEntry(
        alias="bedrock-invoke-claude-sonnet-4-6",
        model="bedrock/invoke/us.anthropic.claude-sonnet-4-6",
        mode="adaptive",
        extra_params=(("aws_region_name", "us-east-1"),),
        required_env=_BEDROCK_REQ,
        caps=_CAPS_4_6,
    ),
    ModelEntry(
        alias="bedrock-invoke-claude-opus-4-5",
        model="bedrock/invoke/us.anthropic.claude-opus-4-5-20251101-v1:0",
        mode="budget",
        extra_params=(("aws_region_name", "us-east-1"),),
        required_env=_BEDROCK_REQ,
        caps=_CAPS_NONE,
    ),
)


BEDROCK_INVOKE_MESSAGES_MODELS: Tuple[ModelEntry, ...] = BEDROCK_INVOKE_CHAT_MODELS


@dataclass(frozen=True)
class Route:
    name: str
    models: Tuple[ModelEntry, ...]


ROUTES: Tuple[Route, ...] = (
    Route("anthropic_direct", ANTHROPIC_DIRECT_MODELS),
    Route("azure_ai", AZURE_AI_MODELS),
    Route("vertex_ai", VERTEX_AI_MODELS),
    Route("bedrock_converse", BEDROCK_CONVERSE_MODELS),
    Route("bedrock_invoke_chat", BEDROCK_INVOKE_CHAT_MODELS),
    Route("bedrock_invoke_messages", BEDROCK_INVOKE_MESSAGES_MODELS),
)


def all_cells() -> List[Tuple[str, ModelEntry, str, CellExpectation]]:
    cells: List[Tuple[str, ModelEntry, str, CellExpectation]] = []
    for route in ROUTES:
        for model in route.models:
            for effort in EFFORTS:
                cells.append((route.name, model, effort, expected(model, effort)))
    return cells
