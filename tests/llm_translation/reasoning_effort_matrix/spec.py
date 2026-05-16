"""Spec + adapters for the ``reasoning_effort`` wire-translation matrix.

This module is the single source of truth the parametrized matrix test in
``test_reasoning_effort_wire_matrix.py`` iterates over. It is deliberately
*independent* of the production mapping code: the expected subtree per
``(tier, effort)`` is hand-authored from the documented contract (see
PRs #27039 / #27074), not derived from ``model_prices`` or the transformation
modules. That independence is what lets the matrix catch data-layer bugs
(e.g. a wrong capability flag) as well as code-layer ones.

What each cell asserts:

* **Wire cells** — LiteLLM must build a request whose reasoning subtree
  (``thinking`` + ``output_config``) matches the expected value *and*
  presence/absence exactly. Captured fully offline (no HTTP, no creds) via
  the established per-provider transformation idiom.
* **Client-400 cells** — LiteLLM must reject the request *before any wire
  call* with a clean 400-class exception (``BadRequestError`` /
  ``AnthropicError`` @400), never a bare ``ValueError`` (the #27074 bug).

The provider-acceptance half (does Anthropic 200 the request we built) is a
separate, CI-gated, VCR-recorded test — see the matrix test module.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import litellm
from litellm.llms.anthropic.common_utils import AnthropicError

# --------------------------------------------------------------------------- #
# Efforts
# --------------------------------------------------------------------------- #

# Sentinel: send NO ``reasoning_effort`` field at all (the "(omit)" sweep row).
OMIT = "__omit__"

# Ordered exactly as the QA sweep grid columns.
EFFORTS: Tuple[str, ...] = (
    OMIT,
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "disabled",
    "invalid",
    "",  # empty string — must be rejected, must NOT leak to the wire
)


def effort_label(effort: str) -> str:
    """Stable, filesystem/pytest-id-safe label for an effort value."""
    if effort == OMIT:
        return "OMIT"
    if effort == "":
        return "EMPTY"
    return effort


# --------------------------------------------------------------------------- #
# Tiers + model -> tier map (HAND-AUTHORED — independent of model_prices)
# --------------------------------------------------------------------------- #

ADAPTIVE_FULL = "adaptive_full"  # adaptive thinking; xhigh AND max accepted
ADAPTIVE_MAX_ONLY = "adaptive_max_only"  # adaptive; max accepted, xhigh -> 400
BUDGET = "budget"  # non-adaptive; thinking={enabled, budget_tokens}, no output_config

TIERS: Tuple[str, ...] = (ADAPTIVE_FULL, ADAPTIVE_MAX_ONLY, BUDGET)

# Canonical model key -> tier. The canonical key is the substring the
# production ``_is_claude_4_x_model`` family checks key on.
MODEL_TIER: Dict[str, str] = {
    "opus-4-7": ADAPTIVE_FULL,
    "opus-4-6": ADAPTIVE_MAX_ONLY,
    "sonnet-4-6": ADAPTIVE_MAX_ONLY,
    "opus-4-5": BUDGET,
    "sonnet-4-5": BUDGET,
    "haiku-4-5": BUDGET,
}

# Reasoning-capable Claude families intentionally NOT in the matrix yet.
# The staleness canary fails loudly if model_prices grows a reasoning-capable
# ``claude-*`` family absent from BOTH this allowlist and MODEL_TIER, forcing
# a human to consciously add the model + its tier + verify its expected column.
CANARY_ALLOWLIST: frozenset = frozenset(
    {
        "3-5-sonnet",
        "3-5-haiku",
        "3-7-sonnet",
        "sonnet-4",  # claude-sonnet-4 (4.0)
        "opus-4",  # claude-opus-4 (4.0)
        "opus-4-1",
        "4-sonnet",
        "4-opus",
    }
)

# --------------------------------------------------------------------------- #
# Budget ladder (mirrors litellm/constants.py defaults — see HEAD-vs-sweep note)
# --------------------------------------------------------------------------- #

_BUDGET_TOKENS: Dict[str, int] = {
    "minimal": 1024,  # floored at ANTHROPIC_MIN_THINKING_BUDGET_TOKENS (#27074)
    "low": 1024,
    "medium": 2048,
    "high": 4096,
    "xhigh": 8192,
    "max": 16384,
}

# Effort -> output_config.effort value for adaptive models.
_ADAPTIVE_EFFORT: Dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


# --------------------------------------------------------------------------- #
# Expected normalized subtree
# --------------------------------------------------------------------------- #

CLIENT_400 = "CLIENT_400"  # sentinel: must raise a clean 400 before the wire


@dataclass(frozen=True)
class Expected:
    """Normalized expected reasoning subtree for one (tier, effort) cell.

    ``thinking`` / ``output_config`` of ``None`` means the key MUST be absent
    from the built request (presence/absence is asserted, not just value).
    """

    thinking: Optional[dict]
    output_config: Optional[dict]


def _expected(tier: str, effort: str) -> Union[Expected, str]:
    # none / omit -> no thinking, no output_config (every tier)
    if effort in (OMIT, "none"):
        return Expected(thinking=None, output_config=None)

    # Garbage -> clean client-side 400 (every tier). #27074 turned these from
    # raw ValueError/500 into BadRequestError/AnthropicError@400.
    if effort in ("disabled", "invalid", ""):
        return CLIENT_400

    if tier == BUDGET:
        return Expected(
            thinking={"type": "enabled", "budget_tokens": _BUDGET_TOKENS[effort]},
            output_config=None,
        )

    # Adaptive tiers.
    if effort == "xhigh" and tier == ADAPTIVE_MAX_ONLY:
        # xhigh requires supports_xhigh_reasoning_effort with NO model-family
        # fallback -> rejected client-side on opus-4-6 / sonnet-4-6.
        return CLIENT_400

    return Expected(
        thinking={"type": "adaptive"},
        output_config={"effort": _ADAPTIVE_EFFORT[effort]},
    )


# Materialized table: (tier, effort) -> Expected | CLIENT_400
EXPECTED: Dict[Tuple[str, str], Union[Expected, str]] = {
    (tier, effort): _expected(tier, effort) for tier in TIERS for effort in EFFORTS
}


# --------------------------------------------------------------------------- #
# Normalized 400-class helper
# --------------------------------------------------------------------------- #

# Bare exceptions that, if raised, mean the request escaped as a generic 500
# (the exact #27074 / #27039 regression). Never acceptable.
_DIRTY_EXCEPTIONS = (ValueError, TypeError, AttributeError, KeyError)


def is_clean_400(exc: BaseException) -> bool:
    """True iff ``exc`` is a clean 400-class rejection (route-agnostic).

    Accepts ``litellm.BadRequestError`` or ``AnthropicError`` with
    ``status_code == 400``. Explicitly rejects bare Python exceptions, which
    are exactly what the #27074 fix eliminated.
    """
    if isinstance(exc, _DIRTY_EXCEPTIONS) and not isinstance(
        exc, litellm.exceptions.APIError
    ):
        return False
    if isinstance(exc, litellm.exceptions.BadRequestError):
        return True
    if isinstance(exc, AnthropicError):
        return getattr(exc, "status_code", None) in (400, "400")
    # Some routes wrap as the generic litellm BadRequest subclass hierarchy.
    return getattr(exc, "status_code", None) in (400, "400") and not isinstance(
        exc, _DIRTY_EXCEPTIONS
    )


# --------------------------------------------------------------------------- #
# Route / entrypoint enumeration
# --------------------------------------------------------------------------- #

CHAT = "chat_completions"
MESSAGES = "v1_messages"

MESSAGES_ENTRYPOINT = "messages"


@dataclass(frozen=True)
class Route:
    name: str
    # entrypoint -> ordered list of (wire_model_id, canonical_tier_key)
    models: Dict[str, List[Tuple[str, str]]]


# Mirrors the QA-sweep proxy config (PR #27039 comment).
_ANTHROPIC_MODELS = [
    ("claude-opus-4-7", "opus-4-7"),
    ("claude-opus-4-6", "opus-4-6"),
    ("claude-opus-4-5", "opus-4-5"),
    ("claude-sonnet-4-6", "sonnet-4-6"),
    ("claude-sonnet-4-5", "sonnet-4-5"),
    ("claude-haiku-4-5", "haiku-4-5"),
]

ROUTES: Tuple[Route, ...] = (
    Route(
        name="anthropic",
        models={
            CHAT: _ANTHROPIC_MODELS,
            MESSAGES: [
                ("claude-opus-4-7", "opus-4-7"),
                ("claude-opus-4-6", "opus-4-6"),
                ("claude-sonnet-4-6", "sonnet-4-6"),
                ("claude-opus-4-5", "opus-4-5"),
                ("claude-haiku-4-5", "haiku-4-5"),
            ],
        },
    ),
    Route(
        name="bedrock_converse",
        models={
            CHAT: [
                ("bedrock/converse/us.anthropic.claude-opus-4-7", "opus-4-7"),
                ("bedrock/converse/us.anthropic.claude-opus-4-6-v1", "opus-4-6"),
                (
                    "bedrock/converse/us.anthropic.claude-opus-4-5-20251101-v1:0",
                    "opus-4-5",
                ),
                ("bedrock/converse/us.anthropic.claude-sonnet-4-6", "sonnet-4-6"),
                (
                    "bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "sonnet-4-5",
                ),
            ],
        },
    ),
    Route(
        name="bedrock_invoke",
        models={
            CHAT: [
                ("bedrock/invoke/us.anthropic.claude-opus-4-7", "opus-4-7"),
                ("bedrock/invoke/us.anthropic.claude-opus-4-6-v1", "opus-4-6"),
                ("bedrock/invoke/us.anthropic.claude-sonnet-4-6", "sonnet-4-6"),
                (
                    "bedrock/invoke/us.anthropic.claude-opus-4-5-20251101-v1:0",
                    "opus-4-5",
                ),
            ],
            MESSAGES: [
                ("anthropic.claude-opus-4-7", "opus-4-7"),
                ("anthropic.claude-opus-4-6", "opus-4-6"),
                ("anthropic.claude-sonnet-4-6", "sonnet-4-6"),
                ("anthropic.claude-opus-4-5", "opus-4-5"),
            ],
        },
    ),
    Route(
        name="vertex_ai",
        models={
            CHAT: [
                ("claude-opus-4-7", "opus-4-7"),
                ("claude-opus-4-6", "opus-4-6"),
                ("claude-sonnet-4-6", "sonnet-4-6"),
                ("claude-haiku-4-5", "haiku-4-5"),
            ],
        },
    ),
    Route(
        name="azure_ai",
        models={
            CHAT: [
                ("claude-opus-4-7", "opus-4-7"),
                ("claude-opus-4-6", "opus-4-6"),
                ("claude-sonnet-4-6", "sonnet-4-6"),
                ("claude-haiku-4-5", "haiku-4-5"),
            ],
        },
    ),
)

ROUTES_BY_NAME: Dict[str, Route] = {r.name: r for r in ROUTES}


# --------------------------------------------------------------------------- #
# Per-route wire-request capture adapters (fully offline — no HTTP, no creds)
# --------------------------------------------------------------------------- #


def _completion_optional_params(config: Any, model: str, effort: str) -> dict:
    """Run ``map_openai_params`` exactly as the proxy would for this effort."""
    non_default: dict = {}
    if effort != OMIT:
        non_default["reasoning_effort"] = effort
    return config.map_openai_params(
        non_default_params=non_default,
        optional_params={},
        model=model,
        drop_params=False,
    )


def _normalize(thinking: Any, output_config: Any) -> Dict[str, Optional[dict]]:
    return {
        "thinking": thinking if isinstance(thinking, dict) else None,
        "output_config": output_config if isinstance(output_config, dict) else None,
    }


_MESSAGES = [{"role": "user", "content": "What is 2+2?"}]


def _capture_anthropic_chat(model: str, effort: str) -> Dict[str, Optional[dict]]:
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    config = AnthropicConfig()
    op = _completion_optional_params(config, model, effort)
    op.setdefault("max_tokens", 1024)
    result = config.transform_request(
        model=model,
        messages=list(_MESSAGES),
        optional_params=op,
        litellm_params={},
        headers={},
    )
    return _normalize(result.get("thinking"), result.get("output_config"))


def _capture_bedrock_converse_chat(
    model: str, effort: str
) -> Dict[str, Optional[dict]]:
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    config = AmazonConverseConfig()
    op = _completion_optional_params(config, model, effort)
    op["maxTokens"] = 1024
    result = config._transform_request(
        model=model,
        messages=list(_MESSAGES),
        optional_params=op,
        litellm_params={},
        headers={},
    )
    additional = result.get("additionalModelRequestFields") or {}
    # thinking value comes from the post-map params (what the PR test asserts);
    # output_config placement is the #27074-item-1 wire regression.
    return _normalize(op.get("thinking"), additional.get("output_config"))


def _capture_bedrock_invoke_chat(model: str, effort: str) -> Dict[str, Optional[dict]]:
    from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
        AmazonAnthropicClaudeConfig,
    )

    config = AmazonAnthropicClaudeConfig()
    op = _completion_optional_params(config, model, effort)
    op.setdefault("max_tokens", 1024)
    result = config.transform_request(
        model=model,
        messages=list(_MESSAGES),
        optional_params=op,
        litellm_params={},
        headers={},
    )
    return _normalize(result.get("thinking"), result.get("output_config"))


def _capture_vertex_chat(model: str, effort: str) -> Dict[str, Optional[dict]]:
    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
        VertexAIAnthropicConfig,
    )

    config = VertexAIAnthropicConfig()
    op = _completion_optional_params(config, model, effort)
    op.setdefault("max_tokens", 1024)
    result = config.transform_request(
        model=model,
        messages=list(_MESSAGES),
        optional_params=op,
        litellm_params={},
        headers={},
    )
    return _normalize(result.get("thinking"), result.get("output_config"))


def _capture_azure_chat(model: str, effort: str) -> Dict[str, Optional[dict]]:
    from litellm.llms.azure_ai.anthropic.transformation import AzureAnthropicConfig

    config = AzureAnthropicConfig()
    op = _completion_optional_params(config, model, effort)
    op.setdefault("max_tokens", 1024)
    result = config.transform_request(
        model=model,
        messages=list(_MESSAGES),
        optional_params=op,
        litellm_params={},
        headers={},
    )
    return _normalize(result.get("thinking"), result.get("output_config"))


def _messages_optional_params(effort: str) -> dict:
    params: dict = {"max_tokens": 1024}
    if effort != OMIT:
        params["reasoning_effort"] = effort
    return params


def _capture_anthropic_messages(model: str, effort: str) -> Dict[str, Optional[dict]]:
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
        AnthropicMessagesConfig,
    )

    config = AnthropicMessagesConfig()
    result = config.transform_anthropic_messages_request(
        model=model,
        messages=list(_MESSAGES),
        anthropic_messages_optional_request_params=_messages_optional_params(effort),
        litellm_params={},
        headers={},
    )
    return _normalize(result.get("thinking"), result.get("output_config"))


def _capture_bedrock_invoke_messages(
    model: str, effort: str
) -> Dict[str, Optional[dict]]:
    from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
        AmazonAnthropicClaudeMessagesConfig,
    )
    from litellm.types.router import GenericLiteLLMParams

    config = AmazonAnthropicClaudeMessagesConfig()
    result = config.transform_anthropic_messages_request(
        model=model,
        messages=list(_MESSAGES),
        anthropic_messages_optional_request_params=_messages_optional_params(effort),
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    return _normalize(result.get("thinking"), result.get("output_config"))


_CHAT_ADAPTERS: Dict[str, Callable[[str, str], Dict[str, Optional[dict]]]] = {
    "anthropic": _capture_anthropic_chat,
    "bedrock_converse": _capture_bedrock_converse_chat,
    "bedrock_invoke": _capture_bedrock_invoke_chat,
    "vertex_ai": _capture_vertex_chat,
    "azure_ai": _capture_azure_chat,
}

_MESSAGES_ADAPTERS: Dict[str, Callable[[str, str], Dict[str, Optional[dict]]]] = {
    "anthropic": _capture_anthropic_messages,
    "bedrock_invoke": _capture_bedrock_invoke_messages,
}


def build_wire_request(
    route: str, entrypoint: str, model: str, effort: str
) -> Dict[str, Optional[dict]]:
    """Capture the normalized reasoning subtree LiteLLM would put on the wire.

    Fully offline. Propagates the production exception for client-rejected
    efforts (the test classifies it via :func:`is_clean_400`).
    """
    adapters = _CHAT_ADAPTERS if entrypoint == CHAT else _MESSAGES_ADAPTERS
    if route not in adapters:
        raise KeyError(f"no {entrypoint} adapter for route {route!r}")
    return adapters[route](model, effort)


# --------------------------------------------------------------------------- #
# Parametrization helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cell:
    route: str
    entrypoint: str
    model: str
    canonical: str
    effort: str

    @property
    def tier(self) -> str:
        return MODEL_TIER[self.canonical]

    @property
    def expected(self) -> Union[Expected, str]:
        return EXPECTED[(self.tier, self.effort)]

    @property
    def id(self) -> str:
        return f"{self.route}__{self.entrypoint}__{self.canonical}__{effort_label(self.effort)}"


def all_cells() -> List[Cell]:
    cells: List[Cell] = []
    for route in ROUTES:
        for entrypoint, models in route.models.items():
            for wire_model, canonical in models:
                for effort in EFFORTS:
                    cells.append(
                        Cell(
                            route=route.name,
                            entrypoint=entrypoint,
                            model=wire_model,
                            canonical=canonical,
                            effort=effort,
                        )
                    )
    return cells


# --------------------------------------------------------------------------- #
# Staleness canary data
# --------------------------------------------------------------------------- #


def _model_prices_path() -> str:
    here = os.path.dirname(__file__)
    # repo_root/model_prices_and_context_window.json
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(root, "model_prices_and_context_window.json")


def unmapped_reasoning_claude_families() -> List[str]:
    """Reasoning-capable anthropic ``claude-*`` families absent from the matrix.

    A non-empty result means a new model shipped that the matrix silently
    under-covers — the canary test fails so a human adds it consciously.
    """
    with open(_model_prices_path()) as fh:
        prices = json.load(fh)

    known = set(MODEL_TIER) | set(CANARY_ALLOWLIST)
    unmapped: set = set()
    for name, meta in prices.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("litellm_provider") != "anthropic":
            continue
        if not meta.get("supports_reasoning"):
            continue
        if "claude" not in name:
            continue
        if any(token in name for token in known):
            continue
        unmapped.add(name)
    return sorted(unmapped)
