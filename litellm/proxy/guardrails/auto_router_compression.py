"""
Decouples prompt compression between an auto router's routing decision and the
model it routes to. An auto router marker deployment may set
``auto_router_routing_compression`` and/or ``auto_router_model_compression`` in its
``litellm_params`` to name the compression guardrail that hop should use, or
``"none"`` to run no compression on that hop. Neither key set means the request's
own compression guardrails (key/team/model-level, or an "Always on" guardrail)
apply to both hops unchanged, exactly as before this feature existed.

Once either key is set, this auto router is authoritative: every other compression
guardrail is suppressed for that request, and only these two settings decide what
each hop sees.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from litellm._logging import verbose_proxy_logger
from litellm.constants import AUTO_ROUTER_SUPPRESSED_COMPRESSION_GUARDRAILS_KEY
from litellm.litellm_core_utils.core_helpers import (
    get_metadata_variable_name_from_kwargs,
    get_or_create_metadata_bucket,
)
from litellm.router_utils.auto_router_model_naming import AUTO_ROUTER_MODEL_PREFIX
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.router import Router
else:
    CustomGuardrail = Any
    Router = Any

COMPRESSION_GUARDRAIL_PROVIDERS: Final = frozenset({"headroom", "compresr"})
_NO_COMPRESSION: Final = "none"

# Metadata key stashing the pre-compression messages so a routing decision that
# names a different compression than the model call still compresses the
# original text, not whatever the model-side guardrail already rewrote it to.
AUTO_ROUTER_ROUTING_MESSAGES_SNAPSHOT_KEY: Final = "_auto_router_routing_messages_snapshot"


@dataclass(frozen=True, slots=True)
class AutoRouterCompressionPolicy:
    """An auto router's compression choice for each hop. ``None`` means no compression."""

    routing: str | None
    model: str | None

    @property
    def is_same(self) -> bool:
        return self.routing == self.model


def _normalized_compression_choice(raw: object) -> str | None:
    if not isinstance(raw, str) or not raw:
        return None
    return None if raw.strip().lower() == _NO_COMPRESSION else raw


def policy_from_litellm_params(litellm_params: Mapping[str, object]) -> AutoRouterCompressionPolicy | None:
    raw_routing: Final = litellm_params.get("auto_router_routing_compression")
    raw_model: Final = litellm_params.get("auto_router_model_compression")
    if raw_routing is None and raw_model is None:
        return None
    return AutoRouterCompressionPolicy(
        routing=_normalized_compression_choice(raw_routing),
        model=_normalized_compression_choice(raw_model),
    )


def policy_for_model(
    llm_router: "Router | None", model_alias: str, team_id: str | None
) -> AutoRouterCompressionPolicy | None:
    """The compression policy declared by the auto router marker deployment `model_alias` resolves to.

    Mirrors the alias lookup in ``_check_and_merge_model_level_guardrails``: this runs
    before routing has picked a strategy, so it takes the first marker deployment for
    the alias rather than disambiguating by request tags.
    """
    if llm_router is None:
        return None
    deployments: Final = llm_router.get_model_list(model_name=model_alias, team_id=team_id) or []
    for deployment in deployments:
        litellm_params: Final = deployment.get("litellm_params") or {}
        model_field = litellm_params.get("model")
        if not isinstance(model_field, str) or not model_field.startswith(AUTO_ROUTER_MODEL_PREFIX):
            continue
        policy = policy_from_litellm_params(litellm_params)
        if policy is not None:
            return policy
    return None


def _active_compression_guardrail_names() -> frozenset[str]:
    """Names of every currently-active guardrail whose type is a compression guardrail."""
    import litellm
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy.guardrails.guardrail_registry import guardrail_class_registry

    compression_classes: Final = tuple(
        cls for name, cls in guardrail_class_registry.items() if name in COMPRESSION_GUARDRAIL_PROVIDERS
    )
    if not compression_classes:
        return frozenset()
    active: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(callback_type=CustomGuardrail)
    return frozenset(
        cb.guardrail_name for cb in active if isinstance(cb, compression_classes) and cb.guardrail_name
    )


async def arm_pre_call(data: dict, llm_router: "Router | None") -> dict:
    """Apply an auto router's compression policy, if any, before guardrails run.

    Suppresses every other compression guardrail, re-enables the model-side
    guardrail the policy names (if any) even when it isn't ``default_on``, and
    snapshots the pre-compression messages so the routing decision can compress
    them independently of whatever the model-side guardrail does to `data`.
    """
    if llm_router is None:
        return data

    model_alias: Final = data.get("model")
    if not isinstance(model_alias, str) or not model_alias:
        return data

    # Read-only until a policy is confirmed: creating the metadata bucket for every
    # request, including the vast majority with no auto-router compression policy,
    # would be an unwanted side effect of merely checking for one.
    metadata_key: Final = get_metadata_variable_name_from_kwargs(data)
    existing_bucket: Final = data.get(metadata_key)
    other_bucket: Final = data.get("metadata" if metadata_key == "litellm_metadata" else "litellm_metadata")
    team_id: Final = (existing_bucket.get("user_api_key_team_id") if isinstance(existing_bucket, dict) else None) or (
        other_bucket.get("user_api_key_team_id") if isinstance(other_bucket, dict) else None
    )

    policy: Final = policy_for_model(llm_router=llm_router, model_alias=model_alias, team_id=team_id)
    if policy is None:
        return data

    _, metadata = get_or_create_metadata_bucket(data)
    suppressed: Final = _active_compression_guardrail_names() - ({policy.model} if policy.model else set())
    if suppressed:
        metadata[AUTO_ROUTER_SUPPRESSED_COMPRESSION_GUARDRAILS_KEY] = sorted(suppressed)

    if policy.model is not None:
        requested = metadata.get("guardrails")
        if isinstance(requested, list):
            if policy.model not in requested:
                requested.append(policy.model)
        else:
            metadata["guardrails"] = [policy.model]

    from litellm.litellm_core_utils.prompt_templates.factory import resolve_structured_messages

    snapshot: Final = resolve_structured_messages(messages=data.get("messages"), request_kwargs=data)
    if snapshot is not None:
        metadata[AUTO_ROUTER_ROUTING_MESSAGES_SNAPSHOT_KEY] = copy.deepcopy(snapshot)

    return data


async def messages_for_routing(
    policy: AutoRouterCompressionPolicy | None,
    messages: list[dict[str, Any]] | None,
    request_kwargs: Mapping[str, object],
) -> list[dict[str, Any]] | None:
    """Messages to use for a routing decision, compressed per `policy.routing`.

    Returns None when there is no policy or the policy's routing side names no
    compression, meaning the caller should route on whatever messages it already
    has. The model call is untouched by this function either way: model-side
    compression, if any, already ran as an ordinary pre-call guardrail before the
    router was ever reached.
    """
    if policy is None or policy.routing is None:
        return None

    from litellm.proxy.common_utils.registry_read_through import (
        get_initialized_guardrail_with_read_through,
    )

    metadata_key: Final = get_metadata_variable_name_from_kwargs(request_kwargs)
    metadata: Final = request_kwargs.get(metadata_key)
    snapshot: Final = metadata.get(AUTO_ROUTER_ROUTING_MESSAGES_SNAPSHOT_KEY) if isinstance(metadata, dict) else None
    original: Final = snapshot if isinstance(snapshot, list) else messages
    if not original:
        return None

    guardrail: Final = await get_initialized_guardrail_with_read_through(policy.routing)
    if guardrail is None:
        verbose_proxy_logger.warning(
            "AutoRouter compression: guardrail '%s' not found; routing on uncompressed messages", policy.routing
        )
        return None

    inputs: GenericGuardrailAPIInputs = {"structured_messages": [dict(m) for m in original]}
    # A throwaway request_data: apply_guardrail writes its stats onto this dict, not
    # the real request's metadata, so routing-side compression never double-counts
    # against extract_compression_saved_tokens's model-savings accounting.
    throwaway_request_data: Final[dict[str, object]] = {
        "messages": original,
        "model": request_kwargs.get("model"),
    }
    result: Final = await guardrail.apply_guardrail(
        inputs=inputs, request_data=throwaway_request_data, input_type="request"
    )
    compressed = result.get("structured_messages")
    return compressed if isinstance(compressed, list) else original
