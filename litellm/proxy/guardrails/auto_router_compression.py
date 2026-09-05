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

import contextvars
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.constants import AUTO_ROUTER_SUPPRESSED_COMPRESSION_GUARDRAILS_KEY
from litellm.litellm_core_utils.core_helpers import get_or_create_metadata_bucket
from litellm.router_utils.auto_router_model_naming import AUTO_ROUTER_MODEL_PREFIX
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.router import Router

COMPRESSION_GUARDRAIL_PROVIDERS: Final = frozenset({"headroom", "compresr"})
_NO_COMPRESSION: Final = "none"

# The pre-compression messages, so a routing decision that does not share the model
# call's compression still classifies on the original text. Deliberately a ContextVar
# rather than a metadata key: `refresh_proxy_server_request_body_snapshot` copies
# metadata into `proxy_server_request.body`, which deployments persist to spend logs,
# and this holds the prompt as it was before any masking guardrail rewrote it.
_routing_messages_snapshot: Final[contextvars.ContextVar[tuple[Mapping[str, object], ...] | None]] = (
    contextvars.ContextVar("litellm_auto_router_routing_messages_snapshot", default=None)
)


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
    llm_router: "Router | None",
    model_alias: str,
    team_id: str | None,
    request_tags: Sequence[str],
) -> AutoRouterCompressionPolicy | None:
    """The compression policy of the auto router marker `model_alias` resolves to.

    Both the proxy's pre-call arming and the router's routing hook resolve the policy
    through here, with the same tag rule, so an alias carrying several tag-scoped
    markers can never suppress one marker's guardrail and then route under another
    marker's policy.
    """
    if llm_router is None:
        return None
    deployments: Final = llm_router.get_model_list(model_name=model_alias, team_id=team_id) or []
    markers: Final = tuple(
        litellm_params
        for deployment in deployments
        if isinstance(litellm_params := deployment.get("litellm_params") or {}, Mapping)
        and str(litellm_params.get("model", "")).startswith(AUTO_ROUTER_MODEL_PREFIX)
    )
    requested: Final = frozenset(request_tags)
    tag_matched: Final = tuple(
        params for params in markers if (tags := params.get("tags")) and requested.issuperset(frozenset(tags))
    )
    for params in (*tag_matched, *markers):
        policy = policy_from_litellm_params(params)
        if policy is not None:
            return policy
    return None


def team_id_from_request(request_kwargs: Mapping[str, object]) -> str | None:
    """The caller's team id, from whichever metadata bucket this surface writes to."""
    for meta_key in ("metadata", "litellm_metadata"):
        meta = request_kwargs.get(meta_key)
        if isinstance(meta, Mapping):
            team_id = meta.get("user_api_key_team_id")
            if isinstance(team_id, str):
                return team_id
    return None


def _active_compression_guardrails() -> tuple["CustomGuardrail", ...]:
    """Every currently-active guardrail whose type is a compression guardrail."""
    import litellm
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy.guardrails.guardrail_registry import guardrail_class_registry

    compression_classes: Final = tuple(
        cls for name, cls in guardrail_class_registry.items() if name in COMPRESSION_GUARDRAIL_PROVIDERS
    )
    if not compression_classes:
        return ()
    active: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(callback_type=CustomGuardrail)
    return tuple(cb for cb in active if isinstance(cb, compression_classes) and cb.guardrail_name)


async def arm_pre_call(data: MutableMapping[str, object], llm_router: "Router | None") -> MutableMapping[str, object]:
    """Apply an auto router's compression policy, if any, before guardrails run.

    Suppresses every other compression guardrail, re-enables the model-side
    guardrail the policy names (if any) even when it isn't ``default_on``, and
    snapshots the pre-compression messages so the routing decision can read them
    independently of whatever the model-side guardrail does to `data`.
    """
    _routing_messages_snapshot.set(None)
    if llm_router is None:
        return data

    model_alias: Final = data.get("model")
    if not isinstance(model_alias, str) or not model_alias:
        return data

    # Read-only until a policy is confirmed: creating the metadata bucket for every
    # request, including the vast majority with no auto-router compression policy,
    # would be an unwanted side effect of merely checking for one.
    from litellm.router_strategy.tag_based_routing import _get_tags_from_request_kwargs

    policy: Final = policy_for_model(
        llm_router=llm_router,
        model_alias=model_alias,
        team_id=team_id_from_request(data),
        request_tags=_get_tags_from_request_kwargs(data),
    )
    if policy is None:
        return data

    _, metadata = get_or_create_metadata_bucket(data)
    # Markers carry a per-process token so a caller cannot suppress a guardrail by
    # naming it in its own request metadata.
    suppressed: Final = tuple(
        marker
        for guardrail in _active_compression_guardrails()
        if guardrail.guardrail_name != policy.model and (marker := guardrail.auto_router_suppression_marker())
    )
    if suppressed:
        metadata[AUTO_ROUTER_SUPPRESSED_COMPRESSION_GUARDRAILS_KEY] = list(suppressed)

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
        _routing_messages_snapshot.set(tuple(dict(message) for message in snapshot))

    return data


def _snapshot_messages() -> list[dict[str, object]] | None:
    snapshot: Final = _routing_messages_snapshot.get()
    return None if snapshot is None else [dict(message) for message in snapshot]


async def messages_for_routing(
    policy: AutoRouterCompressionPolicy | None,
    messages: list[dict[str, object]] | None,
    request_kwargs: Mapping[str, object],
) -> list[dict[str, object]] | None:
    """Messages to use for a routing decision, per `policy.routing`.

    Returns None when the caller should route on whatever messages it already has.
    The model call is untouched either way: model-side compression, if any, already
    ran as an ordinary pre-call guardrail before the router was reached, so when the
    two hops differ the routing decision reads the pre-compression snapshot rather
    than what that guardrail left behind.
    """
    if policy is None:
        return None

    original: Final = _snapshot_messages() or messages

    if policy.routing is None:
        # Explicitly no compression for routing. When the model side compressed, the
        # messages in hand are its output, so fall back to the untouched snapshot.
        return _snapshot_messages() if policy.model is not None else None

    if not original:
        return None

    from litellm.proxy.common_utils.registry_read_through import (
        get_initialized_guardrail_with_read_through,
    )

    guardrail: Final = await get_initialized_guardrail_with_read_through(policy.routing)
    if guardrail is None:
        verbose_proxy_logger.warning(
            "AutoRouter compression: guardrail '%s' not found; routing on uncompressed messages", policy.routing
        )
        return original

    inputs: GenericGuardrailAPIInputs = {
        "structured_messages": [dict(m) for m in original]  # pyright: ignore[reportAssignmentType]  # plain dicts, not AllMessageValues; see headroom.py's own use of this shape
    }
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
