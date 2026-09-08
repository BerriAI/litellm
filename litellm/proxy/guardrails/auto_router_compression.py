"""
Decouples prompt compression between an auto router's routing decision and the model
it routes to, via ``auto_router_routing_compression`` / ``auto_router_model_compression``
on the marker deployment: a guardrail name, or ``"none"``.

Neither key set inherits today's behaviour. Either key set makes the auto router
authoritative and suppresses every other compression guardrail for that request.
"""

import contextvars
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import get_or_create_metadata_bucket
from litellm.router_utils.auto_router_model_naming import AUTO_ROUTER_MODEL_PREFIX
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.router import Router

COMPRESSION_GUARDRAIL_PROVIDERS: Final = frozenset({"headroom", "compresr"})
_NO_COMPRESSION: Final = "none"

# A ContextVar, not metadata: metadata reaches spend logs the caller can read, and a
# suppression list they can read is one they can replay to disable any guardrail.
_suppressed_compression_guardrails: Final[contextvars.ContextVar[frozenset[str]]] = contextvars.ContextVar(
    "litellm_auto_router_suppressed_compression_guardrails", default=frozenset()
)


def suppressed_compression_guardrails() -> frozenset[str]:
    """Names of the compression guardrails this request's auto router suppresses."""
    return _suppressed_compression_guardrails.get()


# Only the proxy calls `arm_pre_call`, so on the SDK path nothing arms and nothing
# compresses; the router must not assume the model hop already ran.
_model_hop_armed: Final[contextvars.ContextVar[bool]] = contextvars.ContextVar(
    "litellm_auto_router_model_hop_armed", default=False
)


def model_hop_compression_armed() -> bool:
    """True when this request's model-side compression guardrail was actually armed."""
    return _model_hop_armed.get()


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

    Pre-call arming and the routing hook both resolve through here, so an alias with
    several tag-scoped markers cannot suppress under one and then route under another.
    """
    if llm_router is None:
        return None
    deployments: Final = llm_router.get_model_list(model_name=model_alias, team_id=team_id) or ()
    markers: Final = tuple(
        litellm_params
        for deployment in deployments
        if isinstance(litellm_params := deployment.get("litellm_params"), Mapping)  # pyright: ignore[reportUnnecessaryIsInstance]  # filters out non-Mapping
        and str(litellm_params.get("model", "")).startswith(AUTO_ROUTER_MODEL_PREFIX)
    )
    requested: Final = frozenset(request_tags)
    tag_matched: Final = tuple(
        params for params in markers if (tags := params.get("tags")) and requested.issuperset(frozenset(tags))
    )
    # Untagged only: a marker scoped to tags this request lacks describes other traffic.
    untagged: Final = tuple(params for params in markers if not params.get("tags"))
    # Lazy, so the first marker carrying a policy wins and the rest are never read.
    candidates: Final = (policy_from_litellm_params(params) for params in (*tag_matched, *untagged))
    return next((policy for policy in candidates if policy is not None), None)


def team_id_from_request(request_kwargs: Mapping[str, object]) -> str | None:
    """The caller's team id, from whichever metadata bucket this surface writes to."""
    for meta_key in ("metadata", "litellm_metadata"):
        meta = request_kwargs.get(meta_key)
        if isinstance(meta, Mapping):
            team_id = meta.get("user_api_key_team_id")
            if isinstance(team_id, str):
                return team_id
    return None


def _compression_guardrail_classes() -> tuple[type, ...]:
    """The registered guardrail classes whose provider compresses prompts."""
    from litellm.proxy.guardrails.guardrail_registry import guardrail_class_registry

    return tuple(cls for name, cls in guardrail_class_registry.items() if name in COMPRESSION_GUARDRAIL_PROVIDERS)


def is_compression_guardrail(guardrail: object) -> bool:
    """Whether `guardrail` is an instance of a compression guardrail provider.

    Both hops validate through here: the policy fields are operator-supplied names, and
    an unvalidated one would get handed the conversation and invoked.
    """
    classes: Final = _compression_guardrail_classes()
    return bool(classes) and isinstance(guardrail, classes)


def _active_compression_guardrails() -> tuple["CustomGuardrail", ...]:
    """Every currently-active guardrail whose type is a compression guardrail."""
    import litellm
    from litellm.integrations.custom_guardrail import CustomGuardrail

    if not _compression_guardrail_classes():
        return ()
    active: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(callback_type=CustomGuardrail)
    return tuple(cb for cb in active if is_compression_guardrail(cb) and cb.guardrail_name)


async def arm_pre_call(
    data: dict[str, object],  # mutable-ok: arms the live request dict in place
    llm_router: "Router | None",
) -> None:
    """Apply an auto router's compression policy, if any, before guardrails run.

    Suppresses every other compression guardrail and re-enables the model-side
    guardrail the policy names (if any) even when it isn't ``default_on``.
    """
    _suppressed_compression_guardrails.set(frozenset())
    _model_hop_armed.set(False)
    if llm_router is None:
        return

    model_alias: Final = data.get("model")
    if not isinstance(model_alias, str) or not model_alias:
        return

    from litellm.router_strategy.tag_based_routing import (
        _get_tags_from_request_kwargs,  # pyright: ignore[reportPrivateUsage]  # used in router.py and budget_limiter.py too
    )

    policy: Final = policy_for_model(
        llm_router=llm_router,
        model_alias=model_alias,
        team_id=team_id_from_request(data),
        request_tags=_get_tags_from_request_kwargs(data),
    )
    if policy is None:
        return

    _suppressed_compression_guardrails.set(
        frozenset(
            name
            for guardrail in _active_compression_guardrails()
            if (name := guardrail.guardrail_name) and name != policy.model
        )
    )

    # Arming adds the name to `metadata["guardrails"]`, which runs it even if not default_on.
    armed_model_hop: Final = policy.model is not None and any(
        guardrail.guardrail_name == policy.model for guardrail in _active_compression_guardrails()
    )
    if policy.model is not None and not armed_model_hop:
        verbose_proxy_logger.warning(
            "AutoRouter compression: '%s' is not an active compression guardrail; the model hop is uncompressed",
            policy.model,
        )

    if armed_model_hop:
        _model_hop_armed.set(True)
        _, metadata = get_or_create_metadata_bucket(data)
        requested: Final = metadata.get("guardrails")
        existing: Final = tuple(requested) if isinstance(requested, (list, tuple)) else ()
        if policy.model not in existing:
            # A list: litellm_pre_call_utils isinstance-checks this key and drops a tuple.
            metadata["guardrails"] = [*existing, policy.model]  # mutable-ok: this key's contract is a list


def _as_routing_messages(
    messages: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:  # mutable-ok: shape fixed by the pre-routing hook protocol
    """A fresh, independently mutable copy, the shape the pre-routing hook takes."""
    return [dict(message) for message in messages]  # mutable-ok: shape fixed by the pre-routing hook protocol


async def messages_for_routing(
    policy: AutoRouterCompressionPolicy | None,
    # list[dict], not Sequence[Mapping]: fixed by the async_pre_routing_hook protocol.
    messages: list[dict[str, object]] | None,  # mutable-ok: shape fixed by the pre-routing hook protocol
    request_kwargs: Mapping[str, object],
) -> list[dict[str, object]] | None:  # mutable-ok: shape fixed by the pre-routing hook protocol
    """Messages to use for a routing decision, per `policy.routing`. None means the
    caller should route on whatever it already has.

    Reads the live messages, never a pre-guardrail copy: this compresses through a real
    guardrail that POSTs the text out, so routing on a pre-masking snapshot would leak
    what the masking guardrail stripped. When the model hop already compressed and the
    hops differ, routing therefore reads the compressed text rather than the original.
    """
    if policy is None or policy.routing is None:
        return None

    if not messages:
        return None

    from litellm.proxy.common_utils.registry_read_through import (
        get_initialized_guardrail_with_read_through,
    )

    guardrail: Final = await get_initialized_guardrail_with_read_through(policy.routing)
    if guardrail is None:
        verbose_proxy_logger.warning(
            "AutoRouter compression: guardrail '%s' not found; routing on uncompressed messages", policy.routing
        )
        return _as_routing_messages(messages)

    if not is_compression_guardrail(guardrail):
        verbose_proxy_logger.warning(
            "AutoRouter compression: guardrail '%s' is not a compression guardrail; routing on uncompressed messages",
            policy.routing,
        )
        return _as_routing_messages(messages)

    inputs: Final[GenericGuardrailAPIInputs] = {
        "structured_messages": _as_routing_messages(messages)  # pyright: ignore[reportAssignmentType]  # plain dicts, not AllMessageValues; see headroom.py's own use of this shape
    }
    model: Final = request_kwargs.get("model")
    # Throwaway: apply_guardrail writes stats here, so routing never double-counts into
    # extract_compression_saved_tokens.
    stats_sink: Final = {"messages": messages, "model": model}  # mutable-ok: apply_guardrail writes its stats here
    result: Final = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=stats_sink,
        input_type="request",
    )
    compressed: Final = result.get("structured_messages")
    return compressed if isinstance(compressed, list) else _as_routing_messages(messages)
