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

# Compression guardrails this request's auto router has switched off. Deliberately a
# ContextVar rather than a metadata key: `refresh_proxy_server_request_body_snapshot`
# copies metadata into `proxy_server_request.body`, which deployments persist to spend
# logs. A suppression list that reaches a log the caller can read is a list the caller
# can replay, which would let any request switch off a PII or content-filter guardrail.
# Nothing here is caller-supplied, so there is no marker to forge in the first place.
_suppressed_compression_guardrails: Final[contextvars.ContextVar[frozenset[str]]] = contextvars.ContextVar(
    "litellm_auto_router_suppressed_compression_guardrails", default=frozenset()
)


def suppressed_compression_guardrails() -> frozenset[str]:
    """Names of the compression guardrails this request's auto router suppresses."""
    return _suppressed_compression_guardrails.get()


# Whether `arm_pre_call` actually armed a model-side compression guardrail for this
# request. Only the proxy calls `arm_pre_call`, so on the SDK path nothing arms and
# nothing compresses; the router must not assume the model hop already ran.
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

    Both the proxy's pre-call arming and the router's routing hook resolve the policy
    through here, with the same tag rule, so an alias carrying several tag-scoped
    markers can never suppress one marker's guardrail and then route under another
    marker's policy.
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
    # Only untagged markers may serve as the fallback. A marker scoped to tags this
    # request does not carry describes a different slice of traffic, so falling back
    # to it would apply, say, an "eu" policy to a "us" request purely on config order.
    untagged: Final = tuple(params for params in markers if not params.get("tags"))
    # Lazily, so the first marker carrying a policy still wins and the rest are never
    # read. A generator rather than a loop-local: the name is bound once per item and
    # never rebound, which `: Final` cannot express inside a loop body.
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

    Both hops are validated through here. The two policy fields are operator-supplied
    names and nothing else constrains them, so without this a name that resolves to an
    ordinary guardrail would be handed the conversation and invoked: the routing hop
    calls `apply_guardrail` directly, which POSTs the content wherever that guardrail
    sends it, and the model hop is added to `metadata["guardrails"]`, which runs it even
    when it is not `default_on`.
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

    # Read-only until a policy is confirmed: creating the metadata bucket for every
    # request, including the vast majority with no auto-router compression policy,
    # would be an unwanted side effect of merely checking for one.
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

    # Only a name that resolves to a real compression guardrail may be armed: this adds
    # it to `metadata["guardrails"]`, which runs it even when it is not `default_on`.
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
            # A list, not a tuple: litellm_pre_call_utils tests this key with
            # isinstance(..., list) and extends it, and would drop a tuple on the floor.
            metadata["guardrails"] = [*existing, policy.model]  # mutable-ok: this key's contract is a list


def _as_routing_messages(
    messages: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:  # mutable-ok: shape fixed by the pre-routing hook protocol
    """A fresh, independently mutable copy, the shape the pre-routing hook takes."""
    return [dict(message) for message in messages]  # mutable-ok: shape fixed by the pre-routing hook protocol


async def messages_for_routing(
    policy: AutoRouterCompressionPolicy | None,
    # list[dict], not Sequence[Mapping]: the async_pre_routing_hook protocol in
    # litellm/types/router.py types `messages` as list[dict[str, Any]].
    messages: list[dict[str, object]] | None,  # mutable-ok: shape fixed by the pre-routing hook protocol
    request_kwargs: Mapping[str, object],
) -> list[dict[str, object]] | None:  # mutable-ok: shape fixed by the pre-routing hook protocol
    """Messages to use for a routing decision, per `policy.routing`.

    Returns None when the caller should route on whatever messages it already has.

    Always reads the live messages, never a pre-guardrail copy of them. The routing
    hop compresses through a real guardrail, which POSTs the text to an external
    compression service, so it must see what every other guardrail has already done
    to the request. Routing on a snapshot taken before the pre-call hook would send
    a masking guardrail's own input straight back out of the proxy.

    The consequence, when the model hop compressed and the two hops differ: the
    messages in hand are that guardrail's output, and there is no un-compressed copy
    left to route on. The routing decision reads the compressed text in that one
    combination rather than leaking the original.
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

    # apply_guardrail below hands this guardrail the conversation and it POSTs the
    # content to whatever service backs it, so the name has to be a compression
    # guardrail rather than any guardrail the operator happened to name.
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
    # A throwaway request_data: apply_guardrail writes its stats onto this dict, not the
    # real request's metadata, so routing-side compression never double-counts against
    # extract_compression_saved_tokens's model-savings accounting.
    stats_sink: Final = {"messages": messages, "model": model}  # mutable-ok: apply_guardrail writes its stats here
    result: Final = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=stats_sink,
        input_type="request",
    )
    compressed: Final = result.get("structured_messages")
    return compressed if isinstance(compressed, list) else _as_routing_messages(messages)
