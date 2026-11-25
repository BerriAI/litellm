"""
Host side of the Rust core's routing-plugin pipeline (``Router(plugins=[...])``).

The Rust core decides where the pipeline applies (only the async pick runs it; the sync pick
refuses a router with plugins) and which deployments the plugins' candidate models keep
(``router-core/src/routing_plugins.rs``). The plugins themselves are live Python objects, so this
module runs them on a ``routing_plugins`` select step, the way the Python router core's
``_run_routing_plugins`` does, and reports the candidate models left on the request.
"""

from collections.abc import Mapping, Sequence
from typing import Final, Protocol

from litellm.litellm_core_utils.core_helpers import as_str_mapping, get_metadata_variable_name_from_kwargs
from litellm.litellm_core_utils.routing_common_utils import resolve_model_group_alias
from litellm.types.router import RoutingContext, RoutingPlugin

SYNC_PATH_MESSAGE: Final = (
    "Router(plugins=[...]) is configured, but this call resolved to the synchronous "
    "deployment-selection path, which never runs the routing-plugin pipeline. This "
    "happens for sync Router methods (e.g. Router.completion()) and for async calls "
    "with a routing_strategy that has no async-native selector (e.g. legacy "
    "'usage-based-routing', v1). Silently skipping "
    "configured plugins would let a policy plugin (e.g. a deny-all rule) be bypassed. "
    "Use an async Router method with a supported routing_strategy (simple-shuffle, "
    "usage-based-routing-v2, cost-based-routing, latency-based-routing, least-busy), "
    "or remove `plugins` from the Router config."
)
_CANDIDATES_KEY: Final = "_routing_plugin_candidate_models"


class PluginRegistry(Protocol):
    def _core_get_model_list(
        self, model_name: str | None = None, team_id: str | None = None
    ) -> Sequence[Mapping[str, object]] | None: ...


def _resolved_litellm_models(
    registry: PluginRegistry, model: str
) -> list[str]:  # mutable-ok: the plugins' candidate list
    """``_core_resolved_litellm_models``: the provider models a call to ``model`` is served by."""
    return [
        litellm_model
        for deployment in registry._core_get_model_list(model_name=model) or ()  # pyright: ignore[reportPrivateUsage]  # the host registry's listing
        if isinstance(litellm_model := (as_str_mapping(deployment.get("litellm_params")) or {}).get("model"), str)
        and litellm_model
    ]


def candidate_models(request: Mapping[str, object]) -> Mapping[str, object]:
    """The answer to a ``routing_plugins`` step: the candidate models the plugins left on the
    request, ``None`` when none narrowed it."""
    metadata: Final = as_str_mapping(request.get(get_metadata_variable_name_from_kwargs(dict(request))))
    return {"models": metadata.get(_CANDIDATES_KEY) if metadata is not None else None}


async def run_plugins(
    plugins: Sequence[RoutingPlugin],
    registry: PluginRegistry,
    model_group_alias: Mapping[str, object],
    model: str,
    request: dict[str, object],  # mutable-ok: the plugins stamp their outcome on the caller's request
    messages: object,
) -> None:
    """``_run_routing_plugins``: run ``plugins`` in order over ``model``'s candidate models, then
    stamp their signals (and a narrowed candidate list) on the request's metadata."""
    from litellm.litellm_core_utils.prompt_templates.factory import resolve_structured_messages

    registered: Final = resolve_model_group_alias(model_group_alias, model) or model
    models: Final = _resolved_litellm_models(registry, registered)
    metadata_key: Final = get_metadata_variable_name_from_kwargs(request)
    metadata: Final = request.setdefault(metadata_key, {})
    raw_messages: Final[object] = messages or list[object]()
    context = RoutingContext(  # rebind-ok: each plugin returns the context the next one reads
        raw_messages=raw_messages,  # pyright: ignore[reportArgumentType]  # the request's messages
        structured_messages=resolve_structured_messages(messages=messages, request_kwargs=request) or [],  # pyright: ignore[reportArgumentType]  # the request's messages
        candidate_models=list(models),
        metadata=metadata,  # pyright: ignore[reportArgumentType]  # the request's metadata dict
    )
    for plugin in plugins:
        context = await plugin.run(context)
    metadata["routing_plugin_signals"] = context.signals  # pyright: ignore[reportIndexIssue]  # the request's metadata dict
    if len(context.candidate_models) < len(models):
        metadata[_CANDIDATES_KEY] = context.candidate_models  # pyright: ignore[reportIndexIssue]  # see above


__all__ = ["SYNC_PATH_MESSAGE", "PluginRegistry", "candidate_models", "run_plugins"]
