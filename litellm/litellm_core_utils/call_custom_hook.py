from typing import Final

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.hook_filter_utils import should_run_hook_for_filters


def _should_run(
    callback: CustomLogger,
    hook_name: str,
    *,
    model: str | None,
    key_alias: str | None,
    model_tags: tuple[str, ...],
    request_tags: tuple[str, ...],
) -> bool:
    if not litellm.enable_hook_filters:
        return True
    hook_filter: Final = None if callback.hook_filters is None else callback.hook_filters.get(hook_name)
    return should_run_hook_for_filters(
        hook_filter, model=model, key_alias=key_alias, model_tags=model_tags, request_tags=request_tags
    )


async def call_custom_hook(
    callback: CustomLogger,
    hook_name: str,
    *,
    target: CustomLogger | None = None,
    model: str | None = None,
    key_alias: str | None = None,
    model_tags: tuple[str, ...] = (),
    request_tags: tuple[str, ...] = (),
    **hook_kwargs: object,  # kwargs-ok: forwards arbitrary, hook-specific kwargs to the named hook method
) -> object:
    """
    The one place any code path invokes a named async hook method on a
    CustomLogger/CustomGuardrail instance. Consults ``callback``'s own
    hook_filters (set by its config loader) before invoking, so every
    dispatch site gets filtering for free by routing through here instead of
    calling the hook method directly.

    ``target`` covers the one real case where the object actually invoked
    differs from the one holding the config: ProxyLogging's unified-guardrail
    wrapper. It defaults to ``callback`` itself.
    """
    if not _should_run(
        callback, hook_name, model=model, key_alias=key_alias, model_tags=model_tags, request_tags=request_tags
    ):
        return None
    hook_fn: Final = getattr(  # pyright: ignore[reportAny]  # dynamic dispatch by name
        target if target is not None else callback, hook_name
    )
    return await hook_fn(**hook_kwargs)  # pyright: ignore[reportAny]  # named hook's own return type


def call_custom_hook_sync(
    callback: CustomLogger,
    hook_name: str,
    *,
    target: CustomLogger | None = None,
    model: str | None = None,
    key_alias: str | None = None,
    model_tags: tuple[str, ...] = (),
    request_tags: tuple[str, ...] = (),
    **hook_kwargs: object,  # kwargs-ok: forwards arbitrary, hook-specific kwargs to the named hook method
) -> object:
    """Sync counterpart to call_custom_hook, for the sync hook methods (log_success_event, logging_hook)."""
    if not _should_run(
        callback, hook_name, model=model, key_alias=key_alias, model_tags=model_tags, request_tags=request_tags
    ):
        return None
    hook_fn: Final = getattr(  # pyright: ignore[reportAny]  # dynamic dispatch by name
        target if target is not None else callback, hook_name
    )
    return hook_fn(**hook_kwargs)  # pyright: ignore[reportAny]  # named hook's own return type
