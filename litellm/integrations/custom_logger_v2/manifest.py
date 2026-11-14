from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias, TypeVar

from litellm.integrations.custom_logger_v2.enums import Activation, Field, Hook, OptOutBy
from litellm.integrations.custom_logger_v2.hooks import InvalidDeclarationError, Subscription
from litellm.integrations.custom_logger_v2.logger import CustomLoggerV2
from litellm.integrations.custom_logger_v2.selection import Selection

ResultT: Final = TypeVar("ResultT")

LegacyBinding: TypeAlias = Callable[[object, object], Awaitable[object] | None]


class LegacyCall(Protocol):
    def __call__(
        self, binding: Callable[[object, object], ResultT], fields: Mapping[Field, object], /
    ) -> ResultT | None: ...


@dataclass(frozen=True, slots=True)
class CallbackManifest:
    logger: CustomLoggerV2
    selection: Selection
    subscriptions: Mapping[Hook, Subscription]
    request_state: type | None
    legacy_call: LegacyCall | None = None


def _instance_subscription(subscription: Subscription, parallel: bool, activation: Activation | None) -> Subscription:
    return replace(
        subscription,
        parallel=subscription.parallel or (parallel and subscription.may_abort),
        activation=activation,
    )


def _caller_can_remove(selection: Selection) -> bool:
    return selection.honors_no_log or OptOutBy.caller in selection.opt_out_by


def build_manifest(logger: CustomLoggerV2) -> CallbackManifest:
    name: Final = type(logger).__name__
    declared: Final = type(logger).subscriptions
    enabled: Final = frozenset(declared) if logger.enabled_hooks is None else logger.enabled_hooks
    not_implemented: Final = enabled - declared.keys()
    if not_implemented:
        hooks: Final = ", ".join(sorted(hook.value for hook in not_implemented))
        raise InvalidDeclarationError(f"{name} enables {hooks}, which it does not implement")
    stray_overrides: Final = logger.activation_overrides.keys() - enabled
    if stray_overrides:
        overridden: Final = ", ".join(sorted(hook.value for hook in stray_overrides))
        raise InvalidDeclarationError(f"{name} overrides activation for {overridden}, which it does not enable")
    subscriptions: Final = MappingProxyType(
        {
            hook: _instance_subscription(declared[hook], logger.parallel, logger.activation_overrides.get(hook))
            for hook in Hook
            if hook in enabled
        }
    )
    selection: Final = logger.selection or Selection(name=name)
    blocking: Final = any(subscription.writes or subscription.may_abort for subscription in subscriptions.values())
    if blocking and _caller_can_remove(selection):
        raise InvalidDeclarationError(
            f"{selection.name} writes or may abort, so neither no-log nor a caller opt-out may remove it"
        )
    return CallbackManifest(
        logger=logger,
        selection=selection,
        subscriptions=subscriptions,
        request_state=type(logger).request_state,
    )
