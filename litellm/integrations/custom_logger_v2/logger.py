import inspect
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import replace
from types import MappingProxyType
from typing import ClassVar, Final, TypeVar

from litellm.integrations.custom_logger_v2.enums import Activation, Field, Hook
from litellm.integrations.custom_logger_v2.hooks import InvalidDeclarationError, Subscription
from litellm.integrations.custom_logger_v2.selection import Selection

HandlerT: Final = TypeVar("HandlerT", bound=Callable[..., object])

_SUBSCRIPTION_ATTR: Final = "__custom_logger_v2_subscription__"


def on(
    hook: Hook,
    *,
    reads: Set[Field] = frozenset(),
    writes: Set[Field] = frozenset(),
    may_abort: bool = False,
    parallel: bool | None = None,
    awaited: bool = False,
) -> Callable[[HandlerT], HandlerT]:
    def decorate(handler: HandlerT) -> HandlerT:
        subscription: Final = Subscription(
            hook=hook,
            handler=handler.__name__,
            reads=frozenset(reads),
            writes=frozenset(writes),
            may_abort=may_abort,
            parallel=(hook is Hook.during_call and may_abort) if parallel is None else parallel,
            awaited=awaited,
        )
        setattr(handler, _SUBSCRIPTION_ATTR, subscription)
        return handler

    return decorate


def _declared_subscription(value: object) -> Subscription | None:
    marker: Final[object] = getattr(value, _SUBSCRIPTION_ATTR, None)
    return marker if isinstance(marker, Subscription) else None


def _namespace(cls: type) -> Mapping[str, object]:
    return vars(cls)


def _declared_in(cls: type) -> Mapping[Hook, Subscription]:
    declared: Final = tuple(
        replace(subscription, handler=name)
        for name, value in _namespace(cls).items()
        if (subscription := _declared_subscription(value)) is not None
    )
    hooks: Final = tuple(subscription.hook for subscription in declared)
    duplicated: Final = sorted({hook.value for hook in hooks if hooks.count(hook) > 1})
    if duplicated:
        raise InvalidDeclarationError(f"{cls.__name__} declares more than one handler for {', '.join(duplicated)}")
    return MappingProxyType({subscription.hook: subscription for subscription in declared})


def _nearest(layers: Sequence[Mapping[Hook, Subscription]], hook: Hook) -> Subscription | None:
    return next((layer[hook] for layer in layers if hook in layer), None)


def _handler_problem(cls: type, subscription: Subscription) -> str | None:
    handler: Final[object] = getattr(cls, subscription.handler, None)
    if not callable(handler):
        return f"{subscription.handler} handles {subscription.hook.value} but is not callable"
    redeclared: Final = _declared_subscription(handler)
    if redeclared is not None and redeclared.hook is not subscription.hook:
        return (
            f"{subscription.handler} is decorated for {redeclared.hook.value} but also inherits "
            f"the {subscription.hook.value} subscription"
        )
    blocking: Final = bool(subscription.writes) or subscription.may_abort or subscription.awaited
    if blocking and not inspect.iscoroutinefunction(handler):
        return f"{subscription.handler} must be async because it writes, may abort or is awaited"
    return None


class CustomLoggerV2:
    _own_subscriptions: ClassVar[Mapping[Hook, Subscription]] = MappingProxyType({})
    subscriptions: ClassVar[Mapping[Hook, Subscription]] = MappingProxyType({})
    request_state: ClassVar[type | None] = None
    declared_hooks: ClassVar[frozenset[Hook]] = frozenset()

    selection: Selection | None = None
    enabled_hooks: frozenset[Hook] | None = None
    parallel: bool = False
    activation_overrides: Mapping[Hook, Activation] = MappingProxyType({})

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._own_subscriptions = _declared_in(cls)
        layers: Final = tuple(base._own_subscriptions for base in cls.__mro__ if issubclass(base, CustomLoggerV2))
        resolved: Final = MappingProxyType(
            {hook: subscription for hook in Hook if (subscription := _nearest(layers, hook)) is not None}
        )
        problems: Final = tuple(
            problem
            for subscription in resolved.values()
            if (problem := _handler_problem(cls, subscription)) is not None
        )
        if problems:
            raise InvalidDeclarationError(f"{cls.__name__}: {'; '.join(problems)}")
        cls.subscriptions = resolved
        CustomLoggerV2.declared_hooks = CustomLoggerV2.declared_hooks | resolved.keys()
