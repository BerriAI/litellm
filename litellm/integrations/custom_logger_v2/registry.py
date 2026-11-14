from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from types import MappingProxyType
from typing import Final

from litellm.integrations.custom_logger_v2.enums import Hook
from litellm.integrations.custom_logger_v2.logger import CustomLoggerV2
from litellm.integrations.custom_logger_v2.manifest import CallbackManifest
from litellm.integrations.custom_logger_v2.planner import Entry, HookPlan, plan_hook

DEFAULT_PLAN_CACHE_SIZE: Final = 256


def _hook_entries(manifests: tuple[CallbackManifest, ...], hook: Hook) -> tuple[Entry, ...]:
    return tuple(
        Entry(position, manifest, manifest.subscriptions[hook])
        for position, manifest in enumerate(manifests)
        if hook in manifest.subscriptions
    )


class RegistrySnapshot:
    __slots__ = ("entries", "manifests", "plan")

    def __init__(self, manifests: tuple[CallbackManifest, ...], plan_cache_size: int) -> None:
        self.manifests: Final = manifests
        self.entries: Final[Mapping[Hook, tuple[Entry, ...]]] = MappingProxyType(
            {hook: _hook_entries(manifests, hook) for hook in Hook}
        )
        self.plan: Final[Callable[[Hook, tuple[int, ...]], HookPlan]] = lru_cache(maxsize=plan_cache_size)(
            self._build_plan
        )

    def _build_plan(self, hook: Hook, selected: tuple[int, ...]) -> HookPlan:
        hook_entries: Final = self.entries[hook]
        return plan_hook(tuple(hook_entries[index] for index in selected))


class CallbackRegistry:
    def __init__(
        self, manifests: Sequence[CallbackManifest] = (), plan_cache_size: int = DEFAULT_PLAN_CACHE_SIZE
    ) -> None:
        self._plan_cache_size: Final = plan_cache_size
        self.snapshot: RegistrySnapshot = RegistrySnapshot(tuple(manifests), plan_cache_size)

    def register(self, manifest: CallbackManifest) -> None:
        self.snapshot = RegistrySnapshot((*self.snapshot.manifests, manifest), self._plan_cache_size)

    def unregister(self, logger: CustomLoggerV2) -> None:
        kept: Final = tuple(manifest for manifest in self.snapshot.manifests if manifest.logger is not logger)
        self.snapshot = RegistrySnapshot(kept, self._plan_cache_size)
