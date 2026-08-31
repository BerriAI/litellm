"""Capability gate for image-bearing requests through the complexity router.

The classifier reads text alone, so an image request whose text classifies cheap lands on a
text-only model and fails with a provider 400 no fallback catches. When `modality_routing` is
enabled, every NEW placement the router makes runs through one `ModalityGate` built per request;
a session pin that is merely kept is not a new placement and stays ungated by design.

The gate is a pure value so that every bound has exactly one implementation: the capability
filter, the plan-mode floor (the walk starts at the floor when the decided tier sits below it
and never steps under it), and the default_model arms (suppressed by a floor, and never offered
on plugin-configured routers, where default_model was never checked against the plugin
pipeline). An INACTIVE gate reproduces the router's pre-existing behavior identically, so the
call sites are unconditional and the flag-off path cannot drift from them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, NamedTuple

from litellm.types.utils import RoutingDecisionCause

from .config import ComplexityTier

DefaultModelPriority = Literal["first", "last", "never"]


def _name(tier: ComplexityTier | str) -> str:
    return tier.value if isinstance(tier, ComplexityTier) else tier


class ModalityPlacement(NamedTuple):
    """Where one request lands: a tier, or None when default_model serves it."""

    tier: ComplexityTier | str | None
    displaced_from: ComplexityTier | str | None
    displaced_default_model: bool

    @property
    def moved(self) -> bool:
        return self.displaced_from is not None or self.displaced_default_model


@dataclass(frozen=True, slots=True)
class ModalityGate:
    """One request's capability constraint, built by `ComplexityRouter._build_modality_gate`."""

    active: bool
    eligible: frozenset[str]
    tier_names: tuple[str, ...]
    pools: Mapping[str, Sequence[str]]
    default_model: str | None
    default_model_available: bool
    default_model_eligible: bool
    has_custom_tiers: bool
    router_name: str

    def place(
        self,
        decided: ComplexityTier | str,
        floor: ComplexityTier | str | None = None,
        default_priority: DefaultModelPriority = "last",
    ) -> ModalityPlacement:
        """The placement for a decided tier, under this request's capability constraint.

        `default_priority` is the seam's ORDERING intent only: "first" is the no-ask and
        classifier-failure default-first paths, "last" the ordinary tier-first paths, "never"
        the pin path. Whether default_model may serve at all is the gate's own call
        (configured, plugin-free, capability-eligible, and no floor, since default_model
        carries no tier guarantee a floor could vouch for).
        """
        default_first: Final = default_priority == "first" and self.default_model_available
        if not self.active:
            return ModalityPlacement(None if default_first else decided, None, False)
        default_usable: Final = (
            default_priority != "never"
            and self.default_model_available
            and self.default_model_eligible
            and floor is None
        )
        if default_first and default_usable:
            return ModalityPlacement(None, None, False)
        decided_key: Final = _name(decided)
        # A caller's stand-in for "the fallback pool" (the no-ask MEDIUM) is not a tier a
        # custom set defines; the walk then starts at the cheapest configured tier and reports
        # displacement against it, never against the stand-in's name.
        effective_key: Final = decided_key if decided_key in self.tier_names else self._cheapest_pooled_name()
        winner: Final = self._walk(effective_key, floor)
        displaced_default: Final = default_first and not default_usable
        if winner is not None:
            displaced_from: Final = self._resolve(effective_key) if winner != effective_key else None
            return ModalityPlacement(self._resolve(winner), displaced_from, displaced_default)
        if default_usable:
            return ModalityPlacement(None, self._resolve(effective_key), displaced_default)
        import litellm

        raise litellm.BadRequestError(
            message=(
                f"Auto-router {self.router_name} received a request with image input, but no model "
                f"in its tiers accepts images and modality_routing is enabled. "
                f"Tiers checked: {', '.join(self.tier_names)}. Add a vision-capable model to a tier, "
                f"or set a vision-capable default_model, or remove the image content."
            ),
            model=self.router_name,
            llm_provider="",
        )

    def decision_cause(self, placement: ModalityPlacement, base: RoutingDecisionCause) -> RoutingDecisionCause:
        return "modality_escalation" if self.active and placement.moved else base

    def decision_signals(
        self, placement: ModalityPlacement, base: tuple[str, ...] | None = None
    ) -> tuple[str, ...] | None:
        if not self.active:
            return base
        return (
            *(base or ()),
            "modality:image",
            *(
                (f"modality_escalated_from:{_name(placement.displaced_from)}",)
                if placement.displaced_from is not None
                else ()
            ),
            *(("modality_displaced_default_model",) if placement.displaced_default_model else ()),
        )

    def pool_filter(self) -> frozenset[str] | None:
        """The eligible set for the tier-to-model pickers, or None when the gate is inactive."""
        return self.eligible if self.active else None

    def _walk(self, effective_key: str, floor: ComplexityTier | str | None) -> str | None:
        names: Final = self.tier_names
        decided_idx: Final = names.index(effective_key) if effective_key in names else 0
        floor_key: Final = _name(floor) if floor is not None else None
        floor_idx: Final = names.index(floor_key) if floor_key is not None and floor_key in names else 0
        start_idx: Final = max(decided_idx, floor_idx)
        ordered: Final = (*names[start_idx:], *reversed(names[floor_idx:start_idx]))
        return next(
            (name for name in ordered if any(entry in self.eligible for entry in self.pools.get(name, ()))),
            None,
        )

    def _cheapest_pooled_name(self) -> str:
        return next((name for name in self.tier_names if self.pools.get(name)), self.tier_names[0])

    def _resolve(self, name: str) -> ComplexityTier | str:
        return name if self.has_custom_tiers else ComplexityTier(name)
