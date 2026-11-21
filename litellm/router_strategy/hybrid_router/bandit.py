"""
Multi-armed bandit algorithms for the hybrid router.

Three algorithms ship: UCB, EpsilonGreedy, and ThompsonSampling.
ThompsonSampling reuses the BanditCell and thompson_sample from the
adaptive_router module. All three expose the same interface so the
hybrid router can swap algorithms via config.
"""

from __future__ import annotations

import math
import random
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.router_strategy.adaptive_router.bandit import (
    BanditCell,
    apply_delta,
    thompson_sample,
)


@dataclass(frozen=True, slots=True)
class ArmStats:
    """Frequentist statistics for one arm (UCB / EpsilonGreedy)."""

    count: int
    sum_rewards: float

    @property
    def mean(self) -> float:
        return self.sum_rewards / self.count if self.count > 0 else 0.0


class MABAlgorithm(ABC):
    """Thread-safe bandit base class."""

    def __init__(self, arms: tuple[str, ...]) -> None:
        if len(arms) < 2:
            raise ValueError("Need at least 2 arms")
        self._arms: Final = arms
        self._lock: Final = threading.Lock()

    @property
    def arms(self) -> tuple[str, ...]:
        return self._arms

    @abstractmethod
    def select_arm(self) -> str: ...

    @abstractmethod
    def select_arm_from(self, eligible: tuple[str, ...]) -> str: ...

    @abstractmethod
    def update(self, arm: str, reward: float) -> None: ...

    @abstractmethod
    def state(self) -> dict[str, object]: ...


class UCB(MABAlgorithm):
    """Upper Confidence Bound (Hoeffding-style).

    Select argmax(mu_i + sqrt(2 * ln(1/delta) / n_i)).
    Smaller delta = wider interval = more exploration.
    """

    def __init__(self, arms: tuple[str, ...], *, delta: float = 0.1) -> None:
        super().__init__(arms)
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0, 1)")
        self._delta: Final = delta
        self._stats: dict[str, ArmStats] = {arm: ArmStats(count=0, sum_rewards=0.0) for arm in arms}
        self._t = 0

    def _ucb_score(self, arm: str) -> float | None:
        stats: Final = self._stats[arm]
        if stats.count == 0:
            return None
        return stats.mean + math.sqrt(2.0 * math.log(1.0 / self._delta) / stats.count)

    def select_arm(self) -> str:
        with self._lock:
            return self._pick(self._arms)

    def select_arm_from(self, eligible: tuple[str, ...]) -> str:
        with self._lock:
            return self._pick(eligible)

    def _pick(self, candidates: tuple[str, ...]) -> str:
        best_arm: str = candidates[0]
        best_score: float = -math.inf
        for arm in candidates:
            stats: Final = self._stats[arm]
            if stats.count == 0:
                return arm
            score: Final = stats.mean + math.sqrt(2.0 * math.log(1.0 / self._delta) / stats.count)
            if score > best_score:
                best_score = score
                best_arm = arm
        return best_arm

    def update(self, arm: str, reward: float) -> None:
        with self._lock:
            old: Final = self._stats[arm]
            self._stats[arm] = ArmStats(count=old.count + 1, sum_rewards=old.sum_rewards + reward)
            self._t += 1

    def state(self) -> dict[str, object]:
        with self._lock:
            return {
                "algorithm": "UCB",
                "arms": list(self._arms),
                "t": self._t,
                "counts": {arm: self._stats[arm].count for arm in self._arms},
                "means": {arm: self._stats[arm].mean for arm in self._arms},
            }


class EpsilonGreedy(MABAlgorithm):
    """Epsilon-greedy: explore uniformly with probability epsilon, else greedy."""

    def __init__(self, arms: tuple[str, ...], *, epsilon: float = 0.1) -> None:
        super().__init__(arms)
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        self._epsilon: Final = epsilon
        self._stats: dict[str, ArmStats] = {arm: ArmStats(count=0, sum_rewards=0.0) for arm in arms}
        self._t = 0

    def select_arm(self) -> str:
        with self._lock:
            return self._pick(self._arms)

    def select_arm_from(self, eligible: tuple[str, ...]) -> str:
        with self._lock:
            return self._pick(eligible)

    def _pick(self, candidates: tuple[str, ...]) -> str:
        for arm in candidates:
            if self._stats[arm].count == 0:
                return arm
        if random.random() < self._epsilon:
            return random.choice(candidates)
        best_arm: str = candidates[0]
        best_mean: float = -math.inf
        for arm in candidates:
            mean: Final = self._stats[arm].mean
            if mean > best_mean:
                best_mean = mean
                best_arm = arm
        return best_arm

    def update(self, arm: str, reward: float) -> None:
        with self._lock:
            old: Final = self._stats[arm]
            self._stats[arm] = ArmStats(count=old.count + 1, sum_rewards=old.sum_rewards + reward)
            self._t += 1

    def state(self) -> dict[str, object]:
        with self._lock:
            return {
                "algorithm": "EpsilonGreedy",
                "arms": list(self._arms),
                "t": self._t,
                "counts": {arm: self._stats[arm].count for arm in self._arms},
                "means": {arm: self._stats[arm].mean for arm in self._arms},
            }


class ThompsonSampling(MABAlgorithm):
    """Beta-Bernoulli Thompson Sampling.

    Reuses BanditCell and thompson_sample from the adaptive_router module.
    Rewards in [0, 1] are treated as fractional: alpha += reward, beta += (1 - reward).
    """

    def __init__(
        self,
        arms: tuple[str, ...],
        *,
        arm_priors: Mapping[str, tuple[float, float]] | None = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        super().__init__(arms)
        self._cells: dict[str, BanditCell] = {}
        for arm in arms:
            if arm_priors and arm in arm_priors:
                alpha, beta = arm_priors[arm]
            else:
                alpha, beta = prior_alpha, prior_beta
            self._cells[arm] = BanditCell(alpha=alpha, beta=beta)
        self._t = 0

    def select_arm(self) -> str:
        with self._lock:
            return self._pick(self._arms)

    def select_arm_from(self, eligible: tuple[str, ...]) -> str:
        with self._lock:
            return self._pick(eligible)

    def _pick(self, candidates: tuple[str, ...]) -> str:
        best_arm: str = candidates[0]
        best_sample: float = -1.0
        for arm in candidates:
            sample: Final = thompson_sample(self._cells[arm])
            if sample > best_sample:
                best_sample = sample
                best_arm = arm
        return best_arm

    def update(self, arm: str, reward: float) -> None:
        with self._lock:
            cell: Final = self._cells[arm]
            self._cells[arm] = BanditCell(alpha=cell.alpha + reward, beta=cell.beta + (1.0 - reward))
            self._t += 1

    def state(self) -> dict[str, object]:
        with self._lock:
            return {
                "algorithm": "ThompsonSampling",
                "arms": list(self._arms),
                "t": self._t,
                "counts": {arm: int(self._cells[arm].alpha + self._cells[arm].beta - 2.0) for arm in self._arms},
                "means": {arm: self._cells[arm].mean for arm in self._arms},
                "alpha": {arm: self._cells[arm].alpha for arm in self._arms},
                "beta": {arm: self._cells[arm].beta for arm in self._arms},
            }


BANDIT_REGISTRY: Final[Mapping[str, Callable[..., MABAlgorithm]]] = MappingProxyType(
    {
        "ucb": lambda arms, *, delta=0.1, **_kw: UCB(arms, delta=delta),
        "epsilon_greedy": lambda arms, *, epsilon=0.1, **_kw: EpsilonGreedy(arms, epsilon=epsilon),
        "thompson": lambda arms, *, arm_priors=None, prior_alpha=1.0, prior_beta=1.0, **_kw: ThompsonSampling(
            arms, arm_priors=arm_priors, prior_alpha=prior_alpha, prior_beta=prior_beta
        ),
    }
)


def make_bandit(name: str, arms: tuple[str, ...], **kwargs: object) -> MABAlgorithm:
    if name not in BANDIT_REGISTRY:
        raise ValueError(f"Unknown bandit {name!r}. Available: {sorted(BANDIT_REGISTRY)}")
    return BANDIT_REGISTRY[name](arms, **kwargs)
