from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from sys import float_info
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

Finite: TypeAlias = Annotated[float, Field(allow_inf_nan=False)]
Outcome: TypeAlias = Literal[0, 1, 2, 3]
CONCEPTS: Final = (
    "async",
    "concurrent",
    "recursive",
    "dynamic programming",
    "graph",
    "database",
    "parser",
    "unicode",
    "serialization",
    "numerical",
    "memory",
    "sorting",
    "regex",
    "migration",
    "compatibility",
    "race",
    "edge case",
    "floating",
    "type",
    "security",
    "compile",
    "constraint",
    "optimiz",
    "algorithm",
)


def verdict_features(crux: str, values: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType(
        {
            "crux_log_length": math.log1p(len(crux)) / 10.0,
            **values,
            **{f"concept:{concept}": float(concept in crux.lower()) for concept in CONCEPTS},
        }
    )


class SelectiveHead(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    features: tuple[str, ...] = Field(default=(), max_length=128)
    coefficients: tuple[tuple[Finite, ...], ...] = Field(default=(), max_length=4)
    intercept: tuple[Finite, ...] = Field(default=(), max_length=4)
    classes: tuple[Outcome, ...] = Field(default=(), max_length=4)
    constant: Outcome | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> SelectiveHead:
        if self.constant is not None:
            if self.features or self.coefficients or self.intercept or self.classes:
                raise ValueError("A constant head cannot also contain fitted coefficients")
            return self
        if not self.features or len(frozenset(self.features)) != len(self.features):
            raise ValueError("A fitted head requires unique feature names")
        if len(frozenset(self.classes)) != len(self.classes) or len(self.classes) == 1:
            raise ValueError("Classifier heads require two to four distinct classes")
        rows: Final = len(self.classes) if len(self.classes) > 2 else 1
        if len(self.coefficients) != rows or len(self.intercept) != rows:
            raise ValueError("Coefficient rows and intercepts must match the head output shape")
        if any(len(row) != len(self.features) for row in self.coefficients):
            raise ValueError("Every coefficient row must match the feature names")
        return self

    def linear(self, values: Mapping[str, float]) -> tuple[float, ...]:
        return tuple(
            sum(coefficient * values.get(feature, 0.0) for feature, coefficient in zip(self.features, row)) + intercept
            for row, intercept in zip(self.coefficients, self.intercept)
        )

    def probabilities(self, values: Mapping[str, float]) -> tuple[float, ...]:
        if self.constant is not None:
            return tuple(float(self.constant == outcome) for outcome in (0, 1, 2, 3))
        logits: Final = self.linear(values)
        if len(self.classes) == 2:
            positive: Final = (
                1.0 / (1.0 + math.exp(-logits[0]))
                if logits[0] >= 0.0
                else math.exp(logits[0]) / (1.0 + math.exp(logits[0]))
            )
            distribution: Final = (1.0 - positive, positive)
            return tuple(
                distribution[self.classes.index(outcome)] if outcome in self.classes else 0.0
                for outcome in (0, 1, 2, 3)
            )
        peak: Final = max(logits)
        weights: Final = tuple(math.exp(value - peak) for value in logits)
        total: Final = sum(weights)
        return tuple(
            weights[self.classes.index(outcome)] / total if outcome in self.classes else 0.0 for outcome in (0, 1, 2, 3)
        )


@dataclass(frozen=True, slots=True)
class SelectiveDecision:
    score: float
    threshold: float
    version: str
    target: str

    @property
    def use_efficient(self) -> bool:
        return math.isfinite(self.score) and self.score <= self.threshold + float_info.epsilon

    @property
    def signals(self) -> tuple[str, ...]:
        return (
            f"selective:version={self.version}",
            f"selective:target={self.target}",
            f"selective:score={self.score:.8f}",
            f"selective:threshold={self.threshold:.8f}",
        )


class SelectivePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1, max_length=128)
    feature_schema: Literal["cap-v1", "v2-v1"]
    target: Literal["paired", "rescue", "per_model", "scalar_calibration", "benefit_per_dollar"]
    threshold: Finite
    heads: tuple[SelectiveHead, ...] = Field(min_length=1, max_length=2)
    costs: tuple[SelectiveHead, ...] = Field(default=(), max_length=2)

    @model_validator(mode="after")
    def _validate_heads(self) -> SelectivePolicy:
        required: Final = 2 if self.target in ("per_model", "scalar_calibration") else 1
        if len(self.heads) != required or len(self.costs) != (2 if self.target == "benefit_per_dollar" else 0):
            raise ValueError("Head counts must match the selective policy target")
        if any(not head.classes and head.constant is None for head in self.heads):
            raise ValueError("Quality heads must contain classes or a constant outcome")
        if any(head.classes or head.constant is not None for head in self.costs):
            raise ValueError("Cost heads must be linear regressions")
        if self.target in ("rescue", "per_model", "scalar_calibration") and any(
            any(outcome not in (0, 1) for outcome in head.classes)
            or (head.constant is not None and head.constant not in (0, 1))
            for head in self.heads
        ):
            raise ValueError("Binary quality heads require outcomes zero and one")
        return self

    def _score(self, features: Mapping[str, float], efficient: float, capable: float) -> float:
        if self.target in ("per_model", "scalar_calibration"):
            probabilities: Final = tuple(
                head.probabilities(
                    MappingProxyType(
                        {"logit_p": math.log(min(max(raw, 1e-5), 1.0 - 1e-5) / (1.0 - min(max(raw, 1e-5), 1.0 - 1e-5)))}
                    )
                    if self.target == "scalar_calibration"
                    else features
                )[1]
                for head, raw in zip(self.heads, (efficient, capable))
            )
            return probabilities[1] - probabilities[0]
        paired: Final = self.heads[0].probabilities(features)
        if self.target == "rescue":
            return paired[1]
        benefit: Final = paired[1] - paired[2]
        if self.target == "benefit_per_dollar":
            predicted_costs: Final = tuple(math.exp(head.linear(features)[0]) for head in self.costs)
            return benefit / max(0.001, predicted_costs[1] - predicted_costs[0])
        return benefit

    def evaluate(self, features: Mapping[str, float], efficient: float, capable: float) -> SelectiveDecision:
        try:
            return SelectiveDecision(
                self._score(features, efficient, capable), self.threshold, self.version, self.target
            )
        except (OverflowError, ZeroDivisionError):
            return SelectiveDecision(math.inf, self.threshold, self.version, self.target)
