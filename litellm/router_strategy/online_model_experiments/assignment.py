from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from itertools import accumulate
from typing import Final, Literal, TypeAlias

ALLOCATION_TOTAL_BASIS_POINTS: Final[int] = 10_000
AssignmentErrorCode: TypeAlias = Literal[
    "empty_experiment_id",
    "empty_identity",
    "empty_secret",
    "invalid_variants",
    "invalid_weights",
]


@dataclass(frozen=True, slots=True)
class ExperimentVariant:
    name: str
    weight_basis_points: int
    model_name: str = ""


@dataclass(frozen=True, slots=True)
class ExperimentAssignment:
    variant_name: str
    bucket: int


@dataclass(frozen=True, slots=True)
class ExperimentAssignmentError:
    code: AssignmentErrorCode
    message: str


AssignmentKeyResult: TypeAlias = str | ExperimentAssignmentError
AssignmentResult: TypeAlias = ExperimentAssignment | ExperimentAssignmentError


def derive_assignment_key(
    experiment_id: str,
    identity: str,
    secret: str,
) -> AssignmentKeyResult:
    if not experiment_id:
        return ExperimentAssignmentError("empty_experiment_id", "experiment_id must not be empty")
    if not identity:
        return ExperimentAssignmentError("empty_identity", "identity must not be empty")
    if not secret:
        return ExperimentAssignmentError("empty_secret", "secret must not be empty")
    material: Final = f"{experiment_id}\x00{identity}".encode()
    return hmac.new(secret.encode(), material, hashlib.sha256).hexdigest()


def assign_variant(
    experiment_id: str,
    assignment_key: str,
    variants: tuple[ExperimentVariant, ...],
) -> AssignmentResult:
    if not experiment_id:
        return ExperimentAssignmentError("empty_experiment_id", "experiment_id must not be empty")
    if not assignment_key:
        return ExperimentAssignmentError("empty_identity", "assignment_key must not be empty")
    if not variants or len({variant.name for variant in variants}) != len(variants):
        return ExperimentAssignmentError("invalid_variants", "variants must be non-empty and uniquely named")
    if any(variant.weight_basis_points <= 0 for variant in variants):
        return ExperimentAssignmentError("invalid_weights", "variant weights must be positive")
    if sum(variant.weight_basis_points for variant in variants) != ALLOCATION_TOTAL_BASIS_POINTS:
        return ExperimentAssignmentError("invalid_weights", "variant weights must sum to 10000 basis points")

    digest: Final = hashlib.sha256(f"{experiment_id}\x00{assignment_key}".encode()).digest()
    bucket: Final = int.from_bytes(digest[:8], byteorder="big") % ALLOCATION_TOTAL_BASIS_POINTS
    upper_bounds: Final = tuple(accumulate(variant.weight_basis_points for variant in variants))
    selected: Final = next(
        variant for variant, upper_bound in zip(variants, upper_bounds) if bucket < upper_bound
    )
    return ExperimentAssignment(variant_name=selected.name, bucket=bucket)
