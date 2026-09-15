from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

from capture_policy import (
    HARD_AGE_SECONDS,
    SCENARIO_BYTES,
    SOFT_AGE_SECONDS,
    ScenarioIdentity,
    ScenarioOutcome,
    canonical_scenario_id,
    publication_error,
)
from capture_session import CaptureResult
from fixture_bundle import Interaction, LoadedBundle, Manifest, interaction_filename, slug_for_test
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class CaptureProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    test_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    candidate_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    runner_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")


class ScenarioSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    identity: ScenarioIdentity
    manifest: Manifest
    interactions: tuple[Interaction, ...]
    provenance: CaptureProvenance
    owner: str
    attempts: tuple[str, ...]
    outcome: ScenarioOutcome


@dataclass(frozen=True, slots=True)
class SnapshotFailure:
    reason: str


def build_snapshot(
    result: CaptureResult, bundle: LoadedBundle, provenance: CaptureProvenance
) -> bytes | SnapshotFailure:
    if not result.publishable:
        return SnapshotFailure(result.error or "capture was not approved")
    node: Final = canonical_scenario_id(result.identity.node)
    interactions: Final = bundle.interactions.get(slug_for_test(node), ())
    if result.response_count != len(interactions) or result.response_count != len(result.attempts):
        return SnapshotFailure("capture interaction and attempt counts differ")
    if bundle.manifest.match_profile != "stateless_v1" or bundle.manifest.format_version != 5:
        return SnapshotFailure("capture requires a strict bundle")
    if any(interaction.request.strict_identity is None for interaction in interactions):
        return SnapshotFailure("capture contains a legacy request")
    error: Final = publication_error(result.outcome, tuple(i.response for i in interactions))
    if error is not None:
        return SnapshotFailure(error)
    encoded: Final = (
        ScenarioSnapshot(
            identity=result.identity,
            manifest=bundle.manifest,
            interactions=interactions,
            provenance=provenance,
            owner=result.owner,
            attempts=result.attempts,
            outcome=result.outcome,
        )
        .model_dump_json()
        .encode()
    )
    return encoded if len(encoded) <= SCENARIO_BYTES else SnapshotFailure("scenario exceeds capture byte limit")


def verify_snapshot(
    content: bytes,
    *,
    expected_sha256: str,
    identity: ScenarioIdentity,
    now: datetime,
) -> ScenarioSnapshot | SnapshotFailure:
    if len(content) > SCENARIO_BYTES or hashlib.sha256(content).hexdigest() != expected_sha256:
        return SnapshotFailure("snapshot size or digest mismatch")
    try:
        snapshot: Final = ScenarioSnapshot.model_validate_json(content)
    except (ValueError, ValidationError):
        return SnapshotFailure("snapshot schema invalid")
    if (
        snapshot.identity.key != identity.key
        or snapshot.manifest.match_profile != "stateless_v1"
        or snapshot.manifest.format_version != 5
    ):
        return SnapshotFailure("snapshot identity or matcher mismatch")
    recorded_at: Final = snapshot.manifest.recorded_at
    if recorded_at.tzinfo is None or now.tzinfo is None:
        return SnapshotFailure("snapshot timestamps must include a timezone")
    age: Final = (now - recorded_at).total_seconds()
    if age < 0 or age >= HARD_AGE_SECONDS:
        return SnapshotFailure("snapshot is outside its hard age limit")
    if (
        not snapshot.interactions
        or len(snapshot.interactions) != len(snapshot.attempts)
        or len(set(snapshot.attempts)) != len(snapshot.attempts)
    ):
        return SnapshotFailure("snapshot interaction and attempt counts differ")
    if any(i.request.strict_identity is None for i in snapshot.interactions):
        return SnapshotFailure("snapshot contains a legacy request")
    error: Final = publication_error(snapshot.outcome, tuple(i.response for i in snapshot.interactions))
    if error is not None:
        return SnapshotFailure(error)
    return snapshot


def refresh_due(snapshot: ScenarioSnapshot, *, now: datetime) -> bool:
    return (now - snapshot.manifest.recorded_at).total_seconds() >= SOFT_AGE_SECONDS


def materialize_snapshot(snapshot: ScenarioSnapshot, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "manifest.json").write_text(snapshot.manifest.model_dump_json(), encoding="utf-8")
    scenario_dir: Final = destination / slug_for_test(canonical_scenario_id(snapshot.identity.node))
    scenario_dir.mkdir()
    for ordinal, interaction in enumerate(snapshot.interactions):
        (scenario_dir / interaction_filename(ordinal, interaction.request)).write_text(
            interaction.model_dump_json(), encoding="utf-8"
        )
