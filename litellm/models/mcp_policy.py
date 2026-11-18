import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, field_validator


@dataclass(frozen=True, slots=True)
class PolicySubject:
    """Whose policy is being validated: a virtual key (its hash), or a keyless user (and the team and organization its token pins)."""

    key_hash: str | None = None
    user_id: str | None = None
    pinned_team_id: str | None = None
    pinned_org_id: str | None = None

    @property
    def mark(self) -> str:
        return (
            f"key:{self.key_hash}"
            if self.key_hash is not None
            else f"user:{self.user_id}:{self.pinned_team_id}:{self.pinned_org_id}"
        )


class ReachableObjects(BaseModel):
    """Every cached auth object the subject can read, including ids whose rows no longer exist."""

    model_config = ConfigDict(frozen=True)

    user_id: str | None = None
    project_id: str | None = None
    team_ids: tuple[str, ...] = ()
    organization_ids: tuple[str, ...] = ()
    permission_ids: tuple[str, ...] = ()
    access_group_ids: tuple[str, ...] = ()


class PolicyFingerprint(BaseModel):
    model_config = ConfigDict(frozen=True)

    digest: str
    uses_toolsets: bool
    reachable: ReachableObjects

    @field_validator("reachable", mode="before")
    @classmethod
    def _decode_reachable(cls, value: object) -> object:
        return json.loads(value) if isinstance(value, str) else value
