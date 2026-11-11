from dataclasses import dataclass

from pydantic import BaseModel


class Readiness(BaseModel):
    status: str = ""
    db: str = ""


class ContainerState(BaseModel):
    Running: bool
    ExitCode: int


@dataclass(frozen=True, slots=True)
class Observation:
    exit_code: int | None
    ready: bool


@dataclass(frozen=True, slots=True)
class Migration:
    name: str
    script: str
