import keyword
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue, StringConstraints

JsonSchema: TypeAlias = dict[str, JsonValue]


def _identifier(value: str) -> str:
    if not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError("must be a Python identifier")
    return value


ProgramName: TypeAlias = Annotated[str, StringConstraints(min_length=1, max_length=128), AfterValidator(_identifier)]


class PythonProgram(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_version: Literal["1"]
    source: str
    function_name: ProgramName
    parameter_names: Annotated[tuple[ProgramName, ...], Field(min_length=1, max_length=32)]
    input_schema: JsonSchema
    output_schema: JsonSchema
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class MontyErrorKind(StrEnum):
    INVALID_PROGRAM = "INVALID_PROGRAM"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
    INPUT_INVALID = "INPUT_INVALID"
    INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
    OUTPUT_INVALID = "OUTPUT_INVALID"
    OUTPUT_TOO_LARGE = "OUTPUT_TOO_LARGE"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TIMEOUT = "TIMEOUT"
    MEMORY_LIMIT = "MEMORY_LIMIT"
    RECURSION_LIMIT = "RECURSION_LIMIT"
    STDOUT_LIMIT = "STDOUT_LIMIT"
    WORKER_CRASHED = "WORKER_CRASHED"
    QUEUE_FULL = "QUEUE_FULL"
    PREDICATE_NOT_BOOL = "PREDICATE_NOT_BOOL"
    CANCELLED = "CANCELLED"
    RUNTIME_UNAVAILABLE = "RUNTIME_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class MontyFailure:
    kind: MontyErrorKind
    message: str


@dataclass(frozen=True, slots=True)
class MontySuccess:
    output: JsonValue
    stdout: str
    duration_ms: int


@dataclass(frozen=True, slots=True)
class CompiledProgram:
    program: PythonProgram
    sandbox_source: str
    input_schema: JsonSchema
    output_schema: JsonSchema


class MontyLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    max_source_bytes: int = Field(default=64 * 1024, gt=0, le=1024 * 1024)
    max_input_bytes: int = Field(default=1024 * 1024, gt=0, le=16 * 1024 * 1024)
    max_output_bytes: int = Field(default=1024 * 1024, gt=0, le=16 * 1024 * 1024)
    max_input_depth: int = Field(default=64, gt=0, le=128)
    max_output_depth: int = Field(default=64, gt=0, le=128)
    max_memory: int = Field(default=64 * 1024 * 1024, gt=0, le=1024 * 1024 * 1024)
    worker_address_space_bytes: int = Field(default=512 * 1024 * 1024, gt=0, le=8 * 1024 * 1024 * 1024)
    feed_seconds: float = Field(default=5, gt=0, le=60)
    max_recursion_depth: int = Field(default=200, gt=0, le=1000)
    max_stdout_bytes: int = Field(default=64 * 1024, gt=0, le=1024 * 1024)
    max_concurrent_workers: int = Field(default=4, gt=0, le=64)
    max_queued_waiters: int = Field(default=16, gt=0, le=1024)
