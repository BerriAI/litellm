from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

JsonSchema = dict[str, JsonValue]


class PythonProgram(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract_version: Literal["1"]
    source: str
    function_name: str
    parameter_names: Annotated[tuple[str, ...], Field(min_length=1)]
    input_schema: JsonSchema
    output_schema: JsonSchema
    source_sha256: str


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
    model_config = ConfigDict(frozen=True)

    max_source_bytes: int = Field(default=64 * 1024, gt=0)
    max_input_bytes: int = Field(default=1024 * 1024, gt=0)
    max_output_bytes: int = Field(default=1024 * 1024, gt=0)
    max_input_depth: int = Field(default=64, gt=0)
    max_output_depth: int = Field(default=64, gt=0)
    max_memory: int = Field(default=64 * 1024 * 1024, gt=0)
    worker_address_space_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    feed_seconds: float = Field(default=5, gt=0)
    max_recursion_depth: int = Field(default=200, gt=0)
    max_stdout_bytes: int = Field(default=64 * 1024, gt=0)
    max_concurrent_workers: int = Field(default=4, gt=0)
    max_queued_waiters: int = Field(default=16, ge=0)
