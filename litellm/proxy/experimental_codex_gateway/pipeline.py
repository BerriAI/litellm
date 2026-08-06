import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm.proxy.experimental_codex_gateway.types import JsonValue


class StageOutcome(str, Enum):
    UNCHANGED = "unchanged"
    TRANSFORMED = "transformed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StageAudit:
    stage: str
    outcome: StageOutcome
    findings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Unchanged:
    body: bytes
    audit: StageAudit


@dataclass(frozen=True, slots=True)
class Transformed:
    body: bytes
    audit: StageAudit


@dataclass(frozen=True, slots=True)
class Failed:
    original_body: bytes
    audit: StageAudit


StageResult: TypeAlias = Unchanged | Transformed | Failed


class RequestStage(Protocol):
    def apply(self, body: bytes, content_type: str) -> StageResult: ...


_SECRET_PATTERN = re.compile(r"(?i)(bearer\s+\S+|sk-[a-z0-9_-]{8,}|api[_-]?key|password|secret)")
_EMAIL_PATTERN = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)")
_INJECTION_PATTERN = re.compile(r"(?i)(ignore (?:all |the )?previous|system prompt|developer message)")
_JSON_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


@dataclass(frozen=True, slots=True)
class InspectionStage:
    name: str = "request_inspection"

    def apply(self, body: bytes, content_type: str) -> StageResult:
        if "json" not in content_type.lower() or not body:
            return Unchanged(body=body, audit=StageAudit(stage=self.name, outcome=StageOutcome.UNCHANGED))
        try:
            parsed = _JSON_ADAPTER.validate_json(body)
        except ValidationError:
            return Failed(
                original_body=body,
                audit=StageAudit(stage=self.name, outcome=StageOutcome.FAILED, findings=("invalid_json",)),
            )
        serialized = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        findings = tuple(
            finding
            for finding, pattern in (
                ("secret", _SECRET_PATTERN),
                ("pii_email", _EMAIL_PATTERN),
                ("pii_phone", _PHONE_PATTERN),
                ("prompt_injection", _INJECTION_PATTERN),
            )
            if pattern.search(serialized)
        )
        return Unchanged(
            body=body,
            audit=StageAudit(stage=self.name, outcome=StageOutcome.UNCHANGED, findings=findings),
        )


@dataclass(frozen=True, slots=True)
class PipelineResult:
    original_body: bytes
    body: bytes
    outcome: StageOutcome
    audit: tuple[StageAudit, ...]


@dataclass(frozen=True, slots=True)
class RequestPipeline:
    stages: tuple[RequestStage, ...]

    def process(self, body: bytes, content_type: str) -> PipelineResult:
        current = body
        audits: tuple[StageAudit, ...] = ()
        outcome = StageOutcome.UNCHANGED
        for stage in self.stages:
            result = stage.apply(current, content_type)
            audits = (*audits, result.audit)
            match result:
                case Failed():
                    return PipelineResult(
                        original_body=body,
                        body=body,
                        outcome=StageOutcome.FAILED,
                        audit=audits,
                    )
                case Transformed():
                    current = result.body
                    outcome = StageOutcome.TRANSFORMED
                case Unchanged():
                    current = result.body
        return PipelineResult(original_body=body, body=current, outcome=outcome, audit=audits)
