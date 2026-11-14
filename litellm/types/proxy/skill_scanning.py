from datetime import datetime
from typing import Annotated, Final, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SkillScannerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint: str = Field(max_length=2048, description="External scanner's authenticated POST endpoint")
    api_key_env: str = Field(
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", description="Environment secret name for scanner auth"
    )
    use_llm: bool = False
    timeout_seconds: int = Field(default=180, ge=10, le=600)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        parsed: Final = urlsplit(value)
        if (
            not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not (
                parsed.scheme == "https"
                or (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1", "localhost"})
            )
        ):
            raise ValueError(
                "Scanner endpoint must use HTTPS without credentials, query or fragment (HTTP loopback allowed)"
            )
        return value


class SkillScanFinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    rule_id: str = Field(max_length=200)
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    message: str = Field(max_length=8000)
    file: str = Field(max_length=2048)
    start_line: int | None = Field(default=None, ge=1)
    explanation: str | None = Field(default=None, max_length=8000)
    remediation: str | None = Field(default=None, max_length=8000)


class SkillScanReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    scanner_version: str = Field(min_length=1, max_length=100)
    complete: bool
    scan_mode: Literal["static-only", "static+llm"]
    risk_score: int = Field(ge=0, le=100)
    findings: tuple[SkillScanFinding, ...] = Field(max_length=500)
    limitations: tuple[Annotated[str, Field(max_length=2000)], ...] = Field(default=(), max_length=100)


class SkillScan(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    status: Literal["running", "completed", "failed"]
    requested_by: str
    started_at: datetime
    expires_at: datetime
    completed_at: datetime | None = None
    archive_sha256: str | None = None
    report: SkillScanReport | None = None
    error: str | None = None

    def at(self, now: datetime) -> "SkillScan":
        if self.status == "running" and self.expires_at <= now:
            return self.model_copy(
                update={"status": "failed", "error": "Scan timed out or was interrupted. Try scanning again."}
            )
        return self


class SkillScanRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SkillScanResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    configured: bool
    scan: SkillScan | None
