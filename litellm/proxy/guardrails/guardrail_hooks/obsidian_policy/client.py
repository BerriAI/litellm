import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm._logging import verbose_proxy_logger

DEFAULT_ENDPOINT: Final = "https://agent-guard.obsec.io/webhook/unified/v1/enforcement"
DEFAULT_CLIENT_PLATFORM: Final = "litellm_gateway"
DEFAULT_TIMEOUT_SECONDS: Final = 2.5
MAX_RETRIES: Final = 1

RESERVED_CLIENT_PLATFORMS: Final = frozenset(
    {"claudecode", "codex", "copilot", "copilot_studio", "cursor", "mcpguard", "mcp_gateway", "n8n"}
)
CLIENT_PLATFORM_PATTERN: Final = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

Stage: TypeAlias = Literal["PreToolUse", "PostToolUse"]
DecisionKind: TypeAlias = Literal["pass", "warn", "block", "not_evaluated", "error"]


def validate_client_platform(name: str) -> str | None:
    """Return an error message if `name` is not usable as a client platform, else None."""
    if not CLIENT_PLATFORM_PATTERN.fullmatch(name):
        return 'client_platform must be 1-64 letters, digits, ".", "_" or "-"'
    if name.lower().replace("-", "_") in RESERVED_CLIENT_PLATFORMS:
        return f'client_platform "{name}" is reserved by Obsidian'
    return None


@dataclass(frozen=True, slots=True)
class ObsidianSettings:
    endpoint: str
    gateway_token: str = field(repr=False)
    client_platform: str
    timeout_seconds: float
    fail_open: bool


@dataclass(frozen=True, slots=True)
class InvalidSettings:
    reason: str
    fail_open: bool


@dataclass(frozen=True, slots=True)
class ToolEvent:
    stage: Stage
    server_name: str
    tool_name: str
    arguments: Mapping[str, object]
    session_id: str
    user_id: str | None = None
    user_email: str | None = None
    tool_response: object | None = None


class MatchedPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str | None = None
    name: str | None = None
    action: str | None = None


class _Verdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    decision: str = ""
    evaluation_id: str | None = None
    message: str | None = None
    matched_policies: tuple[MatchedPolicy, ...] = Field(default=())


class _ErrorInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    code: str | None = None
    message: str | None = None


class _ErrorBody(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    error: _ErrorInfo | None = None


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    kind: DecisionKind
    message: str | None = None
    evaluation_id: str | None = None
    matched_policies: tuple[MatchedPolicy, ...] = ()
    via_fail_mode: bool = False


@dataclass(frozen=True, slots=True)
class _Failure:
    message: str


def build_event(
    event: ToolEvent,
    client_platform: str,
    event_id: str | None = None,
    timestamp: str | None = None,
) -> Mapping[str, object]:
    action_param: Final = {
        "tool_name": event.tool_name,
        "args": dict(event.arguments),
        **(
            {"tool_response": event.tool_response if event.tool_response not in (None, "") else {}}
            if event.stage == "PostToolUse"
            else {}
        ),
    }
    user: Final = {key: value for key, value in (("id", event.user_id), ("email", event.user_email)) if value}
    return {
        "version": "1.0",
        "event_id": event_id or str(uuid.uuid4()),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "client_platform": client_platform,
        "stage": event.stage,
        "session_id": event.session_id,
        "target_service": event.server_name,
        "action_param": action_param,
        **({"user": user} if user else {}),
    }


class ObsidianClient:
    def __init__(self, settings: ObsidianSettings | InvalidSettings, http: httpx.AsyncClient) -> None:
        self._settings: Final = settings
        self._http: Final = http

    @property
    def fail_open(self) -> bool:
        return self._settings.fail_open

    async def evaluate(self, event: ToolEvent) -> PolicyDecision:
        """Ask Obsidian for a verdict on one tool event. Never raises."""
        match self._settings:
            case InvalidSettings():
                return self.failure("error", "the guardrail configuration is invalid, see the proxy log")
            case ObsidianSettings() as settings:
                outcome: Final = await self._send(settings, build_event(event, settings.client_platform))
        match outcome:
            case _Failure(message=message):
                return self.failure("error", message)
            case httpx.Response():
                return await self._decide(outcome)

    async def _send(
        self, settings: ObsidianSettings, body: Mapping[str, object], attempt: int = 0
    ) -> httpx.Response | _Failure:
        try:
            response: Final = await self._http.post(
                settings.endpoint,
                json=body,
                headers=MappingProxyType(
                    {
                        "Authorization": f"Bearer {settings.gateway_token}",
                        "User-Agent": f"{settings.client_platform}/1.0",
                    }
                ),
                timeout=settings.timeout_seconds,
            )
        except httpx.TimeoutException:
            return _Failure(f"Obsidian evaluation timed out after {int(settings.timeout_seconds * 1000)}ms")
        except httpx.TransportError as exc:
            if attempt < MAX_RETRIES:
                return await self._send(settings, body, attempt + 1)
            return _Failure(f"Obsidian request failed: {type(exc).__name__}")
        if response.status_code >= 500 and attempt < MAX_RETRIES:
            return await self._send(settings, body, attempt + 1)
        return response

    async def _decide(self, response: httpx.Response) -> PolicyDecision:
        if not response.is_success:
            return self.failure("error", f"Obsidian returned HTTP {response.status_code}{_error_detail(response)}")
        try:
            verdict: Final = _Verdict.model_validate_json(response.content)
        except ValidationError:
            return self.failure("error", "Obsidian returned a response that is not a valid verdict")

        if verdict.decision in ("pass", "warn"):
            if not verdict.evaluation_id:
                return self.failure(
                    "not_evaluated",
                    verdict.message or "Obsidian did not evaluate the event",
                    matched_policies=verdict.matched_policies,
                )
            return PolicyDecision(
                allowed=True,
                kind=verdict.decision,
                message=verdict.message,
                evaluation_id=verdict.evaluation_id,
                matched_policies=verdict.matched_policies,
            )

        return PolicyDecision(
            allowed=False,
            kind="block",
            message=verdict.message
            or (
                "Blocked by policy"
                if verdict.decision == "block"
                else f'Unrecognised Obsidian decision "{verdict.decision}"'
            ),
            evaluation_id=verdict.evaluation_id,
            matched_policies=verdict.matched_policies,
        )

    def failure(
        self,
        kind: Literal["error", "not_evaluated"],
        message: str,
        matched_policies: tuple[MatchedPolicy, ...] = (),
    ) -> PolicyDecision:
        verbose_proxy_logger.warning(
            "Obsidian policy %s, fail_open=%s: %s", kind, self._settings.fail_open, message[:200]
        )
        return PolicyDecision(
            allowed=self._settings.fail_open,
            kind=kind,
            message=message,
            matched_policies=matched_policies,
            via_fail_mode=True,
        )


def _error_detail(response: httpx.Response) -> str:
    try:
        info: Final = _ErrorBody.model_validate_json(response.content).error
    except ValidationError:
        return ""
    if info is None:
        return ""
    parts: Final = tuple(part for part in (info.code, info.message) if part)
    return f" ({': '.join(parts)[:160]})" if parts else ""
