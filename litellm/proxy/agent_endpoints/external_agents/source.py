import base64
import binascii
import hashlib
import hmac
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Final, Literal

from pydantic import Field, ValidationError

from litellm.proxy.agent_endpoints.external_agents.models import FrozenModel, HookFailure, ObservedAgentEvent, Verdict


class ToolInfo(FrozenModel):
    toolset_name: str | None = None


class Block(FrozenModel):
    type: str
    name: str | None = None
    tool_name: str | None = None
    tool_info: ToolInfo | None = None


class Message(FrozenModel):
    content: tuple[Block, ...] = ()


class Actor(FrozenModel):
    type: Literal["user"]
    id: str | None = None


class Application(FrozenModel):
    application: str


class GrokFrame(FrozenModel):
    type: Literal["prompt", "tool_call"]
    request_id: str
    tenant_id: str = Field(min_length=1)
    actor: Actor
    source: Application
    session_id: str = Field(min_length=1)
    model: str | None = None
    messages: tuple[Message, ...] = ()


def signing_key(secret: str) -> bytes | HookFailure:
    try:
        if not secret.startswith("whsec_"):
            return HookFailure(status=400, reason="Signing secret must start with whsec_")
        key: Final = base64.b64decode(secret[6:], validate=True)
        return key if key else HookFailure(status=400, reason="Empty signing secret")
    except (ValueError, binascii.Error):
        return HookFailure(status=400, reason="Invalid signing secret")


class ExternalAgentHookSource(ABC):
    @abstractmethod
    def verify(
        self, body: bytes, headers: Mapping[str, str], secrets: tuple[str, ...], now: float
    ) -> HookFailure | None:
        pass

    @abstractmethod
    def parse(self, body: bytes, request_id: str) -> ObservedAgentEvent | HookFailure:
        pass

    @abstractmethod
    def format_verdict(self, verdict: Verdict) -> Verdict:
        pass


def _block_names(block: Block) -> tuple[str, ...]:
    if block.type != "tool_use":
        return ()
    candidates: Final = (block.tool_name or block.name, block.tool_info.toolset_name if block.tool_info else None)
    return tuple(name for name in candidates if name)


def _message_names(message: Message) -> tuple[str, ...]:
    return sum((_block_names(block) for block in message.content), ())


class GrokBotHookSource(ExternalAgentHookSource):
    def verify(
        self, body: bytes, headers: Mapping[str, str], secrets: tuple[str, ...], now: float
    ) -> HookFailure | None:
        try:
            timestamp: Final = headers.get("webhook-timestamp", "")
            if abs(now - int(timestamp)) > 300:
                return HookFailure(status=401, reason="Stale webhook timestamp")
            request_id: Final = headers.get("webhook-id", "")
            if not request_id:
                return HookFailure(status=401, reason="Missing webhook id")
            signed: Final = f"{request_id}.{timestamp}.".encode() + body
            signatures: Final = tuple(
                part[3:] for part in headers.get("webhook-signature", "").split() if part.startswith("v1,")
            )
            for secret in secrets:
                key = signing_key(secret)
                if isinstance(key, HookFailure):
                    continue
                expected = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode()
                if any(hmac.compare_digest(expected.encode(), signature.encode()) for signature in signatures):
                    return None
            return HookFailure(status=401, reason="Invalid webhook signature")
        except (ValueError, OverflowError):
            return HookFailure(status=401, reason="Invalid webhook timestamp")

    def parse(self, body: bytes, request_id: str) -> ObservedAgentEvent | HookFailure:
        try:
            frame: Final = GrokFrame.model_validate_json(body)
        except ValidationError:
            return HookFailure(status=400, reason="Invalid hook frame")
        if frame.request_id != request_id:
            return HookFailure(status=400, reason="Webhook id does not match request id")
        names: Final = sum((_message_names(message) for message in frame.messages), ())
        return ObservedAgentEvent(
            request_id=frame.request_id,
            tenant_id=frame.tenant_id,
            external_agent_id=frame.session_id,
            actor_id=frame.actor.id,
            frame_type=frame.type,
            application=frame.source.application,
            model=frame.model,
            tool_names=names if frame.type == "tool_call" else (),
        )

    def format_verdict(self, verdict: Verdict) -> Verdict:
        return verdict
