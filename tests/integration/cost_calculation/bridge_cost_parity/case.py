from dataclasses import dataclass
from collections.abc import Sequence
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from integration.cost_calculation.cost_tracking_case import CostTrackingTestCase, Deployment, ExactExpected
from pydantic import JsonValue

ROW_FIELDS: Final = frozenset({"spend", "prompt_tokens", "completion_tokens"})
_EXPECTED_FIELDS: Final = frozenset(ExactExpected.model_fields)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderCostQuirk:
    """A provider-level reason the bridged leg legitimately bills a different row: the `ExactExpected` fields that
    differ, why, and the provider document that says so."""

    fields: frozenset[str]
    reason: str
    source: str

    def __post_init__(self) -> None:
        unknown: Final = self.fields - _EXPECTED_FIELDS
        if not self.fields or unknown:
            raise ValueError(f"quirk fields must be non-empty ExactExpected fields, got {sorted(unknown)}")
        if not self.source.startswith("https://"):
            raise ValueError("a quirk cites the provider document that explains it")


@dataclass(frozen=True, slots=True, kw_only=True)
class BridgeCostParityTestCase:
    """One stored provider response billed through the provider's own API (`native`) and through a client API that
    LiteLLM bridges into it (`bridged`). Both legs bill the one row unless `quirk` names the fields that differ."""

    native: CostTrackingTestCase
    bridged: CostTrackingTestCase
    quirk: ProviderCostQuirk | None = None
    known_bug: str | None = None

    def __post_init__(self) -> None:
        if self.native.endpoint == self.bridged.endpoint:
            raise ValueError(f"{self.id}: native and bridged cases must use different client endpoints")
        if self.native.response != self.bridged.response:
            raise ValueError(f"{self.id}: native and bridged cases must share one stored provider response")
        if (self.native.model, self.native.deployment) != (self.bridged.model, self.bridged.deployment):
            raise ValueError(f"{self.id}: native and bridged cases must share model and deployment")
        if not isinstance(self.native.expected, ExactExpected) or not isinstance(self.bridged.expected, ExactExpected):
            raise ValueError(f"{self.id}: both legs need an ExactExpected row")
        native: Final = self.native.expected.model_dump()
        bridged: Final = self.bridged.expected.model_dump()
        differing: Final = frozenset(field for field in _EXPECTED_FIELDS if native[field] != bridged[field])
        if differing != self.quirk_fields:
            raise ValueError(
                f"{self.id}: expected rows differ in {sorted(differing)} but the quirk covers {sorted(self.quirk_fields)}"
            )

    @property
    def quirk_fields(self) -> frozenset[str]:
        return self.quirk.fields if self.quirk is not None else frozenset()

    @property
    def compared_row_fields(self) -> frozenset[str]:
        return ROW_FIELDS - self.quirk_fields

    @property
    def id(self) -> str:
        return self.bridged.name


BridgeEndpoint = Literal["/v1/chat/completions", "/v1/responses", "/v1/messages"]

_ENDPOINT_SUFFIX: Final[Mapping[str, str]] = MappingProxyType(
    {"/v1/chat/completions": "chat_bridged", "/v1/responses": "responses_bridged", "/v1/messages": "messages_bridged"}
)


def _prompt_text(request: Mapping[str, JsonValue]) -> str:
    """The user prompt of a plain base request, whichever client shape it was written in."""
    prompt: Final = request.get("input")
    if isinstance(prompt, str):
        return prompt
    messages: Final = request.get("messages")
    if not isinstance(messages, list):
        raise ValueError("a bridge base request needs a string input or a messages list")
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content: Final = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(part["text"]) for part in content if isinstance(part, dict) and "text" in part)
    raise ValueError("a bridge base request needs a user message")


def _bridged_request(base: CostTrackingTestCase, endpoint: BridgeEndpoint) -> dict[str, JsonValue]:
    streamed: Final = base.request.get("stream") is True
    text: Final = _prompt_text(base.request)
    if endpoint == "/v1/responses":
        return {"model": "$MODEL", "input": text, **({"stream": True} if streamed else {})}
    messages: Final[list[JsonValue]] = [{"role": "user", "content": text}]
    if endpoint == "/v1/messages":
        return {"model": "$MODEL", "max_tokens": 412, "messages": messages, **({"stream": True} if streamed else {})}
    return {
        "model": "$MODEL",
        "messages": messages,
        **({"stream": True, "stream_options": {"include_usage": True}} if streamed else {}),
    }


def bridge_pairs(
    bases: Sequence[CostTrackingTestCase],
    endpoints: Sequence[BridgeEndpoint],
    *,
    deployment: Deployment | None = None,
    known_bug: str | None = None,
) -> tuple[BridgeCostParityTestCase, ...]:
    """One parity pair per base and bridged endpoint: the same stored provider response and deployment, billed through
    the base's own endpoint and through each client endpoint LiteLLM bridges into it. `deployment` pins both legs to a
    deployment the base does not carry (an `openai/responses/` deployment for the chat -> Responses bridge)."""
    pinned: Final = tuple(
        base if deployment is None else base.model_copy(update={"deployment": deployment}) for base in bases
    )
    return tuple(
        BridgeCostParityTestCase(
            native=base,
            bridged=base.model_copy(
                update={
                    "name": f"{base.name}-{_ENDPOINT_SUFFIX[endpoint]}",
                    "endpoint": endpoint,
                    "request": _bridged_request(base, endpoint),
                }
            ),
            known_bug=known_bug,
        )
        for base in pinned
        for endpoint in endpoints
        if endpoint != base.endpoint
    )
