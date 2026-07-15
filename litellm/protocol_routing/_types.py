import os
from enum import Enum
from typing import List, Literal

from litellm.exceptions import BadRequestError

ProtocolRoutingMode = Literal["strict", "bridged"]

# Import-time read from env var, matching litellm peer pattern
# (e.g. LITELLM_DROP_PARAMS). Subsequent changes use set_protocol_routing_mode().
_env_value = os.environ.get("LITELLM_PROTOCOL_ROUTING_MODE", "").lower()
_protocol_routing_mode: ProtocolRoutingMode = (
    _env_value if _env_value in ("strict", "bridged") else "bridged"
)


def get_protocol_routing_mode() -> ProtocolRoutingMode:
    return _protocol_routing_mode


def set_protocol_routing_mode(mode: ProtocolRoutingMode) -> None:
    global _protocol_routing_mode
    if mode not in ("strict", "bridged"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'strict' or 'bridged'.")
    _protocol_routing_mode = mode


class SupportedProtocol(str, Enum):
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GOOGLE_GENERATE_CONTENT = "google_generate_content"
    LLM_PASSTHROUGH = "llm_passthrough"


ROUTE_TYPE_TO_PROTOCOL = {
    "acompletion": SupportedProtocol.OPENAI_CHAT,
    "atext_completion": SupportedProtocol.OPENAI_CHAT,
    "aresponses": SupportedProtocol.OPENAI_RESPONSES,
    "anthropic_messages": SupportedProtocol.ANTHROPIC_MESSAGES,
    "agenerate_content": SupportedProtocol.GOOGLE_GENERATE_CONTENT,
    "agenerate_content_stream": SupportedProtocol.GOOGLE_GENERATE_CONTENT,
    "allm_passthrough_route": SupportedProtocol.LLM_PASSTHROUGH,
}


class ProtocolMismatchError(BadRequestError):
    def __init__(
        self,
        model: str,
        requested_protocol: str,
        available_protocols: List[str],
    ):
        self.requested_protocol = requested_protocol
        self.available_protocols = available_protocols
        message = (
            f"No deployment for model '{model}' supports protocol "
            f"'{requested_protocol}'. Available protocols: {available_protocols}. "
            f"Enable bridged mode to allow automatic conversion: "
            f"set env LITELLM_PROTOCOL_ROUTING_MODE=bridged"
        )
        super().__init__(
            message=message,
            model=model,
            llm_provider=requested_protocol,
        )
