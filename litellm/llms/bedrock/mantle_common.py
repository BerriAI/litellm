"""
Shared endpoint resolution for the Bedrock Mantle (Claude Mythos Preview) routes.

The chat (`bedrock/mantle/...`) and Anthropic Messages (`/v1/messages`) configs
both target the bedrock-mantle endpoint over the Anthropic Messages path. They
must honor a caller-supplied `api_base` or `aws_bedrock_runtime_endpoint` so a
deployment can route through a private (VPC) endpoint instead of the public host
"""

from typing import Mapping, Optional

MANTLE_MESSAGES_PATH = "/anthropic/v1/messages"


def get_runtime_endpoint_override(
    *param_sources: Mapping[str, object]
) -> Optional[str]:
    for source in param_sources:
        value = source.get("aws_bedrock_runtime_endpoint")
        if isinstance(value, str):
            return value
    return None


def resolve_mantle_messages_url(
    region: str,
    api_base: Optional[str],
    aws_bedrock_runtime_endpoint: Optional[str],
) -> str:
    base = (
        api_base
        or aws_bedrock_runtime_endpoint
        or f"https://bedrock-mantle.{region}.api.aws"
    ).rstrip("/")
    if base.endswith(MANTLE_MESSAGES_PATH):
        return base
    return f"{base}{MANTLE_MESSAGES_PATH}"
