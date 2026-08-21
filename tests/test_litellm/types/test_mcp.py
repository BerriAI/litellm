"""The advertised MCP spec revisions must stay in lockstep with the pinned MCP SDK.

``MCP_LATEST_SUPPORTED_SPEC_VERSION`` is what LiteLLM puts on the wire for the MCP requests it
builds itself, and the SDK owns negotiation for everything else, so a revision the SDK gained
without ``MCPSpecVersion`` gaining it means LiteLLM is advertising a version it no longer leads
with.
"""

from typing import Final, get_args

from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS
from mcp.types import LATEST_PROTOCOL_VERSION

from litellm.types.mcp import (
    MCP_LATEST_SUPPORTED_SPEC_VERSION,
    MCPSpecVersion,
    MCPSpecVersionType,
)


def test_spec_version_enum_covers_every_sdk_supported_revision():
    known: Final = {member.value for member in MCPSpecVersion}
    assert set(SUPPORTED_PROTOCOL_VERSIONS) <= known, (
        "the pinned MCP SDK negotiates a spec revision MCPSpecVersion does not know about; "
        "add it to the enum and MCPSpecVersionType"
    )


def test_latest_supported_spec_version_is_the_sdk_latest():
    assert MCP_LATEST_SUPPORTED_SPEC_VERSION.value == LATEST_PROTOCOL_VERSION


def test_spec_version_literal_mirrors_the_enum():
    assert set(get_args(MCPSpecVersionType)) == set(MCPSpecVersion)
