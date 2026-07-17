from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES

MCP_SERVERS = [{"type": "url", "url": "https://mcp.deepwiki.com/mcp", "name": "deepwiki"}]


def test_messages_injects_the_mcp_beta_when_mcp_servers_is_set():
    """
    Regression test: /v1/messages must send the MCP connector beta header.

    Given: A request carrying Anthropic's native mcp_servers
    When:  Beta headers are computed for the request
    Then:  mcp-client-2025-04-04 is present

    Anthropic refuses mcp_servers without this beta ("mcp_servers: Extra inputs are
    not permitted"), so the whole connector was unusable on this route while working
    on /chat/completions, which gets the same header from AnthropicConfig.
    """
    headers = AnthropicMessagesConfig._update_headers_with_anthropic_beta(
        headers={},
        optional_params={"mcp_servers": MCP_SERVERS},
    )

    assert ANTHROPIC_BETA_HEADER_VALUES.MCP_CLIENT_2025_04_04.value in headers["anthropic-beta"]


def test_messages_does_not_inject_the_mcp_beta_without_mcp_servers():
    """The beta is opt-in; a request with no MCP connector must not advertise it."""
    headers = AnthropicMessagesConfig._update_headers_with_anthropic_beta(headers={}, optional_params={})

    assert ANTHROPIC_BETA_HEADER_VALUES.MCP_CLIENT_2025_04_04.value not in headers.get("anthropic-beta", "")


def test_messages_mcp_beta_preserves_caller_and_sibling_betas():
    """
    The header is a comma-joined set, so injecting must not drop what is already there.

    A caller-supplied beta and a second feature's beta both have to survive alongside
    the MCP one, otherwise enabling MCP silently disables another feature.
    """
    headers = AnthropicMessagesConfig._update_headers_with_anthropic_beta(
        headers={"anthropic-beta": "some-caller-beta"},
        optional_params={"mcp_servers": MCP_SERVERS, "speed": "fast"},
    )

    betas = {value.strip() for value in headers["anthropic-beta"].split(",")}
    assert ANTHROPIC_BETA_HEADER_VALUES.MCP_CLIENT_2025_04_04.value in betas
    assert ANTHROPIC_BETA_HEADER_VALUES.FAST_MODE_2026_02_01.value in betas
    assert "some-caller-beta" in betas
