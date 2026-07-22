"""
Calls an Amazon Bedrock AgentCore Gateway web-search target (MCP protocol) to search the web.

Web Search on Amazon Bedrock AgentCore exposes Amazon's managed web index through
an AgentCore Gateway MCP endpoint.

AWS docs: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-connector-web-search-tool.html

Authentication (matches the gateway's inbound authorizer type):
- AWS_IAM gateway: the request is SigV4-signed. Credentials come from explicit
  params (aws_access_key_id / aws_secret_access_key / aws_session_token /
  aws_region_name — also settable in a proxy search_tools entry) or the
  standard AWS credential chain (env / profile / IRSA / assumed role)
- CUSTOM_JWT gateway: pass the OAuth2 bearer token (e.g. Cognito
  client_credentials) as api_key, or set AGENTCORE_GATEWAY_TOKEN

Setup:
    1. Create an AgentCore Gateway with a web-search connector target
    2. Set AGENTCORE_GATEWAY_URL (or pass api_base) to the gateway MCP endpoint, e.g.
       https://<gateway-id>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
    3. AWS_IAM: ensure the credentials allow bedrock-agentcore:InvokeGateway
       CUSTOM_JWT: set AGENTCORE_GATEWAY_TOKEN (or pass api_key)

Usage:
    response = litellm.search(
        query="latest AI developments",
        search_provider="agentcore",
        max_results=5,
        aws_access_key_id="...",       # optional — omit to use the default chain
        aws_secret_access_key="...",
    )
"""

import json
import re
from typing import Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.secret_managers.main import get_secret_str

# AgentCore web-search rejects queries longer than 200 characters
AGENTCORE_MAX_QUERY_LENGTH = 200

# The provider contract documents a default of 10 results — send it explicitly
# so the gateway can't silently apply a different default.
AGENTCORE_DEFAULT_MAX_RESULTS = 10

# Default MCP tool name for a gateway web-search connector target:
# "<target-name>___<tool-name>". Override with AGENTCORE_SEARCH_TOOL_NAME
# or optional_params["tool_name"] when the target uses a custom name.
AGENTCORE_DEFAULT_TOOL_NAME = "web-search-tool___WebSearch"

# All web-search connector tools share this suffix; rejecting other names keeps
# a caller-supplied tool_name from invoking unrelated tools on the same gateway
# with the proxy's credentials.
AGENTCORE_TOOL_NAME_SUFFIX = "___WebSearch"


class AgentCoreSearchConfig(BaseSearchConfig, BaseAWSLLM):
    def __init__(self) -> None:
        BaseSearchConfig.__init__(self)
        BaseAWSLLM.__init__(self)

    @staticmethod
    def ui_friendly_name() -> str:
        return "Web Search on Amazon Bedrock"

    def validate_environment(
        self,
        headers: dict,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ) -> dict:
        """
        Set MCP transport headers. Per the MCP Streamable HTTP transport spec,
        the client MUST accept both application/json and text/event-stream.

        Authentication itself happens in sign_request(): bearer token for
        CUSTOM_JWT gateways, AWS SigV4 for AWS_IAM gateways.
        """
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json, text/event-stream"
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict,
        data: Union[dict, list[dict]] | None = None,
        **kwargs,
    ) -> str:
        api_base = api_base or get_secret_str("AGENTCORE_GATEWAY_URL")
        if not api_base:
            raise ValueError(
                "AGENTCORE_GATEWAY_URL is not set. Set it to your AgentCore Gateway MCP "
                "endpoint (https://<gateway-id>.gateway.bedrock-agentcore.<region>"
                ".amazonaws.com/mcp) or pass api_base."
            )
        return api_base

    def transform_search_request(
        self,
        query: Union[str, list[str]],
        optional_params: dict,
        **kwargs,
    ) -> dict:
        """
        Transform Search request to an MCP tools/call request.

        Args:
            query: Search query (string or list of strings). AgentCore only
                supports single string queries; lists are joined with spaces.
            optional_params: Optional parameters for the request
                - max_results: Maximum number of results (1-25), default 10
                - tool_name: Override the MCP tool name of the gateway target

        Returns:
            Dict with the JSON-RPC 2.0 request body
        """
        if isinstance(query, list):
            query = " ".join(query)
        query = query[:AGENTCORE_MAX_QUERY_LENGTH]

        tool_name = (
            optional_params.get("tool_name")
            or get_secret_str("AGENTCORE_SEARCH_TOOL_NAME")
            or AGENTCORE_DEFAULT_TOOL_NAME
        )
        if not tool_name.endswith(AGENTCORE_TOOL_NAME_SUFFIX):
            raise ValueError(
                f"Invalid AgentCore search tool_name '{tool_name}': must end with "
                f"'{AGENTCORE_TOOL_NAME_SUFFIX}' (a web-search connector tool). "
                "Other gateway tools cannot be invoked through this provider."
            )

        arguments: dict[str, Union[str, int]] = {"query": query}
        arguments["maxResults"] = optional_params.get("max_results", AGENTCORE_DEFAULT_MAX_RESULTS)

        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: Union[dict, list[dict]],
        api_base: str,
        api_key: str | None = None,
    ) -> tuple[dict, bytes | None]:
        """
        Authenticate the MCP request.

        CUSTOM_JWT gateways: attach the caller's OAuth2 bearer token (api_key
        or AGENTCORE_GATEWAY_TOKEN) — no AWS credentials involved.

        AWS_IAM gateways: SigV4-sign with the bedrock-agentcore service name.
        """
        if not isinstance(request_data, dict):
            raise ValueError("AgentCore search expects a single dict request body")

        # Server-managed token fallback is gated on the request targeting the
        # operator-configured gateway host — otherwise an authenticated caller
        # could point api_base at their own server (e.g. via
        # /search_tools/test_connection) and receive AGENTCORE_GATEWAY_TOKEN.
        bearer_token = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("AGENTCORE_GATEWAY_TOKEN",),
            base_env_var="AGENTCORE_GATEWAY_URL",
            default_api_base=None,
        )
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
            return headers, json.dumps(request_data).encode()

        # The signing region must match the gateway's region — derive it from
        # standard gateway hostnames so callers don't have to set
        # aws_region_name to a region different from their default. Custom or
        # private hostnames can't be parsed: fall back to an explicitly
        # configured region (param or AWS env vars), and error out rather than
        # silently signing for a guessed region the gateway would reject with
        # a confusing auth error.
        signing_params = dict(optional_params)
        if signing_params.get("aws_region_name") is None:
            match = re.search(
                r"\.gateway\.bedrock-agentcore\.([a-z0-9-]+)\.amazonaws\.com",
                api_base,
            )
            if match:
                signing_params["aws_region_name"] = match.group(1)
            elif not any(get_secret_str(var) for var in ("AWS_REGION", "AWS_REGION_NAME", "AWS_DEFAULT_REGION")):
                raise ValueError(
                    f"Cannot derive the SigV4 signing region from api_base '{api_base}'. "
                    "Set aws_region_name (or the AWS_REGION env var) to the gateway's "
                    "region when using a custom hostname."
                )

        # api_key="" (not None, but falsy) disables BaseAWSLLM's fallback to the
        # AWS_BEARER_TOKEN_BEDROCK env var: that token is a Bedrock Runtime
        # credential and must not be sent to an AgentCore gateway.
        return self._sign_request(
            service_name="bedrock-agentcore",
            headers=headers,
            optional_params=signing_params,
            request_data=request_data,
            api_base=api_base,
            api_key="",
        )

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform an MCP tools/call response to LiteLLM unified SearchResponse.

        The gateway returns JSON-RPC (as plain JSON or a single-message SSE
        stream) whose result.content[] text blocks contain a JSON list of
        {title, url, date/publishedDate, text} entries.
        """
        response_json = self._parse_mcp_body(raw_response)

        if "error" in response_json:
            raise BedrockError(
                status_code=raw_response.status_code if raw_response.status_code >= 400 else 502,
                message=f"AgentCore gateway MCP error: {response_json['error']}",
            )

        results: list[SearchResult] = []
        for block in response_json.get("result", {}).get("content", []):
            if block.get("type") != "text":
                continue
            try:
                parsed = json.loads(block["text"])
            except (json.JSONDecodeError, TypeError):
                continue
            items = parsed.get("results", []) if isinstance(parsed, dict) else parsed
            for item in items:
                if not isinstance(item, dict):
                    continue
                results.append(
                    SearchResult(
                        title=item.get("title") or "",
                        url=item.get("url") or "",
                        snippet=item.get("text") or item.get("snippet") or "",
                        date=item.get("publishedDate") or item.get("date"),
                        last_updated=None,
                    )
                )

        return SearchResponse(results=results, object="search")

    @staticmethod
    def _parse_mcp_body(raw_response: httpx.Response) -> dict:
        """
        Parse a JSON or SSE-framed (Streamable HTTP transport) MCP response.

        Per the SSE spec, an event's data is the concatenation of all its
        ``data:`` lines (joined with newlines), and a stream may carry several
        events (e.g. progress notifications before the JSON-RPC response).
        Return the event whose payload carries the ``id``-matched JSON-RPC
        response — i.e. one containing ``result`` or ``error``.
        """
        text = raw_response.text
        if not text.lstrip().startswith(("event:", "data:", ":", "id:", "retry:")):
            return raw_response.json()

        last_parsed: dict | None = None
        data_lines: list[str] = []
        # Trailing sentinel flushes the final event even without a blank line
        for line in text.splitlines() + [""]:
            if line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
                continue
            if line == "" and data_lines:
                try:
                    parsed = json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    parsed = None
                data_lines = []
                if isinstance(parsed, dict):
                    last_parsed = parsed
                    if "result" in parsed or "error" in parsed:
                        return parsed
        if last_parsed is not None:
            return last_parsed
        raise BedrockError(
            status_code=502,
            message=f"AgentCore gateway returned SSE without a JSON data frame: {text[:200]}",
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict,
    ) -> Exception:
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
