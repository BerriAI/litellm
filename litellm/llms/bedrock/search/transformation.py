"""
Calls an Amazon Bedrock AgentCore Gateway web-search target (MCP protocol) to search the web.

Web Search on Amazon Bedrock AgentCore exposes Amazon's managed web index through
an AgentCore Gateway MCP endpoint.

AWS docs: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-connector-web-search-tool.html

Authentication (matches the gateway's inbound authorizer type):
- AWS_IAM gateway: the request is SigV4-signed. Credentials come from explicit
  params (aws_access_key_id / aws_secret_access_key / aws_session_token /
  aws_region_name, also settable in a proxy search_tools entry) or the
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
        aws_access_key_id="...",       # optional, omit to use the default chain
        aws_secret_access_key="...",
    )
"""

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

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
AGENTCORE_MAX_QUERY_LENGTH: Final = 200

# The provider contract documents a default of 10 results, send it explicitly
# so the gateway can't silently apply a different default.
AGENTCORE_DEFAULT_MAX_RESULTS: Final = 10

# Default MCP tool name for a gateway web-search connector target:
# "<target-name>___<tool-name>". Override with AGENTCORE_SEARCH_TOOL_NAME
# or optional_params["tool_name"] when the target uses a custom name.
AGENTCORE_DEFAULT_TOOL_NAME: Final = "web-search-tool___WebSearch"

# All web-search connector tools share this suffix; rejecting other names keeps
# a caller-supplied tool_name from invoking unrelated tools on the same gateway
# with the proxy's credentials.
AGENTCORE_TOOL_NAME_SUFFIX: Final = "___WebSearch"

# MCP revision this provider speaks. Sent on every request because the gateway is
# called statelessly, without an initialize handshake to negotiate a version.
# AgentCore gateways whose protocolConfiguration leaves supportedVersions unset
# accept only 2025-03-26 and reject anything newer with a -32600 error, so that
# is the default; a gateway pinned to another version needs
# AGENTCORE_MCP_PROTOCOL_VERSION set to match.
AGENTCORE_DEFAULT_MCP_PROTOCOL_VERSION: Final = "2025-03-26"

# Matched against the URL host so a crafted path or query string can't pass for
# a gateway hostname.
_GATEWAY_HOST_PATTERN: Final = re.compile(r"[a-z0-9-]+\.gateway\.bedrock-agentcore\.([a-z0-9-]+)\.amazonaws\.com")

_SSE_EVENT_SEPARATOR: Final = re.compile(r"\r?\n[ \t]*\r?\n")

_SSE_LINE_PREFIXES: Final = ("event:", "data:", ":", "id:", "retry:")


def _gateway_host_match(api_base: str) -> re.Match[str] | None:
    return _GATEWAY_HOST_PATTERN.fullmatch(httpx.URL(api_base).host)


_LOOPBACK_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})


def _credential_safe_transport(api_base: str) -> bool:
    url: Final = httpx.URL(api_base)
    return url.scheme == "https" or url.host in _LOOPBACK_HOSTS


def _string_field(item: Mapping[str, object], *keys: str) -> str | None:
    return next(
        (value for key in keys if isinstance(value := item.get(key), str) and value),
        None,
    )


def _to_search_result(item: Mapping[str, object]) -> SearchResult:
    return SearchResult(
        title=_string_field(item, "title") or "",
        url=_string_field(item, "url") or "",
        snippet=_string_field(item, "text", "snippet") or "",
        date=_string_field(item, "publishedDate", "date"),
        last_updated=None,
    )


def _result_items(parsed: object) -> tuple[Mapping[str, object], ...]:
    items: Final = parsed.get("results", ()) if isinstance(parsed, Mapping) else parsed
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return ()
    return tuple(item for item in items if isinstance(item, Mapping))


def _parse_result_items(raw_text: object) -> tuple[Mapping[str, object], ...]:
    """
    Parse one MCP text block into the search result objects it carries.

    A block holds either a JSON list of results or a {"results": [...]} object;
    anything unparseable is skipped rather than failing the whole response.
    """
    if not isinstance(raw_text, str):
        return ()
    try:
        parsed: Final = json.loads(raw_text)
    except json.JSONDecodeError:
        return ()
    return _result_items(parsed)


def _iter_sse_events(text: str) -> Iterator[Mapping[str, object]]:
    """
    Yield the JSON payload of each SSE event in a Streamable HTTP MCP response.

    Per the SSE spec an event's data is the concatenation of all its ``data:``
    lines (joined with newlines), and a stream may carry several events, e.g.
    progress notifications before the JSON-RPC response.
    """
    for chunk in _SSE_EVENT_SEPARATOR.split(text):
        payload = "\n".join(line[len("data:") :].lstrip() for line in chunk.splitlines() if line.startswith("data:"))
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            yield parsed


class AgentCoreSearchConfig(BaseSearchConfig, BaseAWSLLM):
    def __init__(self) -> None:
        BaseSearchConfig.__init__(self)
        BaseAWSLLM.__init__(self)

    @staticmethod
    def ui_friendly_name() -> str:
        return "Web Search on Amazon Bedrock"

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: BaseSearchConfig hands providers the mutable request header dict
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.validate_environment forwards provider-specific extras
    ) -> dict:  # mutable-ok: the handler passes these headers straight to httpx, which wants a dict
        """
        Set MCP transport headers. Per the MCP Streamable HTTP transport spec,
        the client MUST accept both application/json and text/event-stream, and
        declare its protocol revision with MCP-Protocol-Version.

        Authentication itself happens in sign_request(): bearer token for
        CUSTOM_JWT gateways, AWS SigV4 for AWS_IAM gateways.
        """
        return {  # mutable-ok: httpx request headers are a dict
            **headers,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": get_secret_str("AGENTCORE_MCP_PROTOCOL_VERSION")
            or AGENTCORE_DEFAULT_MCP_PROTOCOL_VERSION,
        }

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict,  # mutable-ok: BaseSearchConfig passes optional params as a dict
        data: dict | list[dict] | None = None,  # mutable-ok: BaseSearchConfig request bodies are JSON dicts
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.get_complete_url forwards provider-specific extras
    ) -> str:
        gateway_url: Final = api_base or get_secret_str("AGENTCORE_GATEWAY_URL")
        if not gateway_url:
            raise ValueError(
                "AGENTCORE_GATEWAY_URL is not set. Set it to your AgentCore Gateway MCP "
                "endpoint (https://<gateway-id>.gateway.bedrock-agentcore.<region>"
                ".amazonaws.com/mcp) or pass api_base."
            )
        return gateway_url

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig accepts a list of queries
        optional_params: dict,  # mutable-ok: BaseSearchConfig passes optional params as a dict
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_request forwards provider-specific extras
    ) -> dict:  # mutable-ok: the JSON-RPC body is serialized as a JSON object
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
        joined_query: Final = " ".join(query) if isinstance(query, list) else query
        tool_name: Final = (
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

        return {  # mutable-ok: JSON-RPC request bodies are JSON objects
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {  # mutable-ok: JSON-RPC request bodies are JSON objects
                "name": tool_name,
                "arguments": {  # mutable-ok: JSON-RPC request bodies are JSON objects
                    "query": joined_query[:AGENTCORE_MAX_QUERY_LENGTH],
                    "maxResults": optional_params.get("max_results", AGENTCORE_DEFAULT_MAX_RESULTS),
                },
            },
        }

    def sign_request(
        self,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig hands providers the mutable request header dict
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig passes optional params as a dict
        request_data: dict[str, object] | list[dict[str, object]],  # mutable-ok: request bodies are JSON dicts
        api_base: str,
        api_key: str | None = None,
    ) -> tuple[dict[str, str], bytes | None]:  # mutable-ok: BaseSearchConfig.sign_request returns httpx headers
        """
        Authenticate the MCP request.

        CUSTOM_JWT gateways: attach the caller's OAuth2 bearer token (api_key
        or AGENTCORE_GATEWAY_TOKEN), no AWS credentials involved.

        AWS_IAM gateways: SigV4-sign with the bedrock-agentcore service name.
        """
        if not isinstance(request_data, dict):
            raise TypeError("AgentCore search expects a single dict request body")

        if not _credential_safe_transport(api_base):
            raise ValueError(
                f"Refusing to send AgentCore credentials over plaintext HTTP to '{api_base}': a bearer "
                "token or SigV4 signature would be readable in transit. Use an https gateway URL "
                "(plain http is allowed only for localhost)."
            )

        # Server-managed credentials only go to a trusted host, otherwise an
        # authenticated caller could point api_base at their own server (e.g. via
        # /search_tools/test_connection) and collect AGENTCORE_GATEWAY_TOKEN or a
        # SigV4 signature with the proxy's credential scope and session token.
        gateway_host_match: Final = _gateway_host_match(api_base)
        bearer_token: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("AGENTCORE_GATEWAY_TOKEN",),
            base_env_var="AGENTCORE_GATEWAY_URL",
            default_api_base=api_base if gateway_host_match else None,
        )
        if bearer_token:
            bearer_headers: Final = {  # mutable-ok: httpx request headers are a dict
                **headers,
                "Authorization": f"Bearer {bearer_token}",
            }
            return bearer_headers, json.dumps(request_data).encode()

        if gateway_host_match is None and not self._is_configured_gateway(api_base):
            raise ValueError(
                f"Refusing to send SigV4-signed AgentCore requests to '{api_base}': it is neither an "
                "AgentCore gateway hostname nor the host in AGENTCORE_GATEWAY_URL. Set "
                "AGENTCORE_GATEWAY_URL to authorize a custom gateway hostname."
            )

        signing_params: Final = (
            optional_params
            if optional_params.get("aws_region_name") is not None
            else {  # mutable-ok: BaseAWSLLM._sign_request takes optional params as a dict
                **optional_params,
                "aws_region_name": self._signing_region(api_base),
            }
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

    @staticmethod
    def _is_configured_gateway(api_base: str) -> bool:
        configured: Final = get_secret_str("AGENTCORE_GATEWAY_URL")
        if not configured:
            return False
        return httpx.URL(configured).host == httpx.URL(api_base).host

    @staticmethod
    def _signing_region(api_base: str) -> str:
        """
        Resolve the SigV4 signing region, which must match the gateway's region.

        Standard gateway hostnames carry it, so callers don't have to set
        aws_region_name to a region different from their default. For custom or
        private hostnames, defer to the AWS configuration chain (env vars and
        the shared config / profile region), and error out when that yields
        nothing rather than silently signing for a guessed region the gateway
        would reject with a confusing auth error.
        """
        match: Final = _gateway_host_match(api_base)
        if match:
            return match.group(1)

        # boto3's session resolution covers env vars AND the AWS shared config
        # (profile region), unlike BaseAWSLLM's helper, which silently defaults
        # to us-west-2 when nothing is configured.
        import boto3

        configured_region: Final = boto3.Session().region_name
        if configured_region:
            return configured_region
        raise ValueError(
            f"Cannot derive the SigV4 signing region from api_base '{api_base}' "
            "or the AWS configuration chain. Set aws_region_name (or AWS_DEFAULT_REGION / "
            "a profile region) to the gateway's region when using a custom hostname."
        )

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_response forwards provider-specific extras
    ) -> SearchResponse:
        """
        Transform an MCP tools/call response to LiteLLM unified SearchResponse.

        The gateway returns JSON-RPC (as plain JSON or a single-message SSE
        stream) whose result.content[] text blocks contain a JSON list of
        {title, url, date/publishedDate, text} entries. Web-search connector
        1.1.0 and later repeat that list in result.structuredContent, which is
        the only machine-readable copy when the text block holds prose instead.
        """
        response_json: Final = self._parse_mcp_body(raw_response)

        error: Final = response_json.get("error")
        if error is not None:
            raise BedrockError(
                status_code=raw_response.status_code if raw_response.status_code >= 400 else 502,
                message=f"AgentCore gateway MCP error: {error}",
            )

        # A failed tools/call is reported in-band, as HTTP 200 with result.isError
        # and the failure text where the results would be.
        result: Final = response_json.get("result")
        if isinstance(result, dict) and result.get("isError"):
            raise BedrockError(
                status_code=raw_response.status_code if raw_response.status_code >= 400 else 502,
                message=f"AgentCore web search tool error: {self._tool_error_message(response_json)}",
            )

        text_items: Final = tuple(
            item for block in self._text_blocks(response_json) for item in _parse_result_items(block.get("text"))
        )
        structured: Final = result.get("structuredContent") if isinstance(result, Mapping) else None
        items: Final = text_items or _result_items(structured)

        results: Final = [_to_search_result(item) for item in items]  # mutable-ok: pydantic list field

        return SearchResponse(results=results, object="search")

    def _tool_error_message(self, response_json: Mapping[str, object]) -> str:
        texts: Final = tuple(
            text for block in self._text_blocks(response_json) if isinstance(text := block.get("text"), str)
        )
        return " ".join(texts) if texts else json.dumps(response_json.get("result"))[:500]

    @staticmethod
    def _text_blocks(response_json: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
        result: Final = response_json.get("result")
        content: Final = result.get("content") if isinstance(result, dict) else None
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            return ()
        return tuple(block for block in content if isinstance(block, dict) and block.get("type") == "text")

    @staticmethod
    def _parse_mcp_body(raw_response: httpx.Response) -> Mapping[str, object]:
        """
        Parse a JSON or SSE-framed (Streamable HTTP transport) MCP response.

        Return the event whose payload carries the JSON-RPC response, i.e. one
        containing ``result`` or ``error``, falling back to the last event when
        the stream carries only notifications.
        """
        text: Final = raw_response.text
        if not text.lstrip().startswith(_SSE_LINE_PREFIXES):
            return raw_response.json()

        events: Final = tuple(_iter_sse_events(text))
        response_event: Final = next(
            (event for event in events if "result" in event or "error" in event),
            None,
        )
        if response_event is not None:
            return response_event
        if events:
            return events[-1]
        raise BedrockError(
            status_code=502,
            message=f"AgentCore gateway returned SSE without a JSON data frame: {text[:200]}",
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict,  # mutable-ok: BaseSearchConfig.get_error_class takes the response headers as a dict
    ) -> Exception:
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
