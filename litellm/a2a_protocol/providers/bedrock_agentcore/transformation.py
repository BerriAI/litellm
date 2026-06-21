"""
Transformation layer for Bedrock AgentCore A2A provider.

Constructs JSON-RPC envelopes, derives AgentCore URLs from model ARNs,
and signs requests via AmazonAgentCoreConfig (SigV4 or JWT).
"""

import json
from typing import Any, AsyncIterator, Dict, Mapping, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

# Reserved outbound header names that must never be sourced from per-request
# ``agent_extra_headers`` for AgentCore requests. ``agent_extra_headers`` carries
# values rewritten from the client-controlled ``x-a2a-{agent}-*`` convention, so
# allowing these would let any caller with access to the agent spoof the AWS
# request identity / SigV4 metadata by overwriting headers the proxy sets from
# trusted server-side config.
#
# The runtime headers (session / user id) are derived server-side from
# ``runtimeSessionId`` / ``runtimeUserId`` in the agent's ``litellm_params``;
# ``authorization`` is set by the AgentCore signer (JWT or SigV4); ``host`` and
# the ``x-amz-*`` family are owned by SigV4 itself.
_RESERVED_EXACT_HEADERS = frozenset(
    {
        "authorization",
        "host",
    }
)
_RESERVED_PREFIX_HEADERS: Tuple[str, ...] = (
    "x-amzn-bedrock-agentcore-runtime-",
    "x-amz-",
)


def _filter_reserved_headers(
    agent_extra_headers: Optional[Mapping[str, str]],
) -> Optional[Dict[str, str]]:
    """
    Strip reserved AWS / AgentCore headers from caller-supplied
    ``agent_extra_headers`` before they are merged into the signed request.

    Returns ``None`` if the result is empty.
    """
    if not agent_extra_headers:
        return None

    filtered: Dict[str, str] = {}
    dropped: list = []
    for k, v in agent_extra_headers.items():
        k_lower = k.lower()
        if k_lower in _RESERVED_EXACT_HEADERS or any(
            k_lower.startswith(prefix) for prefix in _RESERVED_PREFIX_HEADERS
        ):
            dropped.append(k)
            continue
        filtered[k] = v

    if dropped:
        verbose_logger.warning(
            "BedrockAgentCore A2A: dropping reserved header(s) from "
            "agent_extra_headers (not forwarded to AgentCore): %s",
            sorted(dropped),
        )

    return filtered or None


class BedrockAgentCoreA2ATransformation:
    """
    Request/response transformation for Bedrock AgentCore A2A agents.

    Reuses AmazonAgentCoreConfig for URL construction, ARN parsing,
    and request signing. No logic is duplicated.
    """

    @staticmethod
    def get_url_and_signed_request(
        request_id: str,
        params: Dict[str, Any],
        litellm_params: Dict[str, Any],
        method: str = "message/send",
        stream: bool = False,
        agent_extra_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, dict, bytes]:
        """
        Build the AgentCore URL, construct a JSON-RPC envelope, and sign the request.

        Args:
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams
            litellm_params: Agent's litellm_params (model, api_key, etc.)
            method: JSON-RPC method name (default: "message/send")
            stream: Whether this is a streaming request
            agent_extra_headers: Per-request headers (from x-a2a-{agent}-* rewrite and
                admin extra_headers) to forward on the upstream HTTP call. Merged into
                the headers dict before signing so SigV4 includes them in the signature.
                Reserved AWS / AgentCore identity headers (``authorization``, ``host``,
                ``x-amzn-bedrock-agentcore-runtime-*``, ``x-amz-*``) are filtered out
                here to prevent a caller-controlled ``x-a2a-{agent}-*`` header from
                spoofing the AgentCore runtime user id or other SigV4 metadata. Use
                ``api_key`` / ``runtimeUserId`` / ``runtimeSessionId`` in litellm_params
                (not ``agent_extra_headers``) to override those values.

        Returns:
            Tuple of (url, signed_headers, signed_body_bytes)
        """
        # Extract model and strip the "bedrock/" prefix
        # "bedrock/agentcore/arn:aws:..." → "agentcore/arn:aws:..."
        model = litellm_params.get("model", "")
        if model.startswith("bedrock/"):
            agentcore_model = model[len("bedrock/") :]
        else:
            agentcore_model = model

        # Build optional_params from litellm_params (everything except model and custom_llm_provider)
        optional_params = {
            k: v
            for k, v in litellm_params.items()
            if k not in ("model", "custom_llm_provider")
        }

        agentcore_config = AmazonAgentCoreConfig()

        # Derive URL from ARN
        url = agentcore_config.get_complete_url(
            api_base=optional_params.get("api_base"),
            api_key=optional_params.get("api_key"),
            model=agentcore_model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )

        # Construct JSON-RPC 2.0 envelope
        json_rpc_body = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id,
            "params": params,
        }

        # Set required AgentCore session headers (normally set by transform_request,
        # which we skip because it also builds {"prompt": "..."})
        headers: dict = {}
        session_id = agentcore_config._get_runtime_session_id(optional_params)
        headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] = session_id
        runtime_user_id = agentcore_config._get_runtime_user_id(optional_params)
        if runtime_user_id:
            headers["X-Amzn-Bedrock-AgentCore-Runtime-User-Id"] = runtime_user_id

        # Merge per-request agent headers before signing so SigV4 covers them.
        # Reserved headers are stripped first to prevent client-controlled values
        # from spoofing the AgentCore runtime identity / SigV4 metadata.
        safe_extra_headers = _filter_reserved_headers(agent_extra_headers)
        if safe_extra_headers:
            headers.update(safe_extra_headers)

        # Sign the request (SigV4 or JWT depending on api_key presence)
        signed_headers, signed_body = agentcore_config.sign_request(
            headers=headers,
            optional_params=optional_params,
            request_data=json_rpc_body,
            api_base=url,
            api_key=optional_params.get("api_key"),
            model=agentcore_model,
            stream=stream,
        )

        # sign_request returns Optional[bytes] — ensure we have bytes
        if signed_body is None:
            signed_body = json.dumps(json_rpc_body).encode()

        return url, signed_headers, signed_body

    @staticmethod
    async def parse_sse_events(response: Any) -> AsyncIterator[Dict[str, Any]]:
        """
        Parse SSE events from an httpx streaming response.

        Reads line-by-line, parses `data:` lines as JSON, and yields each parsed dict.

        Args:
            response: httpx streaming response

        Yields:
            Parsed JSON dicts from SSE data lines
        """
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("data:"):
                data_str = line[len("data:") :].strip()
                if not data_str:
                    continue
                try:
                    event = json.loads(data_str)
                    yield event
                except json.JSONDecodeError:
                    verbose_logger.debug(
                        f"BedrockAgentCore A2A: Skipping non-JSON SSE line: {data_str[:100]}"
                    )
                    continue
