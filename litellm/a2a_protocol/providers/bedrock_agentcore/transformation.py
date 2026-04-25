"""
Transformation layer for Bedrock AgentCore A2A provider.

Constructs JSON-RPC envelopes, derives AgentCore URLs from model ARNs,
and signs requests via AmazonAgentCoreConfig (SigV4 or JWT).
"""

import json
from typing import Any, AsyncIterator, Dict, Tuple

from litellm._logging import verbose_logger
from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig


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
    ) -> Tuple[str, dict, bytes]:
        """
        Build the AgentCore URL, construct a JSON-RPC envelope, and sign the request.

        Args:
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams
            litellm_params: Agent's litellm_params (model, api_key, etc.)
            method: JSON-RPC method name (default: "message/send")
            stream: Whether this is a streaming request

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
