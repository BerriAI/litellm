"""
Utilities for mapping exceptions to Anthropic error format.

Similar to litellm/litellm_core_utils/exception_mapping_utils.py but for Anthropic response format.
"""

import json
import re
from typing import Dict, Optional

from litellm.litellm_core_utils.safe_json_loads import safe_json_loads

from .exceptions import AnthropicErrorResponse, AnthropicErrorType

# Leading `litellm.SomethingError: ` / `litellm.SomethingException: ` prefix that
# LiteLLM exception classes prepend to their `.message` (often stacked, e.g.
# `litellm.ContextWindowExceededError: litellm.BadRequestError: ...`).
_LITELLM_CLASS_PREFIX = re.compile(r"^\s*litellm\.\w+(?:Error|Exception):\s*")

# Provider exception prefix, e.g. `AnthropicException - {json}` /
# `VertexAIException - ...`. Appears once, right before the raw upstream body.
# Anchored to known provider names rather than `\w+Exception` so a generic
# `TimeoutException - <real error>` / `ConnectionException - <real error>`
# / `RequestException - <real error>` does NOT swallow the front of a
# legitimate runtime error string. Maintained from `litellm/llms/**`
# `<Name>Exception` classes plus common aliases LiteLLM emits.
_PROVIDER_EXCEPTION_NAMES = (
    "Anthropic",
    "AzureOpenAI",
    "Azure",
    "AWSBedrock",
    "Bedrock",
    "Cerebras",
    "ClarifAI",
    "Cohere",
    "CometAPI",
    "Databricks",
    "DeepInfra",
    "Deepgram",
    "DeepSeek",
    "Deepseek",
    "ElevenLabs",
    "FireworksAI",
    "Fireworks",
    "Gemini",
    "Groq",
    "HuggingFace",
    "Huggingface",
    "Hyperbolic",
    "Minimax",
    "MistralAudioTranscription",
    "Mistral",
    "NLPCloud",
    "NvidiaRiva",
    "OllamaChat",
    "Ollama",
    "OpenAI",
    "OpenRouter",
    "OVHCloud",
    "Perplexity",
    "Predibase",
    "Replicate",
    "Sambanova",
    "ScalewayAudioTranscription",
    "Snowflake",
    "TogetherAI",
    "Together",
    "Topaz",
    "VercelAIGateway",
    "VertexAI",
    "Watsonx",
    "XAI",
)
_PROVIDER_EXCEPTION_PREFIX = re.compile(
    r"^\s*(?:" + "|".join(_PROVIDER_EXCEPTION_NAMES) + r")Exception\s*-\s*"
)


# HTTP status code -> Anthropic error type
# Source: https://docs.anthropic.com/en/api/errors
ANTHROPIC_ERROR_TYPE_MAP: Dict[int, AnthropicErrorType] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


class AnthropicExceptionMapping:
    """
    Helper class for mapping exceptions to Anthropic error format.

    Similar pattern to ExceptionCheckers in litellm_core_utils/exception_mapping_utils.py
    """

    @staticmethod
    def get_error_type(status_code: int) -> AnthropicErrorType:
        """Map HTTP status code to Anthropic error type."""
        return ANTHROPIC_ERROR_TYPE_MAP.get(status_code, "api_error")

    @staticmethod
    def _strip_litellm_wrapper_prefixes(raw_message: str) -> str:
        """
        Strip LiteLLM/provider wrapper prefixes off an exception message so the
        embedded upstream body (often a JSON string) is exposed.

        LiteLLM exception classes prepend `litellm.<Class>: ` to `.message`,
        sometimes stacked, and providers prepend `<Provider>Exception - `.
        For example:

            "litellm.RateLimitError: AnthropicException - {\"type\":\"error\",...}"
                -> "{\"type\":\"error\",...}"

        Idempotent: returns the input unchanged when no prefix is present.
        """
        message = raw_message
        # Strip stacked `litellm.XxxError: ` prefixes until none remain.
        while True:
            stripped = _LITELLM_CLASS_PREFIX.sub("", message, count=1)
            if stripped == message:
                break
            message = stripped
        # Strip a single `<Provider>Exception - ` prefix.
        message = _PROVIDER_EXCEPTION_PREFIX.sub("", message, count=1)
        return message

    @staticmethod
    def create_error_response(
        status_code: int,
        message: str,
        request_id: Optional[str] = None,
    ) -> AnthropicErrorResponse:
        """
        Create an Anthropic-formatted error response dict.

        Anthropic error format:
        {
            "type": "error",
            "error": {"type": "...", "message": "..."},
            "request_id": "req_..."
        }
        """
        error_type = AnthropicExceptionMapping.get_error_type(status_code)

        response: AnthropicErrorResponse = {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        }

        if request_id:
            response["request_id"] = request_id

        return response

    @staticmethod
    def extract_error_message(raw_message: str) -> str:
        """
        Extract error message from various provider response formats.

        Handles:
        - Bedrock:           {"detail": {"message": "..."}}
        - AWS:               {"Message": "..."}
        - OpenAI / new-api:  {"error": {"message": "...", ...}}
        - Generic:           {"message": "..."}
        - Plain strings
        """
        parsed = safe_json_loads(raw_message)
        if isinstance(parsed, dict):
            return AnthropicExceptionMapping._extract_message_from_dict(
                parsed, raw_message
            )
        return raw_message

    @staticmethod
    def _is_anthropic_error_dict(parsed: dict) -> bool:
        """
        Check if a parsed dict is in Anthropic error format.

        Anthropic error format:
        {
            "type": "error",
            "error": {"type": "...", "message": "..."}
        }
        """
        return (
            parsed.get("type") == "error"
            and isinstance(parsed.get("error"), dict)
            and "type" in parsed["error"]
            and "message" in parsed["error"]
        )

    @staticmethod
    def _extract_message_from_dict(parsed: dict, raw_message: str) -> str:
        """
        Extract error message from a parsed provider-specific dict.

        Handles:
        - Bedrock:           {"detail": {"message": "..."}}
        - AWS:               {"Message": "..."}
        - OpenAI / new-api:  {"error": {"message": "...", ...}}
        - Generic:           {"message": "..."}

        Falls back to ``raw_message`` only when no recognized message field
        is present, so an upstream JSON body's clean message is preferred
        over a raw string that may carry post-decode debug suffixes.
        """
        # Bedrock format
        if "detail" in parsed and isinstance(parsed["detail"], dict):
            return parsed["detail"].get("message", raw_message)
        # OpenAI / new-api / OpenAI-compatible nested error
        err = parsed.get("error")
        if isinstance(err, dict):
            nested = err.get("message")
            if isinstance(nested, str) and nested:
                return nested
        # AWS/generic format
        return parsed.get("Message") or parsed.get("message") or raw_message

    @staticmethod
    def transform_to_anthropic_error(
        status_code: int,
        raw_message: str,
        request_id: Optional[str] = None,
    ) -> AnthropicErrorResponse:
        """
        Transform an error message to Anthropic format.

        - If already in Anthropic format: passthrough unchanged
        - Otherwise: extract message and create Anthropic error

        Parses JSON only once for efficiency.

        Args:
            status_code: HTTP status code
            raw_message: Raw error message (may be JSON string or plain text)
            request_id: Optional request ID to include

        Returns:
            AnthropicErrorResponse dict
        """
        # Strip LiteLLM/provider wrapper prefixes so an embedded upstream
        # Anthropic error body can be detected and passed through unchanged.
        raw_message = AnthropicExceptionMapping._strip_litellm_wrapper_prefixes(
            raw_message
        )

        # Try to parse as JSON once.
        parsed: Optional[dict] = safe_json_loads(raw_message)
        if not isinstance(parsed, dict):
            parsed = None

        # Fallback for messages where an Anthropic-shaped JSON body is
        # followed by appended debug text (e.g. the Router's
        # ". Received Model Group=...\nAvailable Model Group Fallbacks=..."
        # suffix). `safe_json_loads` rejects trailing garbage; `raw_decode`
        # parses the leading JSON value and ignores anything after it.
        if parsed is None:
            try:
                obj, _ = json.JSONDecoder().raw_decode(raw_message.lstrip())
                if isinstance(obj, dict):
                    parsed = obj
            except json.JSONDecodeError:
                pass

        # If parsed and already in Anthropic format - passthrough
        if parsed is not None and AnthropicExceptionMapping._is_anthropic_error_dict(parsed):
            # Optionally add request_id if provided and not present
            if request_id and "request_id" not in parsed:
                parsed["request_id"] = request_id
            return parsed  # type: ignore

        # Extract message - use parsed dict if available, otherwise raw string
        if parsed is not None:
            message = AnthropicExceptionMapping._extract_message_from_dict(parsed, raw_message)
        else:
            message = raw_message

        return AnthropicExceptionMapping.create_error_response(
            status_code=status_code,
            message=message,
            request_id=request_id,
        )
