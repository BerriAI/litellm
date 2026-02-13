# +-------------------------------------------------------------+
#
#           Use Generic Guardrail API for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import fnmatch
import os
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GenericGuardrailAPIMetadata,
    GenericGuardrailAPIRequest,
    GenericGuardrailAPIResponse,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GUARDRAIL_NAME = "generic_guardrail_api"

# Headers whose values are forwarded as-is (case-insensitive). Glob patterns supported (e.g. x-stainless-*, x-litellm*).
_HEADER_VALUE_ALLOWLIST = frozenset({
    "host",
    "accept-encoding",
    "connection",
    "accept",
    "content-type",
    "user-agent",
    "x-stainless-*",
    "x-litellm-*",
    "content-length",
})

# Placeholder for headers that exist but are not on the allowlist (we don't expose their value).
_HEADER_PRESENT_PLACEHOLDER = "[present]"


def _header_value_allowed(header_name: str) -> bool:
    """Return True if this header's value may be forwarded (allowlist, including globs)."""
    lower = header_name.lower()
    if lower in _HEADER_VALUE_ALLOWLIST:
        return True
    for pattern in _HEADER_VALUE_ALLOWLIST:
        if "*" in pattern and fnmatch.fnmatch(lower, pattern):
            return True
    return False


def _sanitize_inbound_headers(headers: Any) -> Optional[Dict[str, str]]:
    """
    Sanitize inbound headers before passing them to a 3rd party guardrail service.

    - Allowlist: only headers in the allowlist have their values forwarded (exact + glob: x-stainless-*, x-litellm-*).
    - All other headers are included with value "[present]" so the guardrail knows the header existed.
    - Coerces values to str (for JSON serialization).
    """
    if not headers or not isinstance(headers, dict):
        return None

    sanitized: Dict[str, str] = {}
    for k, v in headers.items():
        if k is None:
            continue
        key = str(k)
        if _header_value_allowed(key):
            try:
                sanitized[key] = str(v)
            except Exception:
                continue
        else:
            sanitized[key] = _HEADER_PRESENT_PLACEHOLDER

    return sanitized or None


def _extract_inbound_headers(
    request_data: dict, logging_obj: Optional["LiteLLMLoggingObj"]
) -> Optional[Dict[str, str]]:
    """
    Extract inbound headers from available request context.

    We try multiple locations to support different call paths:
    - proxy endpoints: request_data["proxy_server_request"]["headers"]
    - if the guardrail is passed the proxy_server_request object directly
    - metadata headers captured in litellm_pre_call_utils
    - response hooks: fallback to logging_obj.model_call_details
    """
    # 1) Most common path (proxy): full request context in proxy_server_request
    headers = request_data.get("proxy_server_request", {}).get("headers")
    if headers:
        return _sanitize_inbound_headers(headers)

    # 2) Some guardrails pass proxy_server_request as request_data itself
    headers = request_data.get("headers")
    if headers:
        return _sanitize_inbound_headers(headers)

    # 3) Pre-call: headers stored in request metadata
    metadata_headers = (request_data.get("metadata") or {}).get("headers")
    if metadata_headers:
        return _sanitize_inbound_headers(metadata_headers)

    litellm_metadata_headers = (request_data.get("litellm_metadata") or {}).get(
        "headers"
    )
    if litellm_metadata_headers:
        return _sanitize_inbound_headers(litellm_metadata_headers)

    # 4) Post-call: headers not present on response; fallback to logging object
    if logging_obj and getattr(logging_obj, "model_call_details", None):
        try:
            details = logging_obj.model_call_details or {}
            headers = (
                details.get("litellm_params", {})
                .get("metadata", {})
                .get("headers", None)
            )
            if headers:
                return _sanitize_inbound_headers(headers)
        except Exception:
            pass

    return None


class GenericGuardrailAPI(CustomGuardrail):
    """
    Generic Guardrail API integration for LiteLLM.

    This integration allows you to use any guardrail API that follows the
    LiteLLM Basic Guardrail API spec without needing to write custom integration code.

    The API should accept a POST request with:
    {
        "text": str,
        "request_body": dict,
        "additional_provider_specific_params": dict
    }

    And return:
    {
        "action": "BLOCKED" | "NONE" | "GUARDRAIL_INTERVENED",
        "blocked_reason": str (optional, only if action is BLOCKED),
        "text": str (optional, modified text if action is GUARDRAIL_INTERVENED)
    }
    """

    def __init__(
        self,
        headers: Optional[Dict[str, Any]] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        additional_provider_specific_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.headers = headers or {}

        # If api_key is provided, add it as x-api-key header
        if api_key:
            self.headers["x-api-key"] = api_key

        base_url = api_base or os.environ.get("GENERIC_GUARDRAIL_API_BASE")

        if not base_url:
            raise ValueError(
                "api_base is required for Generic Guardrail API. "
                "Set GENERIC_GUARDRAIL_API_BASE environment variable or pass it in litellm_params"
            )

        # Append the endpoint path if not already present
        if not base_url.endswith("/beta/litellm_basic_guardrail_api"):
            base_url = base_url.rstrip("/")
            self.api_base = f"{base_url}/beta/litellm_basic_guardrail_api"
        else:
            self.api_base = base_url

        self.additional_provider_specific_params = (
            additional_provider_specific_params or {}
        )

        # Set supported event hooks
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
            ]

        super().__init__(**kwargs)

        verbose_proxy_logger.debug(
            "Generic Guardrail API initialized with api_base: %s", self.api_base
        )

    def _extract_user_api_key_metadata(
        self, request_data: dict
    ) -> GenericGuardrailAPIMetadata:
        """
        Extract user API key metadata from request_data.

        Args:
            request_data: Request data dictionary that may contain:
                - metadata (for input requests) with user_api_key_* fields
                - litellm_metadata (for output responses) with user_api_key_* fields

        Returns:
            GenericGuardrailAPIMetadata with extracted user information
        """
        result_metadata = GenericGuardrailAPIMetadata()

        # Get the source of metadata - try both locations
        # 1. For output responses: litellm_metadata (set by handlers with prefixed keys)
        # 2. For input requests: metadata (already present in request_data with prefixed keys)
        litellm_metadata = request_data.get("litellm_metadata", {})
        top_level_metadata = request_data.get("metadata", {})

        # Merge both sources, preferring litellm_metadata if both exist
        metadata_dict = {**top_level_metadata, **litellm_metadata}

        if not metadata_dict:
            return result_metadata

        # Dynamically iterate through GenericGuardrailAPIMetadata fields
        # and extract matching fields from the source metadata
        # Fields in metadata are already prefixed with 'user_api_key_'
        for field_name in GenericGuardrailAPIMetadata.__annotations__.keys():
            value = metadata_dict.get(field_name)
            if value is not None:
                result_metadata[field_name] = value  # type: ignore[literal-required]

        # handle user_api_key_token = user_api_key_hash
        if metadata_dict.get("user_api_key_token") is not None:
            result_metadata["user_api_key_hash"] = metadata_dict.get(
                "user_api_key_token"
            )

        verbose_proxy_logger.debug(
            "Generic Guardrail API: Extracted user metadata: %s",
            {k: v for k, v in result_metadata.items() if v is not None},
        )

        return result_metadata

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply the Generic Guardrail API to the given inputs.

        This is the main method that gets called by the framework.

        Args:
            inputs: Dictionary containing:
                - texts: List of texts to check
                - images: Optional list of images to check
                - tool_calls: Optional list of tool calls to check
            request_data: Request data dictionary containing user_api_key_dict and other metadata
            input_type: Whether this is a "request" or "response" guardrail
            logging_obj: Optional logging object for tracking the guardrail execution

        Returns:
            Tuple of (processed texts, processed images)

        Raises:
            Exception: If the guardrail blocks the request
        """
        verbose_proxy_logger.debug("Generic Guardrail API: Applying guardrail to text")

        # Extract texts and images from inputs
        texts = inputs.get("texts", [])
        images = inputs.get("images")
        tools = inputs.get("tools")
        structured_messages = inputs.get("structured_messages")
        tool_calls = inputs.get("tool_calls")
        model = inputs.get("model")

        # Use provided request_data or create an empty dict
        if request_data is None:
            request_data = {}

        request_body = request_data.get("body") or {}

        # Merge additional provider specific params from config and dynamic params
        additional_params = {**self.additional_provider_specific_params}

        # Get dynamic params from request if available
        dynamic_params = self.get_guardrail_dynamic_request_body_params(request_body)
        if dynamic_params:
            additional_params.update(dynamic_params)

        # Extract user API key metadata
        user_metadata = self._extract_user_api_key_metadata(request_data)
        inbound_headers = _extract_inbound_headers(request_data=request_data, logging_obj=logging_obj)

        # Create request payload
        guardrail_request = GenericGuardrailAPIRequest(
            litellm_call_id=logging_obj.litellm_call_id if logging_obj else None,
            litellm_trace_id=logging_obj.litellm_trace_id if logging_obj else None,
            texts=texts,
            request_data=user_metadata,
            request_headers=inbound_headers,
            litellm_version=litellm_version,
            images=images,
            tools=tools,
            structured_messages=structured_messages,
            tool_calls=tool_calls,
            additional_provider_specific_params=additional_params,
            input_type=input_type,
            model=model,
        )

        # Prepare headers
        headers = {"Content-Type": "application/json"}
        if self.headers:
            headers.update(self.headers)

        try:
            # Make the API request
            # Use mode="json" to ensure all iterables are converted to lists
            response = await self.async_handler.post(
                url=self.api_base,
                json=guardrail_request.model_dump(mode="json"),
                headers=headers,
            )

            response.raise_for_status()
            response_json = response.json()

            verbose_proxy_logger.debug(
                "Generic Guardrail API response: %s", response_json
            )

            guardrail_response = GenericGuardrailAPIResponse.from_dict(response_json)

            # Handle the response
            if guardrail_response.action == "BLOCKED":
                # Block the request
                error_message = (
                    guardrail_response.blocked_reason or "Content violates policy"
                )
                verbose_proxy_logger.warning(
                    "Generic Guardrail API blocked request: %s", error_message
                )
                raise GuardrailRaisedException(
                    guardrail_name=GUARDRAIL_NAME,
                    message=error_message,
                    should_wrap_with_default_message=False,
                )

            # Action is NONE or no modifications needed
            return_inputs = GenericGuardrailAPIInputs(texts=texts)
            if guardrail_response.texts:
                return_inputs["texts"] = guardrail_response.texts
            if guardrail_response.images:
                return_inputs["images"] = guardrail_response.images
            elif images:
                return_inputs["images"] = images
            if guardrail_response.tools:
                return_inputs["tools"] = guardrail_response.tools
            elif tools:
                return_inputs["tools"] = tools
            return return_inputs

        except GuardrailRaisedException:
            # Re-raise guardrail exceptions as-is
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                "Generic Guardrail API: failed to make request: %s", str(e)
            )
            raise Exception(f"Generic Guardrail API failed: {str(e)}")
