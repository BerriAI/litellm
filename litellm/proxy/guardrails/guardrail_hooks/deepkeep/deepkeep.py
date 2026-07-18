# +-------------------------------------------------------------+
#
#           Use DeepKeep AI Firewall for your LLM calls
#                   https://www.deepkeep.ai/
#
# +-------------------------------------------------------------+

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Optional

import httpx

from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
from litellm.exceptions import GuardrailRaisedException, Timeout
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GUARDRAIL_NAME = "deepkeep"

# Default DeepKeep API endpoint path
_DEEPKEEP_GUARDRAIL_ENDPOINT = "/v3/openai/beta/litellm_basic_guardrail_api"


class DeepKeepGuardrailMissingSecrets(Exception):
    """Exception raised when DeepKeep API key or firewall_id is missing."""

    pass


class DeepKeepGuardrailAPIError(Exception):
    """Exception raised when there's an error calling the DeepKeep API."""

    pass


class DeepKeepGuardrail(CustomGuardrail):
    """
    DeepKeep AI Firewall integration for LiteLLM.

    Provides content moderation, prompt injection detection, PII protection,
    and policy enforcement through the DeepKeep AI Firewall API.

    DeepKeep's firewall evaluates LLM inputs and outputs against a configurable
    set of guardrails (detectors + actions) managed via the DeepKeep platform.

    Configuration example (litellm config YAML):
        guardrails:
          - guardrail_name: deepkeep-firewall
            litellm_params:
              guardrail: deepkeep
              mode: pre_call
              api_key: os.environ/DEEPKEEP_API_KEY
              api_base: https://your-deepkeep-instance.example.com
              deepkeep_firewall_id: your-firewall-id
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        firewall_id: str | None = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        extra_headers: Mapping[str, str] | list[str] | None = None,
        **kwargs: Any,
    ):
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)

        # API key
        deepkeep_api_key = api_key or os.environ.get("DEEPKEEP_API_KEY")
        if not deepkeep_api_key:
            raise DeepKeepGuardrailMissingSecrets(
                "DeepKeep API key is required. Set the `DEEPKEEP_API_KEY` environment "
                "variable or pass `api_key` in the guardrail config."
            )
        self.deepkeep_api_key: str = deepkeep_api_key

        # Firewall ID
        self.firewall_id = firewall_id or os.environ.get("DEEPKEEP_FIREWALL_ID")
        if not self.firewall_id:
            raise DeepKeepGuardrailMissingSecrets(
                "DeepKeep firewall_id is required. Set the `DEEPKEEP_FIREWALL_ID` environment "
                "variable or pass `deepkeep_firewall_id` in the guardrail config."
            )

        # API base URL
        base_url = api_base or os.environ.get("DEEPKEEP_API_BASE")
        if not base_url:
            raise DeepKeepGuardrailMissingSecrets(
                "DeepKeep API base URL is required. Set the `DEEPKEEP_API_BASE` environment "
                "variable or pass `api_base` in the guardrail config."
            )

        # Normalize the API base – ensure it ends with the guardrail endpoint
        base_url = base_url.rstrip("/")
        if base_url.endswith(_DEEPKEEP_GUARDRAIL_ENDPOINT.rstrip("/")):
            self.api_base = base_url
        else:
            self.api_base = f"{base_url}{_DEEPKEEP_GUARDRAIL_ENDPOINT}"

        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = unreachable_fallback
        if extra_headers is not None and not isinstance(extra_headers, Mapping):
            verbose_proxy_logger.warning(
                "DeepKeep guardrail ignoring `extra_headers`: expected a mapping of header name to value, got %s. "
                "`litellm_params.extra_headers` is a list of header names to forward and is not supported by this guardrail",
                type(extra_headers).__name__,
            )
        self.extra_headers: dict[str, str] = dict(extra_headers) if isinstance(extra_headers, Mapping) else {}

        # Set supported event hooks
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
            ]

        super().__init__(**kwargs)

        verbose_proxy_logger.debug(
            "DeepKeep guardrail initialized: guardrail_name=%s, api_base=%s, firewall_id=%s",
            kwargs.get("guardrail_name", "unknown"),
            self.api_base,
            self.firewall_id,
        )

    def _extract_user_api_key_metadata(self, request_data: dict) -> dict[str, Any]:
        """
        Extract user API key metadata from request_data for the DeepKeep API.

        Args:
            request_data: Request data dictionary containing metadata.

        Returns:
            Dictionary with user API key metadata fields.
        """
        result_metadata: dict[str, Any] = {}

        litellm_metadata = request_data.get("litellm_metadata", {})
        top_level_metadata = request_data.get("metadata", {})
        metadata_dict = {**top_level_metadata, **litellm_metadata}

        if not metadata_dict:
            return result_metadata

        # Extract standard user API key fields
        _METADATA_KEYS = [
            "user_api_key_hash",
            "user_api_key_alias",
            "user_api_key_user_id",
            "user_api_key_user_email",
            "user_api_key_team_id",
            "user_api_key_team_alias",
            "user_api_key_end_user_id",
            "user_api_key_org_id",
        ]
        for key in _METADATA_KEYS:
            value = metadata_dict.get(key)
            if value is not None:
                result_metadata[key] = value

        # Handle the token → hash alias (only when no explicit hash was provided)
        if metadata_dict.get("user_api_key_token") is not None and "user_api_key_hash" not in result_metadata:
            result_metadata["user_api_key_hash"] = metadata_dict["user_api_key_token"]

        return result_metadata

    def _build_request_headers(self) -> dict[str, str]:
        """Build HTTP headers for the DeepKeep API request."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-API-Key": self.deepkeep_api_key,
        }
        if self.extra_headers:
            headers.update(self.extra_headers)
        return headers

    def _fail_open_passthrough(
        self,
        *,
        inputs: GenericGuardrailAPIInputs,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"],
        error: Exception,
        http_status_code: int | None = None,
    ) -> GenericGuardrailAPIInputs:
        """Allow the request to proceed when the guardrail is unreachable (fail-open mode)."""
        status_suffix = f" http_status_code={http_status_code}" if http_status_code else ""
        verbose_proxy_logger.critical(
            "DeepKeep guardrail unreachable (fail-open). Proceeding without guardrail.%s "
            "guardrail_name=%s api_base=%s input_type=%s litellm_call_id=%s litellm_trace_id=%s",
            status_suffix,
            getattr(self, "guardrail_name", None),
            getattr(self, "api_base", None),
            input_type,
            getattr(logging_obj, "litellm_call_id", None) if logging_obj else None,
            getattr(logging_obj, "litellm_trace_id", None) if logging_obj else None,
            exc_info=error,
        )
        return_inputs: GenericGuardrailAPIInputs = {}
        return_inputs.update(inputs)
        return return_inputs

    def _handle_guardrail_request_error(
        self,
        error: Exception,
        inputs: GenericGuardrailAPIInputs,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"],
        is_unreachable: bool = True,
    ) -> GenericGuardrailAPIInputs:
        """Handle errors from the DeepKeep API with fail-open/fail-closed logic."""
        if is_unreachable and self.unreachable_fallback == "fail_open":
            http_status_code = getattr(getattr(error, "response", None), "status_code", None)
            return self._fail_open_passthrough(
                inputs=inputs,
                input_type=input_type,
                logging_obj=logging_obj,
                error=error,
                **({"http_status_code": http_status_code} if http_status_code else {}),
            )
        verbose_proxy_logger.error("DeepKeep guardrail API error: %s", str(error))
        raise DeepKeepGuardrailAPIError(f"DeepKeep guardrail API failed: {str(error)}")

    @staticmethod
    def _build_return_inputs(
        *,
        response_json: dict[str, Any],
        texts: list,
        images: Any | None,
        tools: Any | None,
        tool_calls: Any | None,
        structured_messages: Any | None,
    ) -> GenericGuardrailAPIInputs:
        """Merge original inputs with any guardrail-modified values from the API response.

        Presence is checked with ``is not None`` (not truthiness) so that an
        intentional empty-list replacement such as ``texts: []`` or
        ``tool_calls: []`` is honoured and forwarded downstream rather than
        silently discarded in favour of the original content.
        """
        return_inputs = GenericGuardrailAPIInputs(texts=texts)
        if response_json.get("texts") is not None:
            return_inputs["texts"] = response_json["texts"]
        if response_json.get("images") is not None:
            return_inputs["images"] = response_json["images"]
        elif images is not None:
            return_inputs["images"] = images
        if response_json.get("tools") is not None:
            return_inputs["tools"] = response_json["tools"]
        elif tools is not None:
            return_inputs["tools"] = tools
        if response_json.get("tool_calls") is not None:
            return_inputs["tool_calls"] = response_json["tool_calls"]
        elif tool_calls is not None:
            return_inputs["tool_calls"] = tool_calls
        if response_json.get("structured_messages") is not None:
            return_inputs["structured_messages"] = response_json["structured_messages"]
        elif structured_messages is not None:
            return_inputs["structured_messages"] = structured_messages
        return return_inputs

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply the DeepKeep AI Firewall guardrail to the given inputs.

        This is the main method called by the LiteLLM framework for guardrail evaluation.

        Args:
            inputs: Dictionary containing texts, images, tools, tool_calls, structured_messages.
            request_data: Request data dictionary containing metadata.
            input_type: Whether this is a "request" (pre-call) or "response" (post-call) guardrail.
            logging_obj: Optional logging object for tracking the guardrail execution.

        Returns:
            GenericGuardrailAPIInputs with original or modified content.

        Raises:
            GuardrailRaisedException: If the guardrail blocks the request.
            DeepKeepGuardrailAPIError: If the API call fails (in fail-closed mode).
        """
        verbose_proxy_logger.debug("DeepKeep guardrail: applying guardrail, input_type=%s", input_type)

        texts = inputs.get("texts", [])
        images = inputs.get("images")
        tools = inputs.get("tools")
        structured_messages = inputs.get("structured_messages")
        tool_calls = inputs.get("tool_calls")
        model = inputs.get("model")

        if request_data is None:
            request_data = {}

        request_body = request_data.get("body") or {}

        # Merge additional provider-specific params from config and dynamic params
        additional_params: dict[str, Any] = {"firewall_id": self.firewall_id}
        dynamic_params = self.get_guardrail_dynamic_request_body_params(request_body)
        if dynamic_params:
            additional_params.update({k: v for k, v in dynamic_params.items() if k != "firewall_id"})

        # Extract user API key metadata
        user_metadata = self._extract_user_api_key_metadata(request_data)

        # Build request payload
        guardrail_request: dict[str, Any] = {
            "litellm_call_id": (logging_obj.litellm_call_id if logging_obj else None),
            "litellm_trace_id": (logging_obj.litellm_trace_id if logging_obj else None),
            "texts": texts,
            "request_data": user_metadata,
            "litellm_version": litellm_version,
            "images": images,
            "tools": tools,
            "structured_messages": structured_messages,
            "tool_calls": tool_calls,
            "additional_provider_specific_params": additional_params,
            "input_type": input_type,
            "model": model,
        }

        headers = self._build_request_headers()

        try:
            response = await self.async_handler.post(
                url=self.api_base,
                json=guardrail_request,
                headers=headers,
            )

            response.raise_for_status()
            response_json = response.json()

            verbose_proxy_logger.debug("DeepKeep guardrail response: %s", response_json)

            action = response_json.get("action", "NONE")

            if action == "BLOCKED":
                error_message = response_json.get("blocked_reason") or "Content violates policy"
                verbose_proxy_logger.warning("DeepKeep guardrail blocked request: %s", error_message)
                raise GuardrailRaisedException(
                    guardrail_name=GUARDRAIL_NAME,
                    message=error_message,
                    should_wrap_with_default_message=False,
                )

            return self._build_return_inputs(
                response_json=response_json,
                texts=texts,
                images=images,
                tools=tools,
                tool_calls=tool_calls,
                structured_messages=structured_messages,
            )

        except GuardrailRaisedException:
            raise
        except Timeout as e:
            return self._handle_guardrail_request_error(e, inputs, input_type, logging_obj)
        except httpx.HTTPStatusError as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            is_unreachable = status_code in (502, 503, 504)
            return self._handle_guardrail_request_error(
                e, inputs, input_type, logging_obj, is_unreachable=is_unreachable
            )
        except httpx.RequestError as e:
            return self._handle_guardrail_request_error(e, inputs, input_type, logging_obj)
        except Exception as e:  # noqa: BLE001  # route unexpected errors through fail-open/closed handling
            return self._handle_guardrail_request_error(e, inputs, input_type, logging_obj, is_unreachable=False)

    @staticmethod
    def get_config_model() -> type | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.deepkeep import (
            DeepKeepGuardrailConfigModel,
        )

        return DeepKeepGuardrailConfigModel
