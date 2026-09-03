"""
Pass-Through Endpoint Message Handler for Unified Guardrails

Architecture overview
---------------------
LiteLLM supports two code-paths for requests:

1. **Standard routes** (``/chat/completions``, ``/v1/messages``, …)
   These go through ``OpenAIChatCompletionsHandler`` which extracts
   ``additional_provider_specific_params`` (e.g. ``secrets.config.mode: block``)
   and forwards them into the guardrail hook before dispatch.

2. **Provider-native passthrough routes** (``/v1beta/models/…:generateContent``,
   ``/model/…/converse``, …).
   These bypass the standard request pipeline.  This module handles guardrail
   translation for those routes so that guardrails still run with the correct
   parameters.

Dispatch chain
--------------
The top-level dispatcher is ``LlmPassthroughRouteHandler``.

* For **bedrock** it delegates to ``BedrockPassthroughGuardrailHandler``
  (which knows how to walk the Converse message schema).
* For **every other provider** (gemini, vertex_ai, anthropic-native, …) it
  delegates to ``PassThroughEndpointHandler`` — the generic fallback defined
  in this file.  The generic handler scans the whole request / response
  payload and forwards guardrail params unchanged, so ``mode: block`` is
  honoured on all providers, not just bedrock.

  Prior to the fix for issue #37638, the dispatcher silently returned the
  original data for unknown providers instead of delegating, which meant
  ``mode: block`` was silently downgraded to the guardrail's built-in default
  (usually ``redact``).

Adding support for a new provider
----------------------------------
If a provider uses a non-standard message schema that needs special treatment
(e.g. a deeply nested content block format), register a dedicated handler in
``_get_provider_handlers()`` following the bedrock pattern.  For providers
that use a flat JSON payload the generic ``PassThroughEndpointHandler`` is
already sufficient — no new code is needed.
"""

from typing import TYPE_CHECKING, Any, Final, Optional

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.proxy._types import PassThroughGuardrailSettings
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging


class PassThroughEndpointHandler(BaseTranslation):
    """
    Generic guardrail handler for provider-native passthrough routes.

    This handler is the *fallback* for any passthrough provider that does not
    have its own dedicated handler (everything except bedrock-Converse).  It:

    * Reads the optional ``passthrough_guardrails_config`` from
      ``litellm_logging_obj`` to discover any JSON-path field-targeting rules.
    * If field-targeting rules are present, it extracts only those fields from
      the payload and runs the guardrail on them.
    * If no targeting rules are present, it serialises the entire
      (non-internal) request / response payload and runs the guardrail on that.

    This approach is intentionally provider-agnostic: because it works on the
    raw JSON dict it handles Gemini ``generateContent``, Vertex AI, Anthropic
    native messages, and any future provider without requiring provider-specific
    schema knowledge.

    Note: this handler does *not* write guardrailed text back into the payload
    (i.e. it is detection / blocking only, not redaction).  Redaction on
    provider-native routes would require knowing the provider's exact content
    schema, which is left to dedicated handlers.
    """

    def _get_guardrail_settings(
        self,
        litellm_logging_obj: Optional["LiteLLMLoggingObj"],
        guardrail_name: str | None,
    ) -> PassThroughGuardrailSettings | None:
        """
        Get the guardrail settings for a specific guardrail from logging_obj.
        """
        from litellm.proxy.pass_through_endpoints.passthrough_guardrails import (
            PassthroughGuardrailHandler,
        )

        if litellm_logging_obj is None:
            return None

        passthrough_config: Final = getattr(litellm_logging_obj, "passthrough_guardrails_config", None)
        if not passthrough_config or not guardrail_name:
            return None

        return PassthroughGuardrailHandler.get_settings(passthrough_config, guardrail_name)

    def _extract_text_for_guardrail(
        self,
        data: dict,
        field_expressions: list[str] | None,
    ) -> str:
        """
        Extract text from data for guardrail processing.

        If field_expressions provided, extracts only those fields.
        Otherwise, returns the full payload as JSON.
        """
        from litellm.proxy.pass_through_endpoints.jsonpath_extractor import (
            JsonPathExtractor,
        )

        if field_expressions:
            text: Final = JsonPathExtractor.extract_fields(
                data=data,
                jsonpath_expressions=field_expressions,
            )
            verbose_proxy_logger.debug(
                "PassThroughEndpointHandler: Extracted targeted fields: %s",
                text[:200] if text else None,
            )
            return text

        # Use entire payload, excluding internal fields
        from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

        payload_to_check: Final = {
            k: v for k, v in data.items() if not k.startswith("_") and k not in ("metadata", "litellm_logging_obj")
        }
        verbose_proxy_logger.debug("PassThroughEndpointHandler: Using full payload for guardrail")
        return safe_dumps(payload_to_check)

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> Any:
        """
        Process input by applying guardrails to targeted fields or full payload.
        """
        guardrail_name: Final = guardrail_to_apply.guardrail_name
        verbose_proxy_logger.debug(
            "PassThroughEndpointHandler: Processing input for guardrail=%s",
            guardrail_name,
        )

        # Get field targeting settings for this guardrail
        settings: Final = self._get_guardrail_settings(litellm_logging_obj, guardrail_name)
        field_expressions: Final = settings.request_fields if settings else None

        # Extract text to check
        text_to_check: Final = self._extract_text_for_guardrail(data, field_expressions)

        if not text_to_check:
            verbose_proxy_logger.debug("PassThroughEndpointHandler: No text to check, skipping guardrail")
            return data

        # Apply guardrail (pass-through doesn't modify the text, just checks it)
        inputs: Final = GenericGuardrailAPIInputs(texts=[text_to_check])
        # Include model information if available
        model: Final = data.get("model")
        if model:
            inputs["model"] = model
        _guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )

        return data

    async def process_output_response(
        self,
        response: Any,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Any | None = None,
        request_data: dict | None = None,
    ) -> Any:
        """
        Process output response by applying guardrails to targeted fields.

        Args:
            response: The response to process
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata to pass to guardrails
        """
        if not isinstance(response, dict):
            verbose_proxy_logger.debug("PassThroughEndpointHandler: Response is not a dict, skipping")
            return response

        guardrail_name: Final = guardrail_to_apply.guardrail_name
        verbose_proxy_logger.debug(
            "PassThroughEndpointHandler: Processing output for guardrail=%s",
            guardrail_name,
        )

        # Get field targeting settings for this guardrail
        settings: Final = self._get_guardrail_settings(litellm_logging_obj, guardrail_name)
        field_expressions: Final = settings.response_fields if settings else None

        # Extract text to check
        text_to_check: Final = self._extract_text_for_guardrail(response, field_expressions)

        if not text_to_check:
            return response

        # Use the real request_data if provided (proxy path), otherwise
        # create a standalone dict (SDK / direct-call path).
        if request_data is None:
            request_data = {"response": response} if not isinstance(response, dict) else response.copy()
        else:
            if "response" not in request_data:
                request_data["response"] = response if not isinstance(response, dict) else response.copy()

        # Add user API key metadata with prefixed keys
        if "litellm_metadata" not in request_data:
            user_metadata: Final = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
            if user_metadata:
                request_data["litellm_metadata"] = user_metadata

        # Apply guardrail (pass-through doesn't modify the text, just checks it)
        inputs: Final = GenericGuardrailAPIInputs(texts=[text_to_check])
        # Include model information from the response if available
        response_model: Final = response.get("model") if isinstance(response, dict) else None
        if response_model:
            inputs["model"] = response_model
        _guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="response",
            logging_obj=litellm_logging_obj,
        )

        return response


# -------------------------------------------------------------------------
# Provider-specific handler registry
# -------------------------------------------------------------------------
# Maps custom_llm_provider strings to their dedicated guardrail translation
# handler class.  Only providers that need *schema-aware* translation (i.e.
# they have a structured message format different from a flat JSON payload)
# require a dedicated entry here.
#
# Currently only "bedrock" is registered because the Bedrock Converse API
# uses a deeply nested content-block schema that requires special extraction
# and write-back logic.
#
# All other providers (gemini, vertex_ai, anthropic-native, …) are served by
# the generic ``PassThroughEndpointHandler`` fallback defined below.
# -------------------------------------------------------------------------
_PROVIDER_HANDLERS: dict[str, type[BaseTranslation]] = {}


def _get_provider_handlers() -> dict[str, type[BaseTranslation]]:
    """Return the registry of provider-specific guardrail handlers (lazy-init)."""
    global _PROVIDER_HANDLERS
    if not _PROVIDER_HANDLERS:
        # Import is deferred to avoid a circular-import at module load time.
        from litellm.llms.bedrock.passthrough.guardrail_translation.handler import (
            BedrockPassthroughGuardrailHandler,
        )

        _PROVIDER_HANDLERS = {"bedrock": BedrockPassthroughGuardrailHandler}
    return _PROVIDER_HANDLERS


def _generic_passthrough_handler() -> BaseTranslation:
    """
    Generic fallback used for all provider-native passthrough routes that do
    not have a dedicated handler (e.g. Gemini generateContent, Vertex AI).

    This ensures guardrail params — including ``additional_provider_specific_params``
    such as ``secrets.config.mode: block`` — are forwarded and honoured on
    every passthrough provider, not just bedrock.

    Fixes: https://github.com/BerriAI/litellm/issues/37638
    """
    return PassThroughEndpointHandler()


class LlmPassthroughRouteHandler(BaseTranslation):
    """
    Top-level dispatcher for ``CallTypes.allm_passthrough_route`` guardrail
    translation.

    Decision tree
    -------------
    1. Look up ``data["custom_llm_provider"]`` in the provider handler registry.
    2. If a dedicated handler exists (currently only ``bedrock``), delegate to it.
    3. Otherwise, delegate to ``PassThroughEndpointHandler`` — the generic
       fallback — which scans the full payload and correctly forwards guardrail
       params including ``additional_provider_specific_params``.

    Before the fix for issue #37638 step 3 was missing: unknown providers
    would receive a silent early-return, causing ``mode=block`` to be
    downgraded to the guardrail's default mode (``redact``) without any
    warning or error.
    """

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> Any:
        provider: Final = data.get("custom_llm_provider")
        handler_cls: Final = _get_provider_handlers().get(provider or "")
        if handler_cls is None:
            # No provider-specific handler registered (e.g. gemini, vertex_ai).
            # Fall back to the generic handler so that guardrail params —
            # including additional_provider_specific_params.secrets.config.mode
            # — are forwarded and blocking guardrails are honoured.
            # See: https://github.com/BerriAI/litellm/issues/37638
            verbose_proxy_logger.debug(
                "LlmPassthroughRouteHandler: no dedicated handler for provider=%s, "
                "delegating to generic PassThroughEndpointHandler",
                provider,
            )
            return await _generic_passthrough_handler().process_input_messages(
                data=data,
                guardrail_to_apply=guardrail_to_apply,
                litellm_logging_obj=litellm_logging_obj,
            )
        return await handler_cls().process_input_messages(
            data=data,
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=litellm_logging_obj,
        )

    async def process_output_response(
        self,
        response: Any,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Any | None = None,
        request_data: dict | None = None,
    ) -> Any:
        provider: Final = (request_data or {}).get("custom_llm_provider")
        handler_cls: Final = _get_provider_handlers().get(provider or "")
        if handler_cls is None:
            # No provider-specific handler registered (e.g. gemini, vertex_ai).
            # Fall back to the generic handler so blocking guardrails are
            # honoured on response payloads too.
            # See: https://github.com/BerriAI/litellm/issues/37638
            verbose_proxy_logger.debug(
                "LlmPassthroughRouteHandler: no dedicated handler for provider=%s, "
                "delegating to generic PassThroughEndpointHandler",
                provider,
            )
            return await _generic_passthrough_handler().process_output_response(
                response=response,
                guardrail_to_apply=guardrail_to_apply,
                litellm_logging_obj=litellm_logging_obj,
                user_api_key_dict=user_api_key_dict,
                request_data=request_data,
            )
        return await handler_cls().process_output_response(
            response=response,
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=litellm_logging_obj,
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
        )

    @staticmethod
    def is_event_stream_response(provider: str | None, content_type: str) -> bool:
        handler_cls: Final = _get_provider_handlers().get(provider or "")
        detector: Final = getattr(handler_cls, "is_event_stream_content_type", None)
        if detector is None:
            return False
        return detector(content_type)

    @staticmethod
    def event_stream_media_type(provider: str | None) -> str | None:
        handler_cls: Final = _get_provider_handlers().get(provider or "")
        getter: Final = getattr(handler_cls, "event_stream_media_type", None)
        if getter is None:
            return None
        return getter()

    @staticmethod
    def _resolve_event_stream_de_anonymizer(provider: str | None):
        handler_cls: Final = _get_provider_handlers().get(provider or "")
        return getattr(handler_cls, "de_anonymize_event_stream", None)

    @staticmethod
    def supports_event_stream_de_anonymization(provider: str | None, endpoint: str | None) -> bool:
        handler_cls: Final = _get_provider_handlers().get(provider or "")
        endpoint_check: Final = getattr(handler_cls, "event_stream_endpoint_is_de_anonymizable", None)
        if endpoint_check is None:
            return False
        return endpoint_check(endpoint or "")

    @staticmethod
    async def de_anonymize_event_stream(
        body_bytes: bytes,
        proxy_logging_obj: "ProxyLogging",
        user_api_key_dict: "UserAPIKeyAuth",
        data: dict,
    ) -> bytes:
        provider: Final = data.get("custom_llm_provider")
        de_anonymize: Final = LlmPassthroughRouteHandler._resolve_event_stream_de_anonymizer(provider)
        if de_anonymize is None:
            verbose_proxy_logger.debug(
                "LlmPassthroughRouteHandler: no event-stream handler for provider=%s, leaving stream unmodified",
                provider,
            )
            return body_bytes
        return await de_anonymize(
            body_bytes=body_bytes,
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_dict=user_api_key_dict,
            data=data,
        )
