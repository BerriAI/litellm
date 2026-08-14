# +-------------------------------------------------------------+
#
#           Use Generic Guardrail API for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import fnmatch
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, Optional

import httpx
from pydantic import JsonValue

from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
from litellm.exceptions import GuardrailRaisedException, Timeout
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    get_session_id_from_request_data,
    log_guardrail_information,
    suppress_guardrail_information_record,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GenericGuardrailAPIMetadata,
    GenericGuardrailAPIRequest,
    GenericGuardrailAPIResponse,
    GuardrailToolParam,
)
from litellm.types.utils import GenericGuardrailAPIInputs

from .background_dispatch import (
    DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT,
    BackgroundDispatcher,
)
from .payload_policy import (
    PayloadLoss,
    PayloadPolicy,
    compile_patterns,
    merge_guardrailed_texts,
    resolve_exclude_fields,
    shape_payload,
)
from .record_scope import (
    DEFAULT_GUARDRAIL_INFORMATION_SCOPE,
    GuardrailInformationScope,
    RecordScope,
)
from .request_filters import (
    SkipDecisionStore,
    SkipPolicy,
    call_type_allowed,
    identity_matches_skip,
    request_matches_skip,
    validate_call_types,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME: Final = "generic_guardrail_api"

# Headers whose values are forwarded as-is (case-insensitive). Glob patterns supported (e.g. x-stainless-*, x-litellm*).
_HEADER_VALUE_ALLOWLIST: Final = frozenset(
    {
        "host",
        "accept-encoding",
        "connection",
        "accept",
        "content-type",
        "user-agent",
        "x-stainless-*",
        "x-litellm-*",
        "content-length",
    }
)

# Placeholder for headers that exist but are not on the allowlist (we don't expose their value).
_HEADER_PRESENT_PLACEHOLDER: Final = "[present]"


def _header_value_allowed(
    header_name: str,
    extra_allowlist: set[str] | None = None,
) -> bool:
    """Return True if this header's value may be forwarded (allowlist, including globs and extra_headers)."""
    lower: Final = header_name.lower()
    if lower in _HEADER_VALUE_ALLOWLIST:
        return True
    for pattern in _HEADER_VALUE_ALLOWLIST:
        if "*" in pattern and fnmatch.fnmatch(lower, pattern):
            return True
    if extra_allowlist and lower in extra_allowlist:
        return True
    return False


def _sanitize_inbound_headers(
    headers: Any,
    extra_allowlist: set[str] | None = None,
) -> dict[str, str] | None:
    """
    Sanitize inbound headers before passing them to a 3rd party guardrail service.

    - Allowlist: default allowlist + extra_allowlist (from litellm_params.extra_headers); only these have values forwarded.
    - All other headers are included with value "[present]" so the guardrail knows the header existed.
    - Coerces values to str (for JSON serialization).
    """
    if not headers or not isinstance(headers, dict):
        return None

    sanitized: Final[dict[str, str]] = {}
    for k, v in headers.items():
        if k is None:
            continue
        key = str(k)
        if _header_value_allowed(key, extra_allowlist=extra_allowlist):
            try:
                sanitized[key] = str(v)
            except Exception:
                continue
        else:
            sanitized[key] = _HEADER_PRESENT_PLACEHOLDER

    return sanitized or None


def _extract_inbound_headers(
    request_data: dict,
    logging_obj: Optional["LiteLLMLoggingObj"],
    extra_allowlist: set[str] | None = None,
) -> dict[str, str] | None:
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
        return _sanitize_inbound_headers(headers, extra_allowlist=extra_allowlist)

    # 2) Some guardrails pass proxy_server_request as request_data itself
    headers = request_data.get("headers")
    if headers:
        return _sanitize_inbound_headers(headers, extra_allowlist=extra_allowlist)

    # 3) Pre-call: headers stored in request metadata
    metadata_headers: Final = (request_data.get("metadata") or {}).get("headers")
    if metadata_headers:
        return _sanitize_inbound_headers(metadata_headers, extra_allowlist=extra_allowlist)

    litellm_metadata_headers: Final = (request_data.get("litellm_metadata") or {}).get("headers")
    if litellm_metadata_headers:
        return _sanitize_inbound_headers(litellm_metadata_headers, extra_allowlist=extra_allowlist)

    # 4) Post-call: headers not present on response; fallback to logging object
    if logging_obj and getattr(logging_obj, "model_call_details", None):
        try:
            details: Final = logging_obj.model_call_details or {}
            headers = details.get("litellm_params", {}).get("metadata", {}).get("headers", None)
            if headers:
                return _sanitize_inbound_headers(headers, extra_allowlist=extra_allowlist)
        except Exception:
            pass

    return None


def _resolve_call_type(
    request_data: Mapping[str, object],
    logging_obj: Optional["LiteLLMLoggingObj"],
) -> str | None:
    """Resolve the call type of the current request.

    The proxy passes the route type straight through to ``function_setup``, which
    stamps it on the logging object before the pre-call hook runs and keeps it
    for the post-call hook, so both sides of a call resolve the same value.
    """
    from_logging: Final = getattr(logging_obj, "call_type", None) if logging_obj else None
    if isinstance(from_logging, str) and from_logging:
        return from_logging
    from_request: Final = request_data.get("call_type")
    return from_request if isinstance(from_request, str) and from_request else None


def _passthrough_inputs(inputs: GenericGuardrailAPIInputs) -> GenericGuardrailAPIInputs:
    """Return the inputs untouched (same value identities), as action=NONE."""
    return GenericGuardrailAPIInputs(**inputs)


def _has_request_side_hook(event_hook: str | Sequence[str] | Mode | None) -> bool:
    """Whether the configured mode(s) include a hook that sees the request.

    Tag-based ``Mode`` config is resolved per request, so it is treated as having
    one rather than emitting a warning that may not apply.
    """
    request_side: Final = frozenset({GuardrailEventHooks.pre_call.value, GuardrailEventHooks.during_call.value})
    if event_hook is None or isinstance(event_hook, Mode):
        return True
    if isinstance(event_hook, str):
        return event_hook in request_side
    return any(hook in request_side for hook in event_hook)


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
        headers: dict[str, Any] | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        additional_provider_specific_params: dict[str, Any] | None = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        fail_on_error: bool | None = True,
        extra_headers: list | None = None,
        streaming_end_of_stream_only: bool | None = None,
        streaming_sampling_rate: int | None = None,
        streaming_transform_mode: Literal["block_only", "incremental_diff"] | None = None,
        fire_and_forget: bool | None = None,
        fire_and_forget_max_inflight: int | None = None,
        send_images: bool | None = None,
        exclude_payload_fields: Sequence[str] | None = None,
        max_messages: int | None = None,
        max_text_chars: int | None = None,
        strip_patterns: Sequence[str] | None = None,
        skip_if_system_prompt_matches: Sequence[str] | None = None,
        skip_if_first_role_in: Sequence[str] | None = None,
        skip_if_key_alias_in: Sequence[str] | None = None,
        skip_if_team_id_in: Sequence[str] | None = None,
        run_only_on_call_types: Sequence[str] | None = None,
        skip_call_types: Sequence[str] | None = None,
        guardrail_information_scope: GuardrailInformationScope | None = None,
        async_handler: AsyncHTTPHandler | None = None,
        dispatcher: BackgroundDispatcher | None = None,
        **kwargs,
    ):
        self.async_handler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.headers = headers or {}
        self.extra_headers = extra_headers or []

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

        self.additional_provider_specific_params = additional_provider_specific_params or {}

        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = unreachable_fallback

        self.fail_on_error: bool = True if fail_on_error is None else fail_on_error

        # Read by UnifiedLLMGuardrails.async_post_call_streaming_iterator_hook
        # via getattr(guardrail_to_apply, "streaming_*", default).
        self.streaming_end_of_stream_only: bool = (
            False if streaming_end_of_stream_only is None else streaming_end_of_stream_only
        )
        if streaming_sampling_rate is not None and streaming_sampling_rate < 1:
            raise ValueError(f"streaming_sampling_rate must be >= 1 (got {streaming_sampling_rate})")
        self.streaming_sampling_rate: int = 5 if streaming_sampling_rate is None else streaming_sampling_rate

        # Read by UnifiedLLMGuardrails.async_post_call_streaming_iterator_hook.
        # "block_only" (default) drops text rewrites on the streaming path;
        # "incremental_diff" emits them as synthetic deltas.
        self.streaming_transform_mode: Literal["block_only", "incremental_diff"] = (
            "block_only" if streaming_transform_mode is None else streaming_transform_mode
        )

        configured_name: Final = kwargs.get("guardrail_name")

        self.fire_and_forget: bool = False if fire_and_forget is None else fire_and_forget
        self._dispatcher: Final = dispatcher or BackgroundDispatcher(
            guardrail_name=configured_name,
            max_inflight=(
                DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT
                if fire_and_forget_max_inflight is None
                else fire_and_forget_max_inflight
            ),
        )

        self._payload_policy: Final = PayloadPolicy(
            send_images=True if send_images is None else send_images,
            exclude_fields=resolve_exclude_fields(exclude_payload_fields, guardrail_name=configured_name),
            max_messages=max_messages,
            max_text_chars=max_text_chars,
            strip_patterns=compile_patterns(strip_patterns, option_name="strip_patterns"),
        )

        self._skip_policy: Final = SkipPolicy(
            system_prompt_patterns=compile_patterns(
                skip_if_system_prompt_matches, option_name="skip_if_system_prompt_matches"
            ),
            first_role_in=frozenset(skip_if_first_role_in or ()),
            key_aliases=frozenset(skip_if_key_alias_in or ()),
            team_ids=frozenset(skip_if_team_id_in or ()),
            run_only_on_call_types=(
                validate_call_types(
                    run_only_on_call_types,
                    option_name="run_only_on_call_types",
                    guardrail_name=configured_name,
                )
                if run_only_on_call_types
                else None
            ),
            skip_call_types=validate_call_types(
                skip_call_types, option_name="skip_call_types", guardrail_name=configured_name
            ),
        )
        self._skip_store: Final = SkipDecisionStore(guardrail_name=configured_name)

        self._record_scope: Final = RecordScope(
            DEFAULT_GUARDRAIL_INFORMATION_SCOPE if guardrail_information_scope is None else guardrail_information_scope
        )

        if self.fire_and_forget:
            # Nothing awaits the response, so a block cannot be honored and the
            # stream can only be observed. Say so at boot rather than surprising
            # the operator with a guardrail that never blocks.
            verbose_proxy_logger.warning(
                "Generic Guardrail API (%s): fire_and_forget=True makes this guardrail observe-only. "
                "action=BLOCKED and action=GUARDRAIL_INTERVENED are ignored, and "
                "fail_on_error=%s / unreachable_fallback=%s cannot block the request. "
                "Streaming is forced to end-of-stream observation.",
                configured_name,
                self.fail_on_error,
                self.unreachable_fallback,
            )
            self.streaming_end_of_stream_only = True

        if self._skip_policy.filters_requests:
            verbose_proxy_logger.warning(
                "Generic Guardrail API (%s): skip_if_system_prompt_matches / skip_if_first_role_in match on the "
                "request body, which the caller controls, so a caller that knows the configured value can exempt "
                "itself from this guardrail. Use skip_if_key_alias_in / skip_if_team_id_in when the exemption must "
                "hold against the caller.",
                configured_name,
            )

        if self._skip_policy.filters_requests and not _has_request_side_hook(kwargs.get("event_hook")):
            verbose_proxy_logger.warning(
                "Generic Guardrail API (%s): skip_if_system_prompt_matches / skip_if_first_role_in need a "
                "request-side hook (pre_call or during_call) to decide anything. mode=%s only sees "
                "responses, so nothing will be skipped.",
                configured_name,
                kwargs.get("event_hook"),
            )

        if self._skip_policy.run_only_on_call_types is not None and self._skip_policy.skip_call_types:
            verbose_proxy_logger.warning(
                "Generic Guardrail API (%s): both run_only_on_call_types and skip_call_types are set. "
                "The allowlist wins; skip_call_types=%s is ignored.",
                configured_name,
                sorted(self._skip_policy.skip_call_types),
            )

        # Set supported event hooks
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))

        super().__init__(**kwargs)

        verbose_proxy_logger.debug("Generic Guardrail API initialized with api_base: %s", self.api_base)

    def _extract_user_api_key_metadata(self, request_data: dict) -> GenericGuardrailAPIMetadata:
        """
        Extract user API key metadata from request_data.

        Args:
            request_data: Request data dictionary that may contain:
                - metadata (for input requests) with user_api_key_* fields
                - litellm_metadata (for output responses) with user_api_key_* fields

        Returns:
            GenericGuardrailAPIMetadata with extracted user information
        """
        result_metadata: Final = GenericGuardrailAPIMetadata()

        # Get the source of metadata - try both locations
        # 1. For output responses: litellm_metadata (set by handlers with prefixed keys)
        # 2. For input requests: metadata (already present in request_data with prefixed keys)
        litellm_metadata: Final = request_data.get("litellm_metadata", {})
        top_level_metadata: Final = request_data.get("metadata", {})

        # Merge both sources, preferring litellm_metadata if both exist
        metadata_dict: Final = {**top_level_metadata, **litellm_metadata}

        if not metadata_dict:
            return result_metadata

        # Dynamically iterate through GenericGuardrailAPIMetadata fields
        # and extract matching fields from the source metadata
        # Fields in metadata are already prefixed with 'user_api_key_'
        for field_name in GenericGuardrailAPIMetadata.__annotations__:
            value = metadata_dict.get(field_name)
            if value is not None:
                result_metadata[field_name] = value

        # handle user_api_key_token = user_api_key_hash
        if metadata_dict.get("user_api_key_token") is not None:
            result_metadata["user_api_key_hash"] = metadata_dict.get("user_api_key_token")

        verbose_proxy_logger.debug(
            "Generic Guardrail API: Extracted user metadata: %s",
            {k: v for k, v in result_metadata.items() if v is not None},
        )

        return result_metadata

    def _fail_open_passthrough(
        self,
        *,
        inputs: GenericGuardrailAPIInputs,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"],
        error: Exception,
        http_status_code: int | None = None,
    ) -> GenericGuardrailAPIInputs:
        status_suffix: Final = f" http_status_code={http_status_code}" if http_status_code else ""
        verbose_proxy_logger.critical(
            "Generic Guardrail API error (fail-open). Proceeding without guardrail.%s "
            "guardrail_name=%s api_base=%s input_type=%s litellm_call_id=%s litellm_trace_id=%s",
            status_suffix,
            getattr(self, "guardrail_name", None),
            getattr(self, "api_base", None),
            input_type,
            getattr(logging_obj, "litellm_call_id", None) if logging_obj else None,
            getattr(logging_obj, "litellm_trace_id", None) if logging_obj else None,
            exc_info=error,
        )
        # Keep flow going - treat as action=NONE (no modifications)
        return _passthrough_inputs(inputs)

    def _build_request_headers(self) -> dict:
        """Build HTTP headers for the guardrail API request."""
        headers: Final = {"Content-Type": "application/json"}
        if self.headers:
            headers.update(self.headers)
        return headers

    def _build_guardrail_return_inputs(
        self,
        *,
        texts: list,
        images: Any,
        tools: Any,
        guardrail_response: GenericGuardrailAPIResponse,
        loss: PayloadLoss,
    ) -> GenericGuardrailAPIInputs:
        # Action is NONE or no modifications needed. A component the guardrail
        # never received in full (payload shaping) keeps the caller's original
        # value: it cannot rewrite what it could not see.
        return_inputs: Final = GenericGuardrailAPIInputs(texts=texts)
        if guardrail_response.texts:
            return_inputs["texts"] = merge_guardrailed_texts(
                original=texts,
                returned=guardrail_response.texts,
                loss=loss,
                guardrail_name=getattr(self, "guardrail_name", None),
            )
        if guardrail_response.images and not loss.images_omitted:
            return_inputs["images"] = guardrail_response.images
        elif images:
            return_inputs["images"] = images
        if guardrail_response.tools and not loss.tools_omitted:
            return_inputs["tools"] = guardrail_response.tools
        elif tools:
            return_inputs["tools"] = tools
        if guardrail_response.stream_holdback_chars is not None:
            return_inputs["stream_holdback_chars"] = guardrail_response.stream_holdback_chars
        return return_inputs

    def _handle_guardrail_request_error(
        self,
        error: Exception,
        inputs: GenericGuardrailAPIInputs,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"],
        is_unreachable: bool = True,
    ) -> GenericGuardrailAPIInputs:
        unreachable_fail_open: Final = is_unreachable and self.unreachable_fallback == "fail_open"
        if unreachable_fail_open or not self.fail_on_error:
            http_status_code: Final = getattr(getattr(error, "response", None), "status_code", None)
            return self._fail_open_passthrough(
                inputs=inputs,
                input_type=input_type,
                logging_obj=logging_obj,
                error=error,
                **({"http_status_code": http_status_code} if http_status_code else {}),
            )
        verbose_proxy_logger.error("Generic Guardrail API: failed to make request: %s", str(error))
        raise Exception(f"Generic Guardrail API failed: {error}")

    def _should_skip_out_of_scope_request(
        self,
        *,
        input_type: Literal["request", "response"],
        structured_messages: Sequence[Mapping[str, object]] | None,
        request_data: Mapping[str, object],
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> bool:
        """Whether this call is out of scope per skip_if_* (Feature 4).

        Only the request carries the system prompt, so the response side replays
        the decision the request side recorded instead of re-deciding.
        """
        if not self._skip_policy.filters_requests:
            return False

        call_id: Final = (getattr(logging_obj, "litellm_call_id", None) if logging_obj else None) or request_data.get(
            "litellm_call_id"
        )

        if input_type == "response":
            return self._skip_store.consume(logging_obj=logging_obj, call_id=call_id)

        if not request_matches_skip(self._skip_policy, structured_messages):
            return False

        self._skip_store.record(logging_obj=logging_obj, call_id=call_id)
        return True

    def _dispatch_background_post(
        self,
        *,
        payload: Mapping[str, JsonValue],
        headers: Mapping[str, str],
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> None:
        context: Final = (
            f"input_type={input_type} "
            f"litellm_call_id={getattr(logging_obj, 'litellm_call_id', None) if logging_obj else None}"
        )

        async def _post() -> None:
            response: Final = await self.async_handler.post(url=self.api_base, json=payload, headers=headers)
            response.raise_for_status()

        self._dispatcher.dispatch(_post, context=context)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """Run the guardrail, then decide whether this call records a log entry.

        The suppression flag is set only once the call has returned normally, so
        a block or a guardrail failure still records under every scope: the
        decorator's exception branch reads the same flag.
        """
        result: Final = await self._apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type=input_type,
            logging_obj=logging_obj,
        )
        if self._record_scope.should_suppress(get_session_id_from_request_data(request_data or {})):
            suppress_guardrail_information_record()
        return result

    async def _apply_guardrail(
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
        texts: Final = inputs.get("texts", [])
        images: Final = inputs.get("images")
        tools: Final = inputs.get("tools")
        structured_messages: Final = inputs.get("structured_messages")
        tool_calls: Final = inputs.get("tool_calls")
        model: Final = inputs.get("model")

        # Use provided request_data or create an empty dict
        if request_data is None:
            request_data = {}

        # Extract user API key metadata (also the basis of the identity filters below)
        user_metadata: Final = self._extract_user_api_key_metadata(request_data)

        call_type: Final = _resolve_call_type(request_data=request_data, logging_obj=logging_obj)
        if not call_type_allowed(self._skip_policy, call_type):
            verbose_proxy_logger.debug(
                "Generic Guardrail API: skipping call_type=%s (input_type=%s) per call-type filter",
                call_type,
                input_type,
            )
            return _passthrough_inputs(inputs)

        if identity_matches_skip(self._skip_policy, user_metadata):
            verbose_proxy_logger.debug(
                "Generic Guardrail API: skipping out-of-scope caller (input_type=%s, key_alias=%s, team_id=%s)",
                input_type,
                user_metadata.get("user_api_key_alias"),
                user_metadata.get("user_api_key_team_id"),
            )
            return _passthrough_inputs(inputs)

        if self._should_skip_out_of_scope_request(
            input_type=input_type,
            structured_messages=structured_messages,
            request_data=request_data,
            logging_obj=logging_obj,
        ):
            verbose_proxy_logger.debug(
                "Generic Guardrail API: skipping out-of-scope request (input_type=%s, litellm_call_id=%s)",
                input_type,
                getattr(logging_obj, "litellm_call_id", None) if logging_obj else None,
            )
            return _passthrough_inputs(inputs)

        request_body: Final = request_data.get("body") or {}

        # Merge additional provider specific params from config and dynamic params
        additional_params: Final = {**self.additional_provider_specific_params}

        # Get dynamic params from request if available
        dynamic_params: Final = self.get_guardrail_dynamic_request_body_params(request_body)
        if dynamic_params:
            additional_params.update(dynamic_params)

        extra_allowlist = {h.lower() for h in self.extra_headers if isinstance(h, str)} if self.extra_headers else None
        inbound_headers: Final = _extract_inbound_headers(
            request_data=request_data,
            logging_obj=logging_obj,
            extra_allowlist=extra_allowlist,
        )

        try:
            # Create request payload
            guardrail_request: Final = GenericGuardrailAPIRequest(
                litellm_call_id=logging_obj.litellm_call_id if logging_obj else None,
                litellm_trace_id=logging_obj.litellm_trace_id if logging_obj else None,
                texts=texts,
                request_data=user_metadata,
                request_headers=inbound_headers,
                litellm_version=litellm_version,
                images=images,
                tools=([GuardrailToolParam.model_validate(t) for t in tools] if tools else None),
                structured_messages=structured_messages,
                tool_calls=tool_calls,
                additional_provider_specific_params=additional_params,
                input_type=input_type,
                model=model,
            )

            headers: Final = self._build_request_headers()

            # Shape the payload (send_images / exclude_payload_fields / max_messages /
            # max_text_chars / strip_patterns) once, so the awaited and the
            # fire_and_forget paths send exactly the same bytes.
            # mode="json" ensures all iterables are converted to lists.
            payload, payload_loss = shape_payload(guardrail_request, self._payload_policy)

            if self.fire_and_forget:
                self._dispatch_background_post(
                    payload=payload,
                    headers=headers,
                    input_type=input_type,
                    logging_obj=logging_obj,
                )
                return _passthrough_inputs(inputs)

            # Make the API request
            response: Final = await self.async_handler.post(
                url=self.api_base,
                json=payload,
                headers=headers,
            )

            response.raise_for_status()
            response_json: Final = response.json()

            verbose_proxy_logger.debug("Generic Guardrail API response: %s", response_json)

            guardrail_response: Final = GenericGuardrailAPIResponse.from_dict(response_json)

            # Handle the response
            if guardrail_response.action == "BLOCKED":
                # Block the request
                error_message: Final = guardrail_response.blocked_reason or "Content violates policy"
                verbose_proxy_logger.warning("Generic Guardrail API blocked request: %s", error_message)
                raise GuardrailRaisedException(
                    guardrail_name=GUARDRAIL_NAME,
                    message=error_message,
                    should_wrap_with_default_message=False,
                )

            return self._build_guardrail_return_inputs(
                texts=texts,
                images=images,
                tools=tools,
                guardrail_response=guardrail_response,
                loss=payload_loss,
            )

        except GuardrailRaisedException:
            raise
        except Timeout as e:
            return self._handle_guardrail_request_error(e, inputs, input_type, logging_obj)
        except httpx.HTTPStatusError as e:
            status_code: Final = getattr(getattr(e, "response", None), "status_code", None)
            is_unreachable: Final = status_code in (502, 503, 504)
            return self._handle_guardrail_request_error(
                e, inputs, input_type, logging_obj, is_unreachable=is_unreachable
            )
        except httpx.RequestError as e:
            return self._handle_guardrail_request_error(e, inputs, input_type, logging_obj)
        except Exception as e:
            return self._handle_guardrail_request_error(e, inputs, input_type, logging_obj, is_unreachable=False)

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
            GenericGuardrailAPIConfigModel,
        )

        return GenericGuardrailAPIConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.during_call,
        ]
