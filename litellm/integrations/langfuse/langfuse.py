#### What this does ####
#    On success, logs events to Langfuse
import os
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from functools import lru_cache
from importlib.metadata import version
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, cast

from packaging.version import Version

import litellm
from litellm._logging import verbose_logger
from litellm.constants import MAX_LANGFUSE_INITIALIZED_CLIENTS
from litellm.integrations.langfuse.langfuse_mock_client import (
    create_mock_langfuse_client,
    should_use_langfuse_mock,
)
from litellm.litellm_core_utils.core_helpers import (
    filter_exceptions_from_params,
    reconstruct_model_name,
    safe_deep_copy,
)
from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    validate_langfuse_environment_value,
)
from litellm.litellm_core_utils.redact_messages import redact_user_api_key_info
from litellm.llms.custom_httpx.http_handler import _get_httpx_client
from litellm.secret_managers.main import str_to_bool
from litellm.types.integrations.langfuse import *
from litellm.types.llms.openai import HttpxBinaryResponseContent, ResponsesAPIResponse
from litellm.types.utils import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    RerankResponse,
    StandardLoggingMetadata,
    StandardLoggingPayload,
    StandardLoggingPromptManagementMetadata,
    TextCompletionResponse,
    TranscriptionResponse,
)

if TYPE_CHECKING:
    from langfuse import Langfuse
    from opentelemetry.context import Context

    from litellm.litellm_core_utils.litellm_logging import DynamicLoggingCache
else:
    Context = Any
    DynamicLoggingCache = Any
    Langfuse = Any


_DENIED_STEERING_KEYS: Final = frozenset({"headers", "endpoint", "caching_groups", "previous_models"})
_NO_METADATA: Final[Mapping[str, object]] = MappingProxyType({})
_REDACTED_PROXY_HEADERS: Final[frozenset[str]] = frozenset({"authorization", "cookie", "referer"})


def _object_mapping(value: object) -> Mapping[str, object] | None:
    """Return ``value`` as an opaque mapping when it is a dict."""
    return value if isinstance(value, dict) else None


class _UsageObject(Protocol):
    """Token-count surface the Langfuse logger reads off a response usage payload."""

    def get(self, key: Literal["cache_creation_input_tokens", "cache_read_input_tokens"], /) -> int | None: ...


def _extract_cache_read_input_tokens(usage_obj) -> int:
    """
    Extract cache_read_input_tokens from usage object.

    Checks both:
    1. Top-level cache_read_input_tokens (Anthropic format)
    2. prompt_tokens_details.cached_tokens (Gemini, OpenAI format)

    See: https://github.com/BerriAI/litellm/issues/18520

    Args:
        usage_obj: Usage object from LLM response

    Returns:
        int: Number of cached tokens read, defaults to 0
    """
    cache_read_input_tokens = usage_obj.get("cache_read_input_tokens") or 0

    # Check prompt_tokens_details.cached_tokens (used by Gemini and other providers)
    if hasattr(usage_obj, "prompt_tokens_details"):
        prompt_tokens_details: Final[object] = getattr(usage_obj, "prompt_tokens_details", None)
        if prompt_tokens_details is not None and hasattr(prompt_tokens_details, "cached_tokens"):
            cached_tokens: Final = getattr(prompt_tokens_details, "cached_tokens", None)
            if cached_tokens is not None and isinstance(cached_tokens, (int, float)) and cached_tokens > 0:
                cache_read_input_tokens = cached_tokens

    return cache_read_input_tokens


def _logging_id(start_time: datetime | None, response_obj: object) -> str | None:
    """Typed view of the timestamped response id Langfuse uses as the generation id."""
    return litellm.utils.get_logging_id(start_time, response_obj)


def _as_steering_flag(value: object) -> bool:
    """A string ``str_to_bool`` does not recognise falls back to its truthiness."""
    if isinstance(value, str):
        parsed: Final = str_to_bool(value)
        return bool(value) if parsed is None else parsed
    return bool(value)


def _as_steering_key_sequence(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(key.strip() for key in value.split(",") if key.strip())
    if isinstance(value, Iterable):
        return tuple(str(key) for key in value)
    return ()


MINIMUM_LANGFUSE_VERSION: Final = "4.7"
UNSUPPORTED_LANGFUSE_VERSION: Final = "5"


def installed_langfuse_version() -> str:
    """Only ``importlib.metadata`` reads correctly on every major.

    ``langfuse.version`` was removed in v4, ``langfuse.__version__`` does not
    exist in v3, and in v2 it reports a different value from the distribution
    that is actually installed.
    """
    return version("langfuse")


def raise_if_unsupported_langfuse_version(installed_version: str) -> None:
    """Fail at logger construction rather than dropping every event at request time.

    v4 moved the callback onto OpenTelemetry, so on an older SDK the import of
    `propagate_attributes` raises inside the per-request handler and the broad
    except there turns it into silent total data loss.
    """
    installed: Final = Version(installed_version)
    # compare majors, not versions: "5.0.0rc1" sorts below "5" but is just as unsupported
    if Version(MINIMUM_LANGFUSE_VERSION) <= installed and installed.major < Version(UNSUPPORTED_LANGFUSE_VERSION).major:
        return
    raise ImportError(
        f"\033[91mlitellm requires langfuse>={MINIMUM_LANGFUSE_VERSION},<{UNSUPPORTED_LANGFUSE_VERSION} for the "
        f"'langfuse' callback, but {installed_version} is installed. Run "
        f"'pip install \"langfuse>={MINIMUM_LANGFUSE_VERSION},<{UNSUPPORTED_LANGFUSE_VERSION}\"' to upgrade, or use "
        f"the 'langfuse_otel' callback, which does not depend on the langfuse SDK\033[0m"
    )


_PROPAGATED_TRACE_KEYS: Final = MappingProxyType(
    {"name": "trace_name", "user_id": "user_id", "session_id": "session_id", "version": "version", "tags": "tags"}
)
_GENERATION_ONLY_KEYS: Final = frozenset(
    {"id", "start_time", "end_time", "parent_observation_id", "usage", "name", "version"}
)


_PROPAGATED_VALUE_MAX_CHARS: Final = 200


def _coerce_propagated_value(value: object) -> str | Sequence[str]:
    """v4 silently drops non-string or >200-char propagated values; v2's pydantic coerced them."""
    if isinstance(value, (list, tuple)):
        return [str(item)[:_PROPAGATED_VALUE_MAX_CHARS] for item in value]
    return str(value)[:_PROPAGATED_VALUE_MAX_CHARS]


def _trace_attributes_for_propagation(trace_params: Mapping[str, object]) -> Mapping[str, object]:
    """Trace-level fields in v4 are propagated onto the observations, not set on a trace object.

    Values are coerced and capped up front: the SDK drops offenders with only a
    warning, and a dropped ``version`` would vanish from the generation too,
    because ``_generation_attributes`` already stripped it as propagated.
    """
    return MappingProxyType(
        {
            propagated: _coerce_propagated_value(trace_params[key])
            for key, propagated in _PROPAGATED_TRACE_KEYS.items()
            if trace_params.get(key) is not None
        }
    )


def _optional_str(value: object) -> str | None:
    """v4 sets attribute values raw; a non-string version would be dropped by the server."""
    return str(value) if value is not None else None


def _trace_public_flag(value: object) -> bool | None:
    """``trace_public`` reaches here as a bool from metadata or a string from a ``langfuse_*`` header."""
    if value is None:
        return None
    return _as_steering_flag(value)


def _generation_attributes(
    generation_params: Mapping[str, object], *, propagated: Mapping[str, object]
) -> Mapping[str, object]:
    """Drop what the v4 wrapper cannot take: ids it generates, and timings set on the span itself.

    ``usage`` is the v2 shape that v4 replaced with ``usage_details``, which the
    caller already builds alongside it.

    v4 has one ``version`` for a trace and its observations, so the trace's
    propagated value covers the generation; a continued trace propagates none
    and the generation keeps its own, as it did in v2.
    """
    keep_version: Final = "version" not in propagated and generation_params.get("version") is not None
    return MappingProxyType(
        {
            key: value
            for key, value in generation_params.items()
            if key not in _GENERATION_ONLY_KEYS or (key == "version" and keep_version)
        }
    )


def resolve_langfuse_credentials(
    langfuse_public_key=None,
    langfuse_secret=None,
    langfuse_secret_key=None,
    langfuse_host=None,
    allow_env_credentials: bool = True,
):
    if allow_env_credentials is False and langfuse_host is not None:
        secret_key = langfuse_secret or langfuse_secret_key
        public_key = langfuse_public_key
    else:
        secret_key = langfuse_secret or langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        public_key = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")

    resolved_host: Final = (
        langfuse_host or os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
    )

    return public_key, secret_key, resolved_host


def parse_langfuse_debug(raw_value: str | None) -> bool:
    """Parse the LANGFUSE_DEBUG value into the boolean flag the langfuse client expects."""
    return raw_value is not None and raw_value.strip().lower() in ("true", "1")


@lru_cache(maxsize=8)
def _warn_invalid_deployment_environment(raw_value: str, error: str) -> None:
    verbose_logger.warning(
        "Ignoring invalid LANGFUSE_TRACING_ENVIRONMENT=%r for the langfuse callback: %s. "
        "Traces will be sent to Langfuse's default environment.",
        raw_value,
        error,
    )


class LangFuseLogger:
    # Class variables or attributes
    def __init__(
        self,
        langfuse_public_key=None,
        langfuse_secret=None,
        langfuse_host=None,
        langfuse_environment: str | None = None,
        flush_interval=1,
        allow_env_credentials: bool = True,
    ):
        try:
            from langfuse import Langfuse
        except Exception as e:
            raise Exception(
                f"\033[91mLangfuse not installed, try running 'pip install langfuse' to fix this error: {e}\n{traceback.format_exc()}\033[0m"
            )
        self.langfuse_sdk_version: str = installed_langfuse_version()
        raise_if_unsupported_langfuse_version(self.langfuse_sdk_version)

        self.public_key, self.secret_key, self.langfuse_host = resolve_langfuse_credentials(
            langfuse_public_key=langfuse_public_key,
            langfuse_secret=langfuse_secret,
            langfuse_host=langfuse_host,
            allow_env_credentials=allow_env_credentials,
        )
        if not (self.langfuse_host.startswith("http://") or self.langfuse_host.startswith("https://")):
            # add http:// if unset, assume communicating over private network - e.g. render
            self.langfuse_host = "http://" + self.langfuse_host
        _env_override: Final = str(langfuse_environment).strip() if langfuse_environment is not None else None
        if _env_override:
            validate_langfuse_environment_value(_env_override)
            self.langfuse_environment: str | None = _env_override
        else:
            self.langfuse_environment = self.resolve_deployment_environment()
        self.langfuse_release = os.getenv("LANGFUSE_RELEASE")
        self.langfuse_debug = parse_langfuse_debug(os.getenv("LANGFUSE_DEBUG"))
        self.langfuse_flush_interval = LangFuseLogger._get_langfuse_flush_interval(flush_interval)

        if should_use_langfuse_mock():
            self.langfuse_client = create_mock_langfuse_client()
            self.is_mock_mode = True
        else:
            self._http_handler: Final = _get_httpx_client()
            self.langfuse_client = self._http_handler.client
            self.is_mock_mode = False

        parameters: Final = {
            "public_key": self.public_key,
            "secret_key": self.secret_key,
            "base_url": self.langfuse_host,
            "release": self.langfuse_release,
            "debug": self.langfuse_debug,
            "flush_interval": self.langfuse_flush_interval,  # flush interval in seconds
            "httpx_client": self.langfuse_client,
            "environment": self.langfuse_environment,
        }
        self.Langfuse: Langfuse = self.safe_init_langfuse_client(parameters)

        # set the current langfuse project id in the environ
        # this is used by Alerting to link to the correct project
        if self.is_mock_mode:
            os.environ["LANGFUSE_PROJECT_ID"] = "mock-project-id"
            verbose_logger.debug("Langfuse Mock: Using mock project ID")
        else:
            try:
                project_id: Final = self.Langfuse.api.projects.get().data[0].id
                os.environ["LANGFUSE_PROJECT_ID"] = project_id
            except Exception:
                verbose_logger.debug("Langfuse project id unavailable, alerting links will omit it")

        if os.getenv("UPSTREAM_LANGFUSE_SECRET_KEY") is not None:
            verbose_logger.warning(
                "UPSTREAM_LANGFUSE_* is no longer supported: the langfuse callback moved to SDK v4, "
                "which has no second ingestion client. The values are ignored."
            )
            self.upstream_langfuse_secret_key = os.getenv("UPSTREAM_LANGFUSE_SECRET_KEY")
            self.upstream_langfuse_public_key = os.getenv("UPSTREAM_LANGFUSE_PUBLIC_KEY")
            self.upstream_langfuse_host = os.getenv("UPSTREAM_LANGFUSE_HOST")
            self.upstream_langfuse_release = os.getenv("UPSTREAM_LANGFUSE_RELEASE")
            self.upstream_langfuse_debug = os.getenv("UPSTREAM_LANGFUSE_DEBUG")

    def safe_init_langfuse_client(self, parameters: dict) -> Langfuse:
        """
        Safely init a langfuse client if the number of initialized clients is less than the max

        Note:
            - Langfuse initializes 1 thread everytime a client is initialized.
            - We've had an incident in the past where we reached 100% cpu utilization because Langfuse was initialized several times.
        """
        if litellm.initialized_langfuse_clients >= MAX_LANGFUSE_INITIALIZED_CLIENTS:
            raise Exception(
                f"Max langfuse clients reached: {litellm.initialized_langfuse_clients} is greater than {MAX_LANGFUSE_INITIALIZED_CLIENTS}"
            )
        from litellm.integrations.langfuse.langfuse_sdk import acquire_langfuse_client

        environment_param: Final = cast(str | None, parameters.get("environment"))  # cast-ok: untyped dict
        release_param: Final = cast(str | None, parameters.get("release"))  # cast-ok: untyped dict
        langfuse_client: Final = acquire_langfuse_client(
            parameters=parameters,
            environment=environment_param,
            release=release_param,
            mock_mode=self.is_mock_mode,
        )
        litellm.initialized_langfuse_clients += 1
        verbose_logger.debug("Created langfuse client number %s", litellm.initialized_langfuse_clients)
        return langfuse_client

    @staticmethod
    def add_metadata_from_header(litellm_params: dict, metadata: dict) -> dict[str, object]:
        """
        Adds metadata from proxy request headers to Langfuse logging if keys start with "langfuse_"
        and overwrites litellm_params.metadata if already included.

        For example if you want to append your trace to an existing `trace_id` via header, send
        `headers: { ..., langfuse_existing_trace_id: your-existing-trace-id }` via proxy request.
        """
        if litellm_params is None:
            return metadata

        if litellm_params.get("proxy_server_request") is None:
            return metadata

        if metadata is None:
            metadata = {}

        proxy_headers: Final = litellm_params.get("proxy_server_request", {}).get("headers", {}) or {}

        for metadata_param_key in proxy_headers:
            if metadata_param_key.startswith("langfuse_"):
                trace_param_key = metadata_param_key.replace("langfuse_", "", 1)
                if trace_param_key in metadata:
                    verbose_logger.warning("Overwriting Langfuse `%s` from request header", trace_param_key)
                else:
                    verbose_logger.debug("Found Langfuse `%s` in request header", trace_param_key)
                metadata[trace_param_key] = proxy_headers.get(metadata_param_key)

        return metadata

    def log_event_on_langfuse(
        self,
        kwargs: dict,
        response_obj: None
        | dict
        | EmbeddingResponse
        | ModelResponse
        | TextCompletionResponse
        | ImageResponse
        | TranscriptionResponse
        | RerankResponse
        | HttpxBinaryResponseContent
        | ResponsesAPIResponse,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        level: str = "DEFAULT",
        status_message: str | None = None,
    ) -> dict:
        """
        Logs a success or error event on Langfuse
        """
        try:
            verbose_logger.debug("Langfuse Logging - Enters logging function for model %s", kwargs)

            # set default values for input/output for langfuse logging
            input = None
            output = None

            litellm_params: Final = kwargs.get("litellm_params", {})
            litellm_call_id: Final = kwargs.get("litellm_call_id", None)
            metadata = litellm_params.get("metadata", {}) or {}  # if litellm_params['metadata'] == None
            metadata = self.add_metadata_from_header(litellm_params, metadata)
            optional_params: Final = safe_deep_copy(kwargs.get("optional_params", {}))

            prompt: Final = {"messages": kwargs.get("messages")}

            functions: Final = optional_params.pop("functions", None)
            tools: Final = optional_params.pop("tools", None)
            # Remove secret_fields to prevent leaking sensitive data (e.g., authorization headers)
            optional_params.pop("secret_fields", None)
            if functions is not None:
                prompt["functions"] = functions
            if tools is not None:
                prompt["tools"] = tools

            # langfuse only accepts str, int, bool, float for logging
            for param, value in optional_params.items():
                if not isinstance(value, (str, int, bool, float)):
                    try:
                        optional_params[param] = str(value)
                    except Exception:
                        # if casting value to str fails don't block logging
                        pass

            input, output = self._get_langfuse_input_output_content(
                kwargs=kwargs,
                response_obj=response_obj,
                prompt=prompt,
                level=level,
                status_message=status_message,
            )
            verbose_logger.debug("OUTPUT IN LANGFUSE: %s; original: %s", output, response_obj)
            trace_id, generation_id = self._log_langfuse_v2(
                user_id=user_id,
                metadata=metadata,
                litellm_params=litellm_params,
                output=output,
                start_time=start_time,
                end_time=end_time,
                kwargs=kwargs,
                optional_params=optional_params,
                input=input,
                response_obj=response_obj,
                level=level,
                litellm_call_id=litellm_call_id,
            )
            verbose_logger.debug("Langfuse Layer Logging - final response object: %s", response_obj)
            verbose_logger.info("Langfuse Layer Logging - logging success")

            return {"trace_id": trace_id, "generation_id": generation_id}
        except Exception as e:
            verbose_logger.exception("Langfuse Layer Error(): Exception occured - %s", e)
            return {"trace_id": None, "generation_id": None}

    def _get_langfuse_input_output_content(
        self,
        kwargs: dict,
        response_obj: None
        | dict
        | EmbeddingResponse
        | ModelResponse
        | TextCompletionResponse
        | ImageResponse
        | TranscriptionResponse
        | RerankResponse
        | HttpxBinaryResponseContent
        | ResponsesAPIResponse,
        prompt: dict,
        level: str,
        status_message: str | None,
    ) -> tuple[dict | None, str | dict | list | None]:
        """
        Get the input and output content for Langfuse logging

        Args:
            kwargs: The keyword arguments passed to the function
            response_obj: The response object returned by the function
            prompt: The prompt used to generate the response
            level: The level of the log message
            status_message: The status message of the log message

        Returns:
            input: The input content for Langfuse logging
            output: The output content for Langfuse logging
        """
        input = None
        output: str | dict | list[Any] | None = None
        if level == "ERROR" and status_message is not None and isinstance(status_message, str):
            input = prompt
            output = status_message
        elif response_obj is not None and (
            kwargs.get("call_type", None) == "embedding" or isinstance(response_obj, litellm.EmbeddingResponse)
        ):
            input = prompt
            output = None
        elif response_obj is not None and isinstance(response_obj, litellm.ModelResponse):
            input = prompt
            output = self._get_chat_content_for_langfuse(response_obj)
        elif response_obj is not None and isinstance(response_obj, litellm.HttpxBinaryResponseContent):
            input = prompt
            output = "speech-output"
        elif response_obj is not None and isinstance(response_obj, litellm.TextCompletionResponse):
            input = prompt
            output = self._get_text_completion_content_for_langfuse(response_obj)
        elif response_obj is not None and isinstance(response_obj, litellm.ImageResponse):
            input = prompt
            output = response_obj.get("data", None)
        elif response_obj is not None and isinstance(response_obj, litellm.TranscriptionResponse):
            input = prompt
            output = response_obj.get("text", None)
        elif response_obj is not None and isinstance(response_obj, litellm.RerankResponse):
            input = prompt
            output = response_obj.results
        elif response_obj is not None and isinstance(response_obj, litellm.ResponsesAPIResponse):
            input = prompt
            output = self._get_responses_api_content_for_langfuse(response_obj)
        elif (
            kwargs.get("call_type") is not None
            and kwargs.get("call_type") == "_arealtime"
            and response_obj is not None
            and isinstance(response_obj, list)
        ):
            input = kwargs.get("input")
            output = response_obj
        elif (
            kwargs.get("call_type") is not None
            and kwargs.get("call_type") == "pass_through_endpoint"
            and response_obj is not None
            and isinstance(response_obj, dict)
        ):
            input = prompt
            output = response_obj.get("response", "")
        return input, output

    async def _async_log_event(self, kwargs, response_obj, start_time, end_time, user_id):
        """
        Langfuse SDK uses a background thread to log events

        This approach does not impact latency and runs in the background
        """

    def _log_langfuse_v2(
        self,
        user_id: str | None,
        metadata: dict[str, object],
        litellm_params: dict,
        output: str | dict | list | None,
        start_time: datetime | None,
        end_time: datetime | None,
        kwargs: dict,
        optional_params: dict,
        input: dict | None,
        response_obj,
        level: str,
        litellm_call_id: str | None,
    ) -> tuple:
        verbose_logger.debug("Langfuse Layer Logging - logging to langfuse v2")

        try:
            standard_logging_object: Final[StandardLoggingPayload | None] = cast(
                StandardLoggingPayload | None,
                kwargs.get("standard_logging_object", None),
            )
            tags = self._get_langfuse_tags(standard_logging_object=standard_logging_object)

            allowlisted_metadata: Final[StandardLoggingMetadata | Mapping[str, object]] = (
                standard_logging_object["metadata"] if standard_logging_object is not None else _NO_METADATA
            )
            end_user_id: Final = allowlisted_metadata.get("user_api_key_end_user_id", None)
            prompt_management_metadata: Final[StandardLoggingPromptManagementMetadata | None] = cast(
                StandardLoggingPromptManagementMetadata | None,
                allowlisted_metadata.get("prompt_management_metadata", None),
            )

            # Clean Metadata before logging - never log raw metadata
            # the raw metadata can contain circular references which leads to infinite recursion
            # we clean out all extra litellm metadata params before logging
            clean_metadata: dict[str, object] = {}
            if prompt_management_metadata is not None:
                clean_metadata["prompt_management_metadata"] = prompt_management_metadata
            metadata_entries: Final = _object_mapping(metadata)
            if metadata_entries is not None:
                for key, value in metadata_entries.items():
                    # generate langfuse tags - Default Tags sent to Langfuse from LiteLLM Proxy
                    if (
                        litellm.langfuse_default_tags is not None
                        and isinstance(litellm.langfuse_default_tags, list)
                        and key in litellm.langfuse_default_tags
                    ):
                        tags.append(f"{key}:{value}")

                    # clean litellm metadata before logging
                    if key in _DENIED_STEERING_KEYS:
                        continue
                    else:
                        clean_metadata[key] = value

            # Add default langfuse tags
            tags = self.add_default_langfuse_tags(tags=tags, kwargs=kwargs, metadata=metadata)

            session_id: Final = clean_metadata.pop("session_id", None)
            trace_name = cast(str | None, clean_metadata.pop("trace_name", None))
            trace_id = clean_metadata.pop("trace_id", None)
            # Use standard_logging_object.trace_id if available (when trace_id from metadata is None)
            # This allows standard trace_id to be used when provided in standard_logging_object
            if trace_id is None and standard_logging_object is not None:
                trace_id = cast(str | None, standard_logging_object.get("trace_id"))
            # Fallback to litellm_call_id if no trace_id found
            if trace_id is None:
                trace_id = kwargs.get("litellm_trace_id") or litellm_call_id
            existing_trace_id: Final = clean_metadata.pop("existing_trace_id", None)
            # If existing_trace_id is provided, use it as the trace_id to return
            # This allows continuing an existing trace while still returning the correct trace_id
            if existing_trace_id is not None:
                trace_id = existing_trace_id
            requested_trace_keys: Final = _as_steering_key_sequence(clean_metadata.pop("update_trace_keys", ()))
            update_trace_keys: Final = (
                requested_trace_keys if _as_steering_flag(litellm.langfuse_enable_update_trace_keys) else ()
            )
            debug: Final = clean_metadata.pop("debug_langfuse", None)
            mask_input: Final = _as_steering_flag(clean_metadata.pop("mask_input", False))
            mask_output: Final = _as_steering_flag(clean_metadata.pop("mask_output", False))
            # Look for masking function in the dedicated location first (set by scrub_sensitive_keys_in_metadata)
            # Fall back to metadata for backwards compatibility
            masking_function: Final = litellm_params.get("_langfuse_masking_function") or clean_metadata.pop(
                "langfuse_masking_function", None
            )

            # Apply custom masking function if provided
            masked_input: Final[object] = (
                self._apply_masking_function(input, masking_function)
                if masking_function is not None and callable(masking_function)
                else input
            )
            masked_output: Final[object] = (
                self._apply_masking_function(output, masking_function)
                if masking_function is not None and callable(masking_function)
                else output
            )

            clean_metadata = redact_user_api_key_info(metadata=clean_metadata)

            if trace_name is None and existing_trace_id is None:
                # just log `litellm-{call_type}` as the trace name
                ## DO NOT SET TRACE_NAME if trace-id set. this can lead to overwriting of past traces.
                trace_name = f"litellm-{kwargs.get('call_type', 'completion')}"

            if existing_trace_id is not None:
                trace_params: dict[str, Any] = {"id": existing_trace_id}

                # Update the following keys for this trace
                for metadata_param_key in update_trace_keys:
                    trace_param_key = metadata_param_key.replace("trace_", "")
                    if trace_param_key not in trace_params:
                        updated_trace_value = clean_metadata.pop(metadata_param_key, None)
                        if updated_trace_value is not None:
                            trace_params[trace_param_key] = updated_trace_value

                # Pop the trace specific keys that would have been popped if there were a new trace
                for key in list(filter(lambda key: key.startswith("trace_"), clean_metadata.keys())):
                    clean_metadata.pop(key, None)

                # Special keys that are found in the function arguments and not the metadata
                if "input" in update_trace_keys:
                    trace_params["input"] = masked_input if not mask_input else "redacted-by-litellm"
                if "output" in update_trace_keys:
                    trace_params["output"] = masked_output if not mask_output else "redacted-by-litellm"
            else:  # don't overwrite an existing trace
                trace_params = {
                    "id": trace_id,
                    "name": trace_name,
                    "session_id": session_id,
                    "input": masked_input if not mask_input else "redacted-by-litellm",
                    "version": clean_metadata.pop(
                        "trace_version", clean_metadata.get("version", None)
                    ),  # If provided just version, it will applied to the trace as well, if applied a trace version it will take precedence
                    "user_id": end_user_id,
                }
                for key in list(filter(lambda key: key.startswith("trace_"), clean_metadata.keys())):
                    trace_params[key.replace("trace_", "")] = clean_metadata.pop(key, None)

                if level == "ERROR":
                    trace_params["status_message"] = masked_output
                else:
                    trace_params["output"] = masked_output if not mask_output else "redacted-by-litellm"

            if debug is True or (isinstance(debug, str) and debug.lower() == "true"):
                debug_metadata: Final = {
                    key: value for key, value in metadata.items() if isinstance(value, (str, int, float, bool))
                }
                trace_params["metadata"] = {
                    **(trace_params.get("metadata") or _NO_METADATA),
                    "metadata_passed_to_litellm": debug_metadata,
                }

            cost: Final = kwargs.get("response_cost", None)
            verbose_logger.debug("trace: %s", cost)

            hidden_params: Final = standard_logging_object.get("hidden_params") if standard_logging_object else None

            if (
                litellm.langfuse_default_tags is not None
                and isinstance(litellm.langfuse_default_tags, list)
                and "proxy_base_url" in litellm.langfuse_default_tags
            ):
                proxy_base_url: Final = os.environ.get("PROXY_BASE_URL", None)
                if proxy_base_url is not None:
                    tags.append(f"proxy_base_url:{proxy_base_url}")

            api_base: Final = litellm_params.get("api_base", None)
            vertex_location: Final = kwargs.get("vertex_location", None)
            aws_region_name: Final = kwargs.get("aws_region_name", None)

            candidate_enrichments: Final = (
                ("litellm_response_cost", cost, True),
                ("hidden_params", filter_exceptions_from_params(hidden_params), hidden_params is not None),
                ("api_base", api_base, bool(api_base)),
                ("vertex_location", vertex_location, bool(vertex_location)),
                ("aws_region_name", aws_region_name, bool(aws_region_name)),
                ("cache_hit", kwargs.get("cache_hit") or False, "cache_hit" in kwargs),
            )
            enrichments: Final[Mapping[str, object]] = {
                key: value for key, value, include in candidate_enrichments if include
            }

            if "cache_hit" in kwargs and kwargs["cache_hit"] is None:
                kwargs["cache_hit"] = False  # rebind-ok: pre-existing normalization other integrations rely on
            if existing_trace_id is None:
                trace_params.update({"tags": tags})

            proxy_server_request: Final = litellm_params.get("proxy_server_request", None)
            if proxy_server_request:
                proxy_server_request.get("method", None)
                proxy_server_request.get("url", None)
                headers: Final = proxy_server_request.get("headers", None)
                clean_headers: Final = {}
                if headers:
                    for key, value in headers.items():
                        # these headers can leak our API keys and/or JWT tokens
                        if key.lower() not in _REDACTED_PROXY_HEADERS:
                            clean_headers[key] = value

            generation_id = None
            usage = None
            usage_details = None
            if response_obj is not None:
                if hasattr(response_obj, "id") and response_obj.get("id", None) is not None:
                    generation_id = _logging_id(start_time, response_obj)
                _usage_obj: Final[_UsageObject | None] = getattr(response_obj, "usage", None)

                if _usage_obj:
                    # Safely get usage values, defaulting None to 0 for Langfuse compatibility.
                    # Some providers may return null for token counts.
                    prompt_tokens: Final = getattr(_usage_obj, "prompt_tokens", None) or 0
                    completion_tokens: Final = getattr(_usage_obj, "completion_tokens", None) or 0
                    total_tokens: Final = getattr(_usage_obj, "total_tokens", None) or 0

                    cache_creation_input_tokens: Final = _usage_obj.get("cache_creation_input_tokens") or 0
                    cache_read_input_tokens: Final = _extract_cache_read_input_tokens(_usage_obj)

                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_cost": cost,
                    }
                    # According to langfuse documentation: "the input value must be reduced by the number of cache_read_input_tokens"
                    input_tokens: Final = prompt_tokens - cache_read_input_tokens
                    usage_details = LangfuseUsageDetails(
                        input=input_tokens,
                        output=completion_tokens,
                        total=total_tokens,
                        cache_creation_input_tokens=cache_creation_input_tokens,
                        cache_read_input_tokens=cache_read_input_tokens,
                    )

            generation_name = clean_metadata.pop("generation_name", None)
            if generation_name is None:
                # if `generation_name` is None, use sensible default values
                # If using litellm proxy user `key_alias` if not None
                # If `key_alias` is None, just log `litellm-{call_type}` as the generation name
                _user_api_key_alias: Final = cast(str | None, clean_metadata.get("user_api_key_alias", None))
                generation_name = f"litellm-{cast(str, kwargs.get('call_type', 'completion'))}"
                if _user_api_key_alias is not None:
                    generation_name = f"litellm:{_user_api_key_alias}"

            if response_obj is not None:
                system_fingerprint = getattr(response_obj, "system_fingerprint", None)
            else:
                system_fingerprint = None

            if system_fingerprint is not None:
                optional_params["system_fingerprint"] = system_fingerprint

            custom_llm_provider: Final = cast(str | None, kwargs.get("custom_llm_provider"))
            model_name: Final = reconstruct_model_name(kwargs.get("model", ""), custom_llm_provider, metadata)

            generation_params = {
                "name": generation_name,
                "id": clean_metadata.pop("generation_id", generation_id),
                "start_time": start_time,
                "end_time": end_time,
                "model": model_name,
                "model_parameters": optional_params,
                "input": masked_input if not mask_input else "redacted-by-litellm",
                "output": masked_output if not mask_output else "redacted-by-litellm",
                "usage": usage,
                "usage_details": usage_details,
                "cost_details": {"total": cost}  # mutable-ok: langfuse serializes this payload
                if usage is not None and isinstance(cost, (int, float))
                else None,
                "metadata": {  # mutable-ok: langfuse serializes this payload, a proxy is not json-encodable
                    **(trace_params.get("metadata") or {}),
                    **log_requester_metadata(redact_user_api_key_info(metadata=allowlisted_metadata)),  # pyright: ignore[reportArgumentType]  # TypedDict in, plain metadata dict out
                    **enrichments,
                },
                "level": level,
                "version": _optional_str(clean_metadata.pop("version", None)),
            }

            parent_observation_id: Final = metadata.get("parent_observation_id", None)
            if parent_observation_id is not None:
                generation_params["parent_observation_id"] = parent_observation_id

            generation_params = _add_prompt_to_generation_params(
                generation_params=generation_params,
                clean_metadata=clean_metadata,
                prompt_management_metadata=prompt_management_metadata,
                langfuse_client=self.Langfuse,
            )
            if masked_output is not None and isinstance(masked_output, str) and level == "ERROR":
                generation_params["status_message"] = masked_output

            generation_params["completion_start_time"] = kwargs.get("completion_start_time", None)

            # langfuse ships in the proxy-runtime extra, so this module must import cleanly without it
            from litellm.integrations.langfuse.langfuse_sdk import (
                open_trace_context,
                propagate_attributes,
                resolve_observation_id,
                resolve_trace_id,
                start_generation,
                to_unix_nanos,
            )

            resolved_trace_id: Final = resolve_trace_id(trace_id)  # pyright: ignore[reportArgumentType]  # metadata value, str or None at runtime

            propagated_trace_attributes: Final = _trace_attributes_for_propagation(trace_params)
            with propagate_attributes(**propagated_trace_attributes):  # pyright: ignore[reportArgumentType]  # kwargs-ok: keys fixed by _PROPAGATED_TRACE_KEYS, values are the SDK's own trace fields
                trace_context, claim_trace_root = open_trace_context(
                    client=self.Langfuse,
                    trace_id=resolved_trace_id,
                    parent_observation_id=resolve_observation_id(parent_observation_id),  # pyright: ignore[reportArgumentType]  # metadata value, str or None at runtime
                )
                log_provider_specific_information_as_span(
                    client=self.Langfuse,
                    context=trace_context,
                    enrichments=enrichments,
                    claim_trace_root=claim_trace_root,
                )
                self._log_guardrail_information_as_span(
                    client=self.Langfuse,
                    context=trace_context,
                    standard_logging_object=standard_logging_object,
                    claim_trace_root=claim_trace_root,
                )
                generation: Final = start_generation(
                    client=self.Langfuse,
                    context=trace_context,
                    name=generation_params["name"],  # pyright: ignore[reportArgumentType]  # always the str set a few lines up
                    start_time=start_time,
                    claim_trace_root=claim_trace_root,
                    release=trace_params.get("release"),
                    public=_trace_public_flag(trace_params.get("public")),
                    attributes=_generation_attributes(generation_params, propagated=propagated_trace_attributes),
                )
                if existing_trace_id is not None and ("input" in update_trace_keys or "output" in update_trace_keys):
                    # with a real parent the generation is not the trace root, so trace-level
                    # I/O has to be stamped explicitly; v2 updated the trace object directly
                    generation.set_trace_io(  # pyright: ignore[reportDeprecated]  # the SDK keeps it exactly for this legacy trace-level contract
                        input=trace_params.get("input") if "input" in update_trace_keys else None,
                        output=trace_params.get("output") if "output" in update_trace_keys else None,
                    )
                generation.end(end_time=to_unix_nanos(end_time))

            # log_event_on_langfuse tuple-unpacks this and re-wraps it in the dict callers cache.
            # The wrapper's id is the exported observation id; the pre-computed generation_id would
            # name nothing in langfuse, because v4 derives observation ids from the OTel span.
            return resolved_trace_id, generation.id
        except Exception:
            verbose_logger.error("Langfuse Layer Error - %s", traceback.format_exc())
            return None, None

    @staticmethod
    def _get_chat_content_for_langfuse(
        response_obj: ModelResponse,
    ) -> str | None:
        """
        Get the chat content for Langfuse logging
        """
        if response_obj.choices and len(response_obj.choices) > 0:
            output: Final = response_obj["choices"][0]["message"].json()
            return output
        else:
            return None

    @staticmethod
    def _get_text_completion_content_for_langfuse(
        response_obj: TextCompletionResponse,
    ):
        """
        Get the text completion content for Langfuse logging
        """
        if response_obj.choices and len(response_obj.choices) > 0:
            return response_obj.choices[0].text
        else:
            return None

    @staticmethod
    def _get_responses_api_content_for_langfuse(
        response_obj: ResponsesAPIResponse,
    ):
        """
        Get the responses API content for Langfuse logging
        """
        if hasattr(response_obj, "output") and response_obj.output:
            # ResponsesAPIResponse.output is a list of strings
            return response_obj.output
        else:
            return None

    @staticmethod
    def _get_langfuse_tags(
        standard_logging_object: StandardLoggingPayload | None,
    ) -> list[str]:
        if standard_logging_object is None:
            return []
        return standard_logging_object.get("request_tags", []) or []

    def add_default_langfuse_tags(self, tags, kwargs, metadata):
        """
        Helper function to add litellm default langfuse tags

        - Special LiteLLM tags:
            - cache_hit
            - cache_key

        """
        if litellm.langfuse_default_tags is not None and isinstance(litellm.langfuse_default_tags, list):
            if "cache_hit" in litellm.langfuse_default_tags:
                _cache_hit_value: Final = kwargs.get("cache_hit", False)
                tags.append(f"cache_hit:{_cache_hit_value}")
            if "cache_key" in litellm.langfuse_default_tags:
                _hidden_params: Final = metadata.get("hidden_params", {}) or {}
                _cache_key = _hidden_params.get("cache_key", None)
                if _cache_key is None and litellm.cache is not None:
                    # fallback to using "preset_cache_key"
                    _preset_cache_key: Final = litellm.cache._get_preset_cache_key_from_kwargs(**kwargs)  # pyright: ignore[reportPrivateUsage]  # kwargs-ok: no public preset-cache-key accessor
                    _cache_key = _preset_cache_key
                tags.append(f"cache_key:{_cache_key}")
        return tags

    @staticmethod
    def _apply_masking_function(data: object, masking_function: Callable[[object], object]) -> object:
        """
        Apply a masking function to data, handling different data types.

        Args:
            data: The data to mask (can be str, dict, list, or None)
            masking_function: A callable that takes data and returns masked data

        Returns:
            The masked data
        """
        if data is None:
            return None

        try:
            if isinstance(data, str):
                return masking_function(data)
            elif isinstance(data, dict):
                masked_dict: Final = {}
                for key, value in data.items():
                    masked_dict[key] = LangFuseLogger._apply_masking_function(value, masking_function)
                return masked_dict
            elif isinstance(data, list):
                return [LangFuseLogger._apply_masking_function(item, masking_function) for item in data]
            else:
                # For other types, try to apply the function directly
                return masking_function(data)
        except Exception as e:
            verbose_logger.warning("Failed to apply masking function: %s. Returning original data.", e)
            return data

    @staticmethod
    def resolve_deployment_environment() -> str | None:
        """Resolve LANGFUSE_TRACING_ENVIRONMENT: stripped value, "default" plus a warning when invalid, None when unset."""
        raw: Final = os.getenv("LANGFUSE_TRACING_ENVIRONMENT")
        if not raw:
            return None
        value: Final = raw.strip()
        try:
            validate_langfuse_environment_value(value)
        except ValueError as e:
            _warn_invalid_deployment_environment(raw, str(e))
            return "default"
        return value

    @staticmethod
    def _get_langfuse_flush_interval(flush_interval: int) -> int:
        """
        Get the langfuse flush interval to initialize the Langfuse client

        Reads `LANGFUSE_FLUSH_INTERVAL` from the environment variable.
        If not set, uses the flush interval passed in as an argument.

        Args:
            flush_interval: The flush interval to use if LANGFUSE_FLUSH_INTERVAL is not set

        Returns:
            [int] The flush interval to use to initialize the Langfuse client
        """
        return int(os.getenv("LANGFUSE_FLUSH_INTERVAL") or flush_interval)

    def _log_guardrail_information_as_span(
        self,
        client: "Langfuse",
        context: "Context",
        standard_logging_object: StandardLoggingPayload | None,
        claim_trace_root: bool,
    ):
        """
        Log guardrail information as a span
        """
        if standard_logging_object is None:
            verbose_logger.debug("Not logging guardrail information as span because standard_logging_object is None")
            return

        guardrail_information: Final = standard_logging_object.get("guardrail_information", None)
        if not guardrail_information:
            verbose_logger.debug("Not logging guardrail information as span because guardrail_information is empty")
            return

        if not isinstance(guardrail_information, list):
            verbose_logger.debug(
                "Not logging guardrail information as span because guardrail_information is not a list: %s",
                type(guardrail_information),
            )
            return

        from litellm.integrations.langfuse.langfuse_sdk import start_child_span, to_unix_nanos

        for guardrail_entry in guardrail_information:
            if not isinstance(guardrail_entry, dict):
                verbose_logger.debug(
                    "Skipping guardrail entry with unexpected type: %s",
                    type(guardrail_entry),
                )
                continue

            span = start_child_span(
                client=client,
                context=context,
                name="guardrail",
                start_time=guardrail_entry.get("start_time", None),
                claim_trace_root=claim_trace_root,
                attributes={  # mutable-ok: langfuse serializes this payload, a proxy is not json-encodable
                    "input": guardrail_entry.get("guardrail_request", None),
                    "output": guardrail_entry.get("guardrail_response", None),
                    "metadata": {
                        "guardrail_name": guardrail_entry.get("guardrail_name", None),
                        "guardrail_mode": guardrail_entry.get("guardrail_mode", None),
                        "guardrail_masked_entity_count": guardrail_entry.get("masked_entity_count", None),
                    },
                },
            )

            verbose_logger.debug("Logged guardrail information as span: %s", span)
            span.end(end_time=to_unix_nanos(guardrail_entry.get("end_time", None)))


def _add_prompt_to_generation_params(
    generation_params: dict,
    clean_metadata: dict,
    prompt_management_metadata: StandardLoggingPromptManagementMetadata | None,
    langfuse_client: object,
) -> dict:
    from langfuse import Langfuse
    from langfuse.model import (
        ChatPromptClient,
        Prompt_Chat,
        Prompt_Text,
        TextPromptClient,
    )

    langfuse_client = cast(Langfuse, langfuse_client)

    user_prompt: Final = clean_metadata.pop("prompt", None)
    if user_prompt is None and prompt_management_metadata is None:
        pass
    elif isinstance(user_prompt, dict):
        if user_prompt.get("type", "") == "chat":
            _prompt_chat: Final = Prompt_Chat(**user_prompt)
            generation_params["prompt"] = ChatPromptClient(prompt=_prompt_chat)
        elif user_prompt.get("type", "") == "text":
            _prompt_text: Final = Prompt_Text(**user_prompt)
            generation_params["prompt"] = TextPromptClient(prompt=_prompt_text)
        elif "version" in user_prompt and "prompt" in user_prompt:
            # prompts
            if isinstance(user_prompt["prompt"], str):
                prompt_text_params: Final = getattr(Prompt_Text, "model_fields", Prompt_Text.__fields__)
                _data = {
                    "name": user_prompt["name"],
                    "prompt": user_prompt["prompt"],
                    "version": user_prompt["version"],
                    "config": user_prompt.get("config", None),
                }
                if "labels" in prompt_text_params and "tags" in prompt_text_params:
                    _data["labels"] = user_prompt.get("labels", []) or []
                    _data["tags"] = user_prompt.get("tags", []) or []
                _prompt_obj = Prompt_Text(**_data)  # pyright: ignore[reportArgumentType]  # kwargs-ok: shape mirrors the pydantic model, values from the user's prompt dict
                generation_params["prompt"] = TextPromptClient(prompt=_prompt_obj)

            elif isinstance(user_prompt["prompt"], list):
                prompt_chat_params: Final = getattr(Prompt_Chat, "model_fields", Prompt_Chat.__fields__)
                _data = {
                    "name": user_prompt["name"],
                    "prompt": user_prompt["prompt"],
                    "version": user_prompt["version"],
                    "config": user_prompt.get("config", None),
                }
                if "labels" in prompt_chat_params and "tags" in prompt_chat_params:
                    _data["labels"] = user_prompt.get("labels", []) or []
                    _data["tags"] = user_prompt.get("tags", []) or []

                _prompt_obj = Prompt_Chat(**_data)  # pyright: ignore[reportArgumentType]  # kwargs-ok: shape mirrors the pydantic model, values from the user's prompt dict

                generation_params["prompt"] = ChatPromptClient(prompt=_prompt_obj)
            else:
                verbose_logger.error("[Non-blocking] Langfuse Logger: Invalid prompt format")
        else:
            verbose_logger.error("[Non-blocking] Langfuse Logger: Invalid prompt format. No prompt logged to Langfuse")
    elif prompt_management_metadata is not None and prompt_management_metadata["prompt_integration"] == "langfuse":
        try:
            generation_params["prompt"] = langfuse_client.get_prompt(prompt_management_metadata["prompt_id"])
        except Exception as e:
            verbose_logger.debug("[Non-blocking] Langfuse Logger: Error getting prompt client for logging: %s", e)

    else:
        generation_params["prompt"] = user_prompt

    return generation_params


def log_provider_specific_information_as_span(
    *,
    client: "Langfuse",
    context: "Context",
    enrichments: Mapping[str, Any],
    claim_trace_root: bool,
):
    """
    Logs provider-specific information as spans.

    Parameters:
        trace: The tracing object used to log spans.
        enrichments: The litellm-computed fields on the emitted payload.

    Returns:
        None
    """

    _hidden_params: Final[Mapping[str, object] | None] = enrichments.get("hidden_params", None)
    if _hidden_params is None:
        return

    vertex_ai_grounding_metadata: Final = _hidden_params.get("vertex_ai_grounding_metadata", None)

    if vertex_ai_grounding_metadata is not None:
        if isinstance(vertex_ai_grounding_metadata, list):
            for elem in vertex_ai_grounding_metadata:
                if isinstance(elem, dict):
                    for key, value in elem.items():
                        _end_grounding_span(
                            client=client, context=context, name=key, value=value, claim_trace_root=claim_trace_root
                        )
                else:
                    _end_grounding_span(
                        client=client,
                        context=context,
                        name="vertex_ai_grounding_metadata",
                        value=elem,
                        claim_trace_root=claim_trace_root,
                    )
        else:
            _end_grounding_span(
                client=client,
                context=context,
                name="vertex_ai_grounding_metadata",
                value=vertex_ai_grounding_metadata,
                claim_trace_root=claim_trace_root,
            )


def _end_grounding_span(
    *, client: "Langfuse", context: "Context", name: str, value: object, claim_trace_root: bool
) -> None:
    from litellm.integrations.langfuse.langfuse_sdk import start_child_span

    start_child_span(
        client=client,
        context=context,
        name=name,
        start_time=None,
        claim_trace_root=claim_trace_root,
        attributes={"input": value},  # mutable-ok: langfuse serializes this payload
    ).end()


def log_requester_metadata(clean_metadata: Mapping[str, Any]):
    returned_metadata: Final = {}
    requester_metadata: Final = clean_metadata.get("requester_metadata") or {}
    for k, v in clean_metadata.items():
        if k not in requester_metadata:
            returned_metadata[k] = v

    returned_metadata.update({"requester_metadata": requester_metadata})

    return returned_metadata
