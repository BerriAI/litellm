from json import JSONDecodeError
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Type,
    cast,
)

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.exceptions import Timeout as LiteLLMTimeout
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )


_ANALYZE_ENDPOINT = "/v1/guard/analyze"
_DEFAULT_VIGIL_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_BLOCK_REASON_MAX_CHARS = 500
_METADATA_STRING_MAX_CHARS = 500
_METADATA_ARRAY_MAX_ITEMS = 10
_VALID_DECISIONS = ("ALLOWED", "SANITIZED", "BLOCKED")
_TRANSIENT_STATUS_CODES = frozenset({429, 502, 503, 504})
_METADATA_ALLOWLIST = (
    "model",
    "model_group",
    "provider",
    "region",
    "deployment",
    "user",
    "user_id",
    "session_id",
    "conversation_id",
    "request_id",
    "tenant_id",
    "org_id",
)

_FallbackMode = Literal["fail_closed", "fail_open"]


class _AsyncPostHandler(Protocol):
    def post(
        self,
        *,
        url: str,
        headers: Dict[str, str],
        json: Dict[str, Any],
        timeout: httpx.Timeout,
    ) -> Awaitable[httpx.Response]: ...


class VigilGuardMissingConfig(ValueError):
    pass


class VigilGuardBackendError(Exception):
    pass


class VigilGuardGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        unreachable_fallback: Optional[str] = None,
        async_handler: Optional[_AsyncPostHandler] = None,
        **kwargs: Any,
    ) -> None:
        resolved_base = api_base or get_secret_str("VIGIL_GUARD_URL")
        if not resolved_base:
            raise VigilGuardMissingConfig(
                "Vigil Guard api_base is required. Set api_base in the guardrail "
                "config or the VIGIL_GUARD_URL environment variable."
            )
        self.api_base = resolved_base.rstrip("/")

        resolved_key = api_key or get_secret_str("VIGIL_GUARD_API_KEY")
        if not resolved_key:
            raise VigilGuardMissingConfig(
                "Vigil Guard api_key is required. Set api_key in the guardrail "
                "config or the VIGIL_GUARD_API_KEY environment variable."
            )
        self.api_key = resolved_key

        fallback = (unreachable_fallback or "fail_closed").lower()
        self.unreachable_fallback: _FallbackMode = (
            "fail_open" if fallback == "fail_open" else "fail_closed"
        )

        self.async_handler: _AsyncPostHandler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ]

        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.vigil_guard import (
            VigilGuardGuardrailConfigModel,
        )

        return VigilGuardGuardrailConfigModel

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        texts = inputs.get("texts") or []
        if not any(isinstance(text, str) and text.strip() for text in texts):
            return inputs

        source = "user_input" if input_type == "request" else "model_output"
        metadata = self._collect_metadata(request_data, logging_obj)

        result_texts: List[str] = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                result_texts.append(text)
                continue

            try:
                analysis = await self._analyze(
                    text=text, source=source, metadata=metadata
                )
            except (
                httpx.HTTPError,
                LiteLLMTimeout,
                JSONDecodeError,
                OSError,
            ) as exc:
                return self._handle_backend_failure(exc, inputs, source)

            decision = analysis.get("decision") if isinstance(analysis, dict) else None
            if decision not in _VALID_DECISIONS:
                verbose_proxy_logger.error(
                    "Vigil Guard unrecognized decision for guardrail_name=%s "
                    "source=%s: %r",
                    self.guardrail_name,
                    source,
                    decision,
                )
                return self._handle_backend_failure(
                    VigilGuardBackendError(
                        "Vigil Guard returned an unrecognized decision."
                    ),
                    inputs,
                    source,
                )

            if decision == "BLOCKED":
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=self._build_block_reason(analysis),
                    should_wrap_with_default_message=False,
                )

            if decision == "SANITIZED":
                result_texts.append(self._resolve_sanitized_text(text, analysis))
            else:
                result_texts.append(text)

        guardrailed: GenericGuardrailAPIInputs = {"texts": result_texts}
        if "images" in inputs:
            guardrailed["images"] = inputs["images"]
        if "tools" in inputs:
            guardrailed["tools"] = inputs["tools"]
        return guardrailed

    def _handle_backend_failure(
        self, exc: Exception, inputs: GenericGuardrailAPIInputs, source: str
    ) -> GenericGuardrailAPIInputs:
        if self.unreachable_fallback == "fail_open":
            verbose_proxy_logger.error(
                "Vigil Guard backend failure with fail_open; allowing request "
                "unscanned. guardrail_name=%s source=%s error=%s",
                self.guardrail_name,
                source,
                str(exc),
            )
            return cast(GenericGuardrailAPIInputs, dict(inputs))
        verbose_proxy_logger.error(
            "Vigil Guard backend failure with fail_closed; blocking request. "
            "guardrail_name=%s source=%s error=%s",
            self.guardrail_name,
            source,
            str(exc),
        )
        raise exc

    async def _analyze(
        self, text: str, source: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload = {
            "text": text,
            "source": source,
            "mode": "full",
            "metadata": metadata,
        }
        endpoint = f"{self.api_base}{_ANALYZE_ENDPOINT}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = await self._post_with_retry(endpoint, headers, payload)
        return response.json()

    async def _post_with_retry(
        self, endpoint: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> httpx.Response:
        for attempt in range(2):
            try:
                response = await self.async_handler.post(
                    url=endpoint,
                    headers=headers,
                    json=payload,
                    timeout=_DEFAULT_VIGIL_TIMEOUT,
                )
                response.raise_for_status()
                return response
            except Exception as exc:
                if attempt == 0 and self._is_transient(exc):
                    verbose_proxy_logger.debug(
                        "Vigil Guard transient failure; retrying once: %s",
                        type(exc).__name__,
                    )
                    continue
                raise
        raise AssertionError("unreachable")

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in _TRANSIENT_STATUS_CODES
        return isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                LiteLLMTimeout,
            ),
        )

    @staticmethod
    def _build_block_reason(analysis: Dict[str, Any]) -> str:
        for key in ("blockMessage", "decisionReason"):
            value = analysis.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:_BLOCK_REASON_MAX_CHARS]
        categories = analysis.get("categories")
        if isinstance(categories, list):
            names = [c for c in categories if isinstance(c, str) and c.strip()]
            if names:
                return ", ".join(names)[:_BLOCK_REASON_MAX_CHARS]
        return "Blocked by policy"

    @staticmethod
    def _resolve_sanitized_text(original: str, analysis: Dict[str, Any]) -> str:
        for key in ("sanitizedText", "redactedText", "outputText"):
            value = analysis.get(key)
            if isinstance(value, str):
                return value
        return original

    def _collect_metadata(
        self, request_data: dict, logging_obj: Optional["LiteLLMLoggingObj"]
    ) -> Dict[str, Any]:
        sources: List[dict] = []
        if isinstance(request_data, dict):
            sources.append(request_data)
            for nested_key in ("metadata", "litellm_metadata"):
                nested = request_data.get(nested_key)
                if isinstance(nested, dict):
                    sources.append(nested)

        collected: Dict[str, Any] = {}
        for field in _METADATA_ALLOWLIST:
            for source in sources:
                if field in source and source[field] is not None:
                    clamped = self._clamp_metadata_value(source[field])
                    if clamped is not None:
                        collected[field] = clamped
                        break

        call_id = self._extract_call_id(request_data, logging_obj)
        if call_id:
            collected["litellm_call_id"] = call_id

        return collected

    @staticmethod
    def _clamp_metadata_value(value: Any) -> Any:
        if isinstance(value, str):
            return value[:_METADATA_STRING_MAX_CHARS]
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, list):
            clamped: List[Any] = []
            for item in value[:_METADATA_ARRAY_MAX_ITEMS]:
                if isinstance(item, str):
                    clamped.append(item[:_METADATA_STRING_MAX_CHARS])
                elif isinstance(item, (int, float)):
                    clamped.append(item)
            return clamped or None
        return None

    @staticmethod
    def _extract_call_id(
        request_data: dict, logging_obj: Optional["LiteLLMLoggingObj"]
    ) -> Optional[str]:
        if logging_obj is not None:
            call_id = getattr(logging_obj, "litellm_call_id", None)
            if isinstance(call_id, str) and call_id:
                return call_id
        if isinstance(request_data, dict):
            call_id = request_data.get("litellm_call_id")
            if isinstance(call_id, str) and call_id:
                return call_id
            metadata = request_data.get("metadata")
            if isinstance(metadata, dict):
                nested = metadata.get("litellm_call_id")
                if isinstance(nested, str) and nested:
                    return nested
        return None
