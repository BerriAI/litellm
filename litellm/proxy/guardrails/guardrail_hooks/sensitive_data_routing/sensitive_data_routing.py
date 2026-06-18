"""
Sensitive Data Routing guardrail.

Detects sensitive data in a request and, instead of blocking or redacting it,
reroutes the request to an on-premise model. When sticky sessions are enabled,
every later turn in the same session is also routed on-premise so a conversation
that once touched sensitive data never leaves the on-premise model.
"""

import re
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    List,
    Optional,
    Pattern,
    Type,
    Union,
)

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

CACHE_KEY_PREFIX = "sensitive_data_routing"


class SensitiveDataRoutingGuardrail(CustomGuardrail):
    def __init__(
        self,
        on_premise_model: str,
        guardrail_name: Optional[str] = None,
        prebuilt_patterns: Optional[List[str]] = None,
        regex_patterns: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        sticky_session: bool = True,
        session_ttl_seconds: int = 14400,
        event_hook: Optional[Union[str, GuardrailEventHooks]] = None,
        default_on: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            guardrail_name=guardrail_name or "sensitive_data_routing",
            supported_event_hooks=[GuardrailEventHooks.pre_call],
            event_hook=(
                GuardrailEventHooks(event_hook)
                if event_hook is not None
                else GuardrailEventHooks.pre_call
            ),
            default_on=default_on,
            **kwargs,
        )
        self.on_premise_model = on_premise_model
        self.sticky_session = sticky_session
        self.session_ttl_seconds = session_ttl_seconds
        self._patterns = self._compile_patterns(prebuilt_patterns, regex_patterns)
        self._keywords = [k.lower() for k in (keywords or [])]
        if not self._patterns and not self._keywords:
            raise ValueError(
                "sensitive_data_routing requires at least one of prebuilt_patterns, "
                "regex_patterns, or keywords"
            )

    @staticmethod
    def _compile_patterns(
        prebuilt_patterns: Optional[List[str]],
        regex_patterns: Optional[List[str]],
    ) -> List[Pattern]:
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
            get_compiled_pattern,
        )

        compiled: List[Pattern] = [
            get_compiled_pattern(name) for name in (prebuilt_patterns or [])
        ]
        compiled.extend(
            re.compile(pattern, re.IGNORECASE) for pattern in (regex_patterns or [])
        )
        return compiled

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.sensitive_data_routing import (
            SensitiveDataRoutingConfigModel,
        )

        return SensitiveDataRoutingConfigModel

    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,
        call_type: str,
    ) -> Optional[dict]:
        session_id = self._get_session_id(data)

        session_pinned = (
            self.sticky_session
            and session_id is not None
            and await self._is_session_pinned(cache, session_id)
        )
        detected = (not session_pinned) and self._contains_sensitive_data(
            data, call_type
        )

        if not session_pinned and not detected:
            return None

        if detected and self.sticky_session and session_id is not None:
            await self._pin_session(cache, session_id)

        original_model = data.get("model")
        data["model"] = self.on_premise_model
        self._log_route(
            data=data,
            original_model=original_model,
            detected=detected,
            session_id=session_id,
        )
        return data

    def _contains_sensitive_data(self, data: dict, call_type: str) -> bool:
        messages = self.get_guardrails_messages_for_call_type(
            call_type=CallTypes(call_type), data=data
        )
        for text in self._iter_message_texts(messages):
            if any(pattern.search(text) for pattern in self._patterns):
                return True
            lowered = text.lower()
            if any(keyword in lowered for keyword in self._keywords):
                return True
        return False

    @staticmethod
    def _iter_message_texts(messages: Optional[List[Any]]) -> Iterator[str]:
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                yield content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        yield part["text"]

    @staticmethod
    def _get_session_id(data: dict) -> Optional[str]:
        session_id = data.get("litellm_session_id")
        if session_id:
            return str(session_id)
        for meta_key in ("metadata", "litellm_metadata"):
            meta = data.get(meta_key)
            if isinstance(meta, dict) and meta.get("session_id"):
                return str(meta["session_id"])
        return None

    def _session_cache_key(self, session_id: str) -> str:
        return f"{CACHE_KEY_PREFIX}:{self.guardrail_name}:{session_id}"

    async def _is_session_pinned(self, cache: "DualCache", session_id: str) -> bool:
        return bool(
            await cache.async_get_cache(key=self._session_cache_key(session_id))
        )

    async def _pin_session(self, cache: "DualCache", session_id: str) -> None:
        await cache.async_set_cache(
            key=self._session_cache_key(session_id),
            value=True,
            ttl=self.session_ttl_seconds,
        )

    def _log_route(
        self,
        data: dict,
        original_model: Optional[str],
        detected: bool,
        session_id: Optional[str],
    ) -> None:
        verbose_proxy_logger.info(
            "sensitive_data_routing: rerouting model=%s -> %s (detected=%s, session=%s)",
            original_model,
            self.on_premise_model,
            detected,
            session_id,
        )
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response={
                "action": "route",
                "on_premise_model": self.on_premise_model,
                "trigger": "detection" if detected else "sticky_session",
            },
            request_data=data,
            guardrail_status="guardrail_intervened",
            event_type=GuardrailEventHooks.pre_call,
        )
