"""Highflame guardrail integration for LiteLLM.

Calls Highflame Shield's ``POST /v1/shield/guard`` endpoint. Authentication
uses a service key (``HIGHFLAME_API_KEY``) exchanged for a short-lived JWT at
the AuthN token endpoint; the JWT is cached and refreshed automatically.

Docs: https://docs.highflame.ai
"""

import asyncio
import time
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.highflame import (
    HIGHFLAME_CAPABILITY_MAP,
    HighflameGuardRequest,
    HighflameGuardResponse,
)
from litellm.types.utils import CallTypesLiteral, GuardrailStatus

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

DEFAULT_API_BASE = "https://api.highflame.ai"
DEFAULT_TOKEN_URL = "https://auth.highflame.ai/oauth2/token"
# Refresh the JWT this many seconds before it actually expires.
_TOKEN_REFRESH_BUFFER = 60


class HighflameGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        token_url: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        application: Optional[str] = None,
        shield_mode: str = "enforce",
        default_on: bool = True,
        guardrail_name: str = "highflame",
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        """Initialize the Highflame guardrail.

        Calls: ``{api_base}/v1/shield/guard`` (default api_base
        ``https://api.highflame.ai``).

        Args:
            api_key: Highflame service key (``hf_sk_...``). Falls back to the
                ``HIGHFLAME_API_KEY`` secret.
            api_base: Shield host. Falls back to ``HIGHFLAME_API_BASE`` then
                ``https://api.highflame.ai``.
            token_url: AuthN token-exchange URL. Falls back to
                ``HIGHFLAME_TOKEN_URL`` then ``https://auth.highflame.ai/oauth2/token``.
            capabilities: OWASP-aligned capability names (see
                ``HIGHFLAME_CAPABILITY_MAP``). Empty = all enabled in policy.
            application: Highflame application name for policy-scoped guards.
            shield_mode: Shield mode — enforce | monitor | alert | modify.
        """
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.highflame_api_key = api_key or get_secret_str("HIGHFLAME_API_KEY")
        self.api_base = (
            api_base or get_secret_str("HIGHFLAME_API_BASE") or DEFAULT_API_BASE
        ).rstrip("/")
        self.token_url = (
            token_url or get_secret_str("HIGHFLAME_TOKEN_URL") or DEFAULT_TOKEN_URL
        )
        self.capabilities = capabilities or []
        self.application = application
        self.shield_mode = shield_mode or "enforce"
        self.metadata = metadata
        self.default_on = default_on

        # JWT cache (service key -> bearer token).
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()

        verbose_proxy_logger.debug(
            "Highflame Guardrail: initialized guardrail_name=%s api_base=%s "
            "capabilities=%s application=%s mode=%s",
            guardrail_name,
            self.api_base,
            self.capabilities,
            self.application,
            self.shield_mode,
        )

        super().__init__(guardrail_name=guardrail_name, default_on=default_on, **kwargs)

    def _resolve_detectors(self) -> List[str]:
        """Map configured OWASP capability aliases to Shield detector IDs."""
        detectors: List[str] = []
        for capability in self.capabilities:
            mapped = HIGHFLAME_CAPABILITY_MAP.get(capability)
            if mapped is None:
                verbose_proxy_logger.warning(
                    "Highflame Guardrail: unknown capability '%s' ignored. "
                    "Known: %s",
                    capability,
                    ", ".join(sorted(HIGHFLAME_CAPABILITY_MAP)),
                )
                continue
            detectors.extend(mapped)
        # De-duplicate while preserving order.
        seen: set = set()
        ordered: List[str] = []
        for detector in detectors:
            if detector not in seen:
                seen.add(detector)
                ordered.append(detector)
        return ordered

    async def _get_token(self) -> str:
        """Return a cached JWT, exchanging the service key when needed."""
        if self.highflame_api_key is None:
            raise ValueError(
                "HighflameGuardrailException - no API key. Set the 'api_key' "
                "litellm_param or the HIGHFLAME_API_KEY environment variable."
            )
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token
        async with self._token_lock:
            # Re-check inside the lock — another coroutine may have refreshed.
            if self._access_token and time.time() < self._token_expires_at:
                return self._access_token
            response = await self.async_handler.post(
                url=self.token_url,
                json={"grant_type": "api_key", "api_key": self.highflame_api_key},
            )
            response.raise_for_status()
            token_data = response.json()
            self._access_token = token_data["access_token"]
            self._token_expires_at = (
                time.time()
                + int(token_data.get("expires_in", 3600))
                - _TOKEN_REFRESH_BUFFER
            )
            return self._access_token

    async def call_highflame_guard(
        self,
        content: str,
        content_type: str,
        action: str,
        event_type: GuardrailEventHooks,
    ) -> HighflameGuardResponse:
        """Call Shield's ``POST /v1/shield/guard``.

        Fails open (returns ``{"decision": "allow"}``) on transport / auth
        errors so a Shield outage does not take down the proxy; the failure is
        logged for observability.
        """
        start_time = datetime.now()
        status: GuardrailStatus = "guardrail_failed_to_respond"
        guard_response: Optional[HighflameGuardResponse] = None
        exception_str = ""

        request_body: HighflameGuardRequest = {
            "content": content,
            "content_type": content_type,
            "action": action,
            "mode": self.shield_mode,
        }
        detectors = self._resolve_detectors()
        if detectors:
            request_body["detectors"] = detectors
        if self.application:
            request_body["application"] = self.application
        if self.metadata:
            request_body["metadata"] = {
                k: v
                for k, v in self.metadata.items()
                if k != "standard_logging_guardrail_information"
            }

        try:
            token = await self._get_token()
            url = f"{self.api_base}/v1/shield/guard"
            verbose_proxy_logger.debug("Highflame Guardrail: POST %s", url)
            response = await self.async_handler.post(
                url=url,
                headers={"Authorization": f"Bearer {token}"},
                json=dict(request_body),
            )
            response.raise_for_status()
            guard_response = response.json()
            status = "success"
            return guard_response
        except Exception as e:  # noqa: BLE001 — fail open, log below
            exception_str = str(e)
            verbose_proxy_logger.warning(
                "Highflame Guardrail: guard call failed, failing open: %s",
                exception_str,
            )
            return {"decision": "allow"}
        finally:
            guardrail_json_response: Union[Exception, str, dict, List[dict]] = (
                dict(guard_response)
                if status == "success" and guard_response is not None
                else exception_str
            )
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response=guardrail_json_response,
                request_data={
                    "content": content,
                    "content_type": content_type,
                    "metadata": self.metadata or {},
                },
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
                event_type=event_type,
            )

    def _raise_if_denied(self, guard_response: HighflameGuardResponse) -> None:
        """Raise HTTP 400 when Shield returns a deny decision."""
        decision = (guard_response or {}).get("decision", "allow")
        if decision != "deny":
            return
        policy_reason = guard_response.get("policy_reason") or (
            f"Request blocked by Highflame guardrails ({self.guardrail_name})."
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Violated guardrail policy",
                "highflame_guardrail_response": guard_response,
                "policy_reason": policy_reason,
                "signals": guard_response.get("signals", []),
            },
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: litellm.DualCache,
        data: Dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, Dict]]:
        """Evaluate the user prompt before the LLM call."""
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_last_user_message,
        )
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data
        if "messages" not in data:
            return data
        text = get_last_user_message(data["messages"])
        if text is None:
            return data

        guard_response = await self.call_highflame_guard(
            content=text,
            content_type="prompt",
            action="process_prompt",
            event_type=event_type,
        )
        self._raise_if_denied(guard_response)
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return data

    async def async_post_call_success_hook(
        self,
        data: Dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """Evaluate the LLM response after a successful call."""
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return response

        text = self._extract_response_text(response)
        if not text:
            return response

        guard_response = await self.call_highflame_guard(
            content=text,
            content_type="response",
            action="process_response",
            event_type=event_type,
        )
        self._raise_if_denied(guard_response)
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return response

    @staticmethod
    def _extract_response_text(response) -> Optional[str]:
        """Best-effort extraction of assistant text from a litellm response."""
        try:
            choices = getattr(response, "choices", None)
            if not choices:
                return None
            parts: List[str] = []
            for choice in choices:
                message = getattr(choice, "message", None)
                content = getattr(message, "content", None) if message else None
                if isinstance(content, str) and content:
                    parts.append(content)
            return "\n".join(parts) if parts else None
        except Exception:  # noqa: BLE001 — never break the response path
            return None

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.highflame import (
            HighflameGuardrailConfigModel,
        )

        return HighflameGuardrailConfigModel
