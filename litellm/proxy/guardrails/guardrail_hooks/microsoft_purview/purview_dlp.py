"""Microsoft Purview DLP guardrail for LiteLLM.

Scans chat messages and model responses through the Microsoft Purview DLP
API and either blocks or logs violations depending on ``block_on_violation``.

Authentication uses the Azure AD OAuth2 client-credentials flow:
  tenant_id   – your Azure AD tenant
  client_id   – Azure AD application (client) ID (or PURVIEW_CLIENT_ID env var)
  api_key     – Azure AD client secret (the BaseLitellmParams ``api_key`` field,
                or PURVIEW_CLIENT_SECRET env var)

The ``api_base`` field (also from BaseLitellmParams) overrides the default
Purview DLP scan endpoint.

Bug fixes implemented here
--------------------------
1. ``_check_content`` except block: API/network errors are only re-raised when
   ``block_on_violation=True``.  When ``block_on_violation=False``, errors are
   logged and the method returns ``None`` instead.

2. ``async_logging_hook``: prompt audit and response audit each have their own
   ``try/except`` so that a failure during the prompt scan never skips the
   response scan.

3. ``_convert_content_list_to_str``: extracts text from ``tool_calls[].function.
   arguments`` and the legacy ``function_call.arguments`` field in addition to
   the standard ``content`` field, preventing callers from bypassing the DLP
   check by placing sensitive text in tool-call payloads.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypesLiteral, ModelResponse

if TYPE_CHECKING:
    from litellm.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME = "microsoft_purview"

_DEFAULT_API_BASE = "https://m365compliance.microsoft.com"
_TOKEN_URL_TEMPLATE = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
_DEFAULT_SCOPE = "https://m365compliance.microsoft.com/.default"


class MicrosoftPurviewDLPGuardrail(CustomGuardrail):
    """LiteLLM guardrail that scans content through Microsoft Purview DLP.

    Supports pre-call (request) and post-call (response) scanning as well as a
    logging-only audit path via ``async_logging_hook``.
    """

    def __init__(
        self,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        block_on_violation: bool = True,
        sensitive_info_types: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if "event_hook" not in kwargs:
            kwargs["event_hook"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.logging_only,
            ]

        super().__init__(**kwargs)

        self.tenant_id = tenant_id or os.getenv("PURVIEW_TENANT_ID", "")
        self.client_id = client_id or os.getenv("PURVIEW_CLIENT_ID", "")
        self.client_secret = api_key or os.getenv("PURVIEW_CLIENT_SECRET", "")
        self.api_base = (
            api_base or os.getenv("PURVIEW_API_BASE") or _DEFAULT_API_BASE
        ).rstrip("/")
        self.block_on_violation = block_on_violation
        self.sensitive_info_types = sensitive_info_types

        self._http_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        # Simple in-memory token cache: (token_str, expires_at_unix)
        self._cached_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_access_token(self) -> str:
        """Return a cached OAuth2 access token, refreshing when expired."""
        if self._cached_token and time.monotonic() < self._token_expires_at - 30:
            return self._cached_token

        if not self.tenant_id:
            raise ValueError(
                "Microsoft Purview DLP: tenant_id is required (set via config or "
                "PURVIEW_TENANT_ID environment variable)"
            )
        if not self.client_id:
            raise ValueError(
                "Microsoft Purview DLP: client_id is required (set via config or "
                "PURVIEW_CLIENT_ID environment variable)"
            )
        if not self.client_secret:
            raise ValueError(
                "Microsoft Purview DLP: client_secret / api_key is required (set via "
                "config or PURVIEW_CLIENT_SECRET environment variable)"
            )

        token_url = _TOKEN_URL_TEMPLATE.format(tenant_id=self.tenant_id)
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": _DEFAULT_SCOPE,
        }

        response = await self._http_client.post(token_url, data=payload)
        response.raise_for_status()
        token_data: Dict[str, Any] = response.json()

        self._cached_token = token_data["access_token"]
        expires_in: int = token_data.get("expires_in", 3600)
        self._token_expires_at = time.monotonic() + float(expires_in)
        return self._cached_token  # type: ignore[return-value]

    def _convert_content_list_to_str(self, message: Dict[str, Any]) -> str:
        """Extract all auditable text from a message dict.

        Covers:
        - ``content`` (str or list of content-part dicts with a ``text`` key)
        - ``tool_calls[].function.arguments``  (tool invocations forwarded to
          the LLM that may carry sensitive prompt text)
        - ``function_call.arguments``  (legacy single-function-call field)
        """
        parts: List[str] = []

        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if text:
                        parts.append(text)
        elif isinstance(content, str) and content:
            parts.append(content)

        # tool_calls[].function.arguments
        for tc in message.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            args = (tc.get("function") or {}).get("arguments") or ""
            if args:
                parts.append(args)

        # Legacy function_call.arguments
        fc = message.get("function_call")
        if isinstance(fc, dict):
            args = fc.get("arguments") or ""
            if args:
                parts.append(args)

        return "\n".join(parts)

    def _extract_messages_text(self, messages: List[Dict[str, Any]]) -> str:
        """Concatenate all auditable text from a messages list."""
        return "\n".join(self._convert_content_list_to_str(m) for m in messages).strip()

    def _extract_response_text(self, response: Any) -> str:
        """Extract text content from a model response object.

        Also includes tool-call arguments generated by the model so that
        model-produced tool invocations containing sensitive data are audited.
        """
        if not isinstance(response, ModelResponse):
            return ""

        parts: List[str] = []
        for choice in getattr(response, "choices", []):
            message = getattr(choice, "message", None)
            if message is None:
                continue

            content = getattr(message, "content", None)
            if content and isinstance(content, str):
                parts.append(content)

            # Model-generated tool calls
            for tc in getattr(message, "tool_calls", None) or []:
                args = getattr(getattr(tc, "function", None), "arguments", None) or ""
                if args:
                    parts.append(args)

            # Legacy function_call
            fc = getattr(message, "function_call", None)
            if fc:
                args = getattr(fc, "arguments", None) or ""
                if args:
                    parts.append(args)

        return "\n".join(parts).strip()

    async def _call_purview_api(self, text: str) -> Dict[str, Any]:
        """POST ``text`` to the Purview DLP scan endpoint and return the JSON."""
        token = await self._get_access_token()
        url = f"{self.api_base}/api/DLP/SensitiveInformation/Evaluate"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {"contentToClassify": text}
        if self.sensitive_info_types:
            body["sensitiveInfoTypes"] = [
                {"name": sit} for sit in self.sensitive_info_types
            ]

        verbose_proxy_logger.debug(
            "Microsoft Purview DLP: POST %s (text length=%d)", url, len(text)
        )
        response = await self._http_client.post(url, json=body, headers=headers)
        response.raise_for_status()
        result: Dict[str, Any] = response.json()
        verbose_proxy_logger.debug("Microsoft Purview DLP response: %s", result)
        return result

    @staticmethod
    def _has_violation(purview_response: Dict[str, Any]) -> bool:
        """Return True if the Purview response indicates a DLP policy match."""
        # The API returns a list of matches; any non-empty match is a violation.
        matches = purview_response.get("matches") or purview_response.get(
            "sensitiveInfoTypeMatches"
        )
        if matches:
            return True
        # Fallback: some endpoint variants use a top-level matchState field.
        return purview_response.get("matchState") == "MATCH_FOUND"

    async def _check_content(
        self,
        text: str,
        block_on_violation: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Scan *text* through Purview DLP.

        - If a policy violation is found and ``block_on_violation=True``, raises
          ``HTTPException(400)``.
        - If a policy violation is found and ``block_on_violation=False``, logs a
          warning and returns the Purview response without raising.
        - API/network errors follow the same flag:
          ``block_on_violation=True``  → re-raise.
          ``block_on_violation=False`` → log and return ``None``.
        """
        try:
            result = await self._call_purview_api(text)

            if self._has_violation(result):
                if block_on_violation:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Content blocked by Microsoft Purview DLP policy",
                            "purview_response": result,
                        },
                    )
                verbose_proxy_logger.warning(
                    "Microsoft Purview DLP: policy violation detected "
                    "(log-only mode, not blocking): %s",
                    result,
                )

            return result

        except HTTPException:
            # Intentional policy-based block — always propagate.
            raise
        except Exception as exc:
            verbose_proxy_logger.error(
                "Microsoft Purview DLP: API/network error: %s",
                str(exc),
                exc_info=True,
            )
            if block_on_violation:
                raise
            # block_on_violation=False → audit-only mode; swallow the error.
            return None

    # ------------------------------------------------------------------
    # Guardrail hooks
    # ------------------------------------------------------------------

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        """Scan the outgoing request messages before the LLM call."""
        verbose_proxy_logger.debug("Microsoft Purview DLP: pre-call hook")

        event_type = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        messages: List[Dict[str, Any]] = data.get("messages") or []
        text = self._extract_messages_text(messages)
        if not text:
            return data

        try:
            await self._check_content(text, block_on_violation=self.block_on_violation)
        except HTTPException:
            raise
        except Exception as exc:
            verbose_proxy_logger.error(
                "Microsoft Purview DLP: unexpected pre-call error: %s",
                str(exc),
                exc_info=True,
            )
            if self.block_on_violation:
                raise

        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        response: Any,
    ) -> Any:
        """Scan the model response after a successful LLM call."""
        verbose_proxy_logger.debug("Microsoft Purview DLP: post-call hook")

        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            return response

        text = self._extract_response_text(response)
        if not text:
            return response

        try:
            await self._check_content(text, block_on_violation=self.block_on_violation)
        except HTTPException:
            raise
        except Exception as exc:
            verbose_proxy_logger.error(
                "Microsoft Purview DLP: unexpected post-call error: %s",
                str(exc),
                exc_info=True,
            )
            if self.block_on_violation:
                raise

        return response

    async def async_logging_hook(
        self,
        kwargs: dict,
        result: Any,
        start_time: Any,
        end_time: Any,
    ) -> tuple:
        """Audit both the prompt and the response through Purview DLP.

        Each audit runs in its own ``try/except`` so that an API/network error
        on the prompt scan never prevents the response from being audited.
        In this hook ``block_on_violation`` is always ``False``; violations are
        logged but the hook never raises (the call has already completed).
        """
        verbose_proxy_logger.debug("Microsoft Purview DLP: logging hook")

        # --- Audit the prompt ---
        try:
            messages: List[Dict[str, Any]] = kwargs.get("messages") or []
            prompt_text = self._extract_messages_text(messages)
            if prompt_text:
                await self._check_content(prompt_text, block_on_violation=False)
        except Exception as exc:
            verbose_proxy_logger.error(
                "Microsoft Purview DLP: prompt audit error in logging hook: %s",
                str(exc),
                exc_info=True,
            )

        # --- Audit the response (runs even if prompt audit failed) ---
        try:
            response_text = self._extract_response_text(result)
            if response_text:
                await self._check_content(response_text, block_on_violation=False)
        except Exception as exc:
            verbose_proxy_logger.error(
                "Microsoft Purview DLP: response audit error in logging hook: %s",
                str(exc),
                exc_info=True,
            )

        return kwargs, result

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.microsoft_purview import (
            MicrosoftPurviewDLPConfigModel,
        )

        return MicrosoftPurviewDLPConfigModel
