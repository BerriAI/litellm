"""
Microsoft Purview DLP Guardrail for LiteLLM.

Supports three modes:
- pre_call:      Block sensitive data in prompts before they reach the LLM.
- post_call:     Block sensitive data in LLM responses.
- logging_only:  Log interactions to Purview for audit/compliance without blocking.
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks

from .base import PurviewGuardrailBase

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )
    from litellm.types.utils import (
        CallTypesLiteral,
        EmbeddingResponse,
        ImageResponse,
        ModelResponse,
    )


class MicrosoftPurviewDLPGuardrail(PurviewGuardrailBase, CustomGuardrail):
    """
    Microsoft Purview DLP guardrail.

    Evaluates prompts and responses against Microsoft Purview DLP policies
    via the Microsoft Graph ``processContent`` API.
    """

    def __init__(
        self,
        guardrail_name: str,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        purview_app_name: str = "LiteLLM",
        user_id_field: str = "user_id",
        logging_only: bool = False,
        **kwargs: Any,
    ):
        if logging_only:
            kwargs["event_hook"] = GuardrailEventHooks.logging_only

        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
        ]

        super().__init__(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            purview_app_name=purview_app_name,
            user_id_field=user_id_field,
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            **kwargs,
        )
        self._logging_only = logging_only
        self.guardrail_provider = "microsoft_purview"
        self._executor = ThreadPoolExecutor(max_workers=1)
        verbose_proxy_logger.info(
            "Initialized Microsoft Purview DLP Guardrail: %s (logging_only=%s)",
            guardrail_name,
            logging_only,
        )

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        return None  # Config model can be added later for UI support

    # ------------------------------------------------------------------
    # Core DLP check
    # ------------------------------------------------------------------

    async def _check_content(
        self,
        user_id: str,
        text: str,
        activity: str,
        request_data: Dict[str, Any],
        block_on_violation: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate content against Purview DLP policies.

        Args:
            user_id: Entra object ID.
            text: Content to evaluate.
            activity: ``"uploadText"`` or ``"downloadText"``.
            request_data: Original request dict (used for logging metadata).
            block_on_violation: If False, log only — do not raise.

        Returns:
            The processContent response dict.
        """
        start_time = datetime.now()
        status = "success"
        response: Dict[str, Any] = {}

        try:
            etag, _ = await self._compute_protection_scopes(user_id)
            correlation_id = request_data.get("litellm_call_id") or str(uuid.uuid4())
            response = await self._process_content(
                user_id=user_id,
                text=text,
                activity=activity,
                etag=etag,
                correlation_id=correlation_id,
            )

            if self._should_block(response):
                status = "guardrail_intervened"
        except Exception:
            status = "guardrail_failed_to_respond"
            raise
        finally:
            end_time = datetime.now()
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider=self.guardrail_provider,
                guardrail_json_response=response,
                request_data=request_data,
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                duration=(end_time - start_time).total_seconds(),
            )

        if block_on_violation and status == "guardrail_intervened":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Microsoft Purview DLP: Content blocked by policy",
                    "activity": activity,
                },
            )

        return response

    # ------------------------------------------------------------------
    # Pre-call hook — DLP on prompts
    # ------------------------------------------------------------------

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: Any,
        data: Dict[str, Any],
        call_type: "CallTypesLiteral",
    ) -> Optional[Dict[str, Any]]:
        """Check user prompt against Purview DLP policies before LLM call."""
        user_id = self._resolve_user_id(data, user_api_key_dict)
        if not user_id:
            verbose_proxy_logger.warning(
                "Purview DLP: No user_id found, skipping pre-call check"
            )
            return data

        messages: Optional[List] = data.get("messages")
        if not messages:
            return data

        user_prompt = self.get_user_prompt(messages)
        if user_prompt:
            await self._check_content(
                user_id=user_id,
                text=user_prompt,
                activity="uploadText",
                request_data=data,
                block_on_violation=True,
            )
        return None

    # ------------------------------------------------------------------
    # Post-call hook — DLP on responses
    # ------------------------------------------------------------------

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        response: Union[Any, "ModelResponse", "EmbeddingResponse", "ImageResponse"],
    ) -> Any:
        """Check LLM response against Purview DLP policies."""
        from litellm.types.utils import Choices, ModelResponse

        user_id = self._resolve_user_id(data, user_api_key_dict)
        if not user_id:
            verbose_proxy_logger.warning(
                "Purview DLP: No user_id found, skipping post-call check"
            )
            return response

        if (
            isinstance(response, ModelResponse)
            and response.choices
            and isinstance(response.choices[0], Choices)
        ):
            content = response.choices[0].message.content or ""
            if content:
                await self._check_content(
                    user_id=user_id,
                    text=content,
                    activity="downloadText",
                    request_data=data,
                    block_on_violation=True,
                )
        return response

    # ------------------------------------------------------------------
    # Logging-only hook — audit without blocking
    # ------------------------------------------------------------------

    def logging_hook(
        self, kwargs: dict, result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Sync wrapper for async_logging_hook (follows Presidio pattern)."""

        def run_in_new_loop() -> Tuple[dict, Any]:
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(
                    self.async_logging_hook(
                        kwargs=kwargs, result=result, call_type=call_type
                    )
                )
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

        try:
            _ = asyncio.get_running_loop()
            future = self._executor.submit(run_in_new_loop)
            return future.result()
        except RuntimeError:
            return run_in_new_loop()

    async def async_logging_hook(
        self, kwargs: dict, result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Send both prompt and response to Purview for audit logging.

        Errors are logged but never raised — this mode is non-blocking.
        """
        try:
            metadata = kwargs.get("metadata") or kwargs.get("litellm_metadata") or {}
            user_id = metadata.get(self.user_id_field) or kwargs.get(
                "user_api_key_user_id"
            )

            if not user_id:
                verbose_proxy_logger.debug("Purview audit: no user_id, skipping")
                return kwargs, result

            # Log prompt (uploadText)
            messages = kwargs.get("messages")
            if messages:
                user_prompt = self.get_user_prompt(messages)
                if user_prompt:
                    await self._check_content(
                        user_id=user_id,
                        text=user_prompt,
                        activity="uploadText",
                        request_data=kwargs,
                        block_on_violation=False,
                    )

            # Log response (downloadText)
            from litellm.types.utils import Choices, ModelResponse

            if isinstance(result, ModelResponse) and result.choices:
                if isinstance(result.choices[0], Choices):
                    content = result.choices[0].message.content or ""
                    if content:
                        await self._check_content(
                            user_id=user_id,
                            text=content,
                            activity="downloadText",
                            request_data=kwargs,
                            block_on_violation=False,
                        )
        except Exception as e:
            verbose_proxy_logger.error("Purview audit logging error: %s", e)

        return kwargs, result
