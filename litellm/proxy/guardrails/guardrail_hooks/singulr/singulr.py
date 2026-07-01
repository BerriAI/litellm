"""
Singulr guardrail integration for LiteLLM.
Calls the Singulr Guard API to scan messages.
"""

import os
from typing import Any, Optional, cast
from urllib.parse import urlparse

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.litellm_logging import (
    Logging as LiteLLMLoggingObj,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.base import (
    GuardrailConfigModel,
)
from litellm.types.proxy.guardrails.guardrail_hooks.singulr import (
    SingulrGuardrailPayload,
    SingulrGuardrailRequest,
)
from litellm.types.utils import GenericGuardrailAPIInputs

_DEFAULT_API_BASE = "http://localhost:8003"
_GUARD_ENDPOINT = "/api/v1/ai-gateway/litellm"


class SingulrGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        enforcement_entity_id: Optional[str] = None,
        guardrail_id: Optional[str] = None,
        block_on_error: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key or os.environ.get("SINGULR_API_KEY")

        self.api_base = (api_base or os.environ.get("SINGULR_API_BASE") or _DEFAULT_API_BASE).rstrip("/")

        parsed = urlparse(self.api_base)
        if parsed.scheme == "http" and parsed.hostname not in (
            "localhost",
            "127.0.0.1",
        ):
            verbose_proxy_logger.warning(
                "Singulr: api_base %s uses plain HTTP. Guardrail payloads contain "
                "full message content and will be sent unencrypted. Use HTTPS for "
                "any non-local endpoint.",
                self.api_base,
            )

        self.enforcement_entity_id = enforcement_entity_id or os.environ.get("SINGULR_ENFORCEMENT_ENTITY_ID")
        self.guardrail_id = guardrail_id or os.environ.get("SINGULR_GUARDRAIL_ID")

        if block_on_error is None:
            env = os.environ.get("SINGULR_BLOCK_ON_ERROR", "true")
            self.block_on_error = env.lower() in ("true", "1", "yes")
        else:
            self.block_on_error = block_on_error

        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
            ]

        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> Optional[type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.singulr import (
            SingulrGuardrailConfigModel,
        )

        return SingulrGuardrailConfigModel

    def _build_payload(
        self,
        request_data: dict[str, Any],
        input_type: str,
    ) -> dict[str, Any]:
        request = SingulrGuardrailRequest.model_validate(request_data)
        if not request.model_dump(exclude_none=True):
            return {}

        payload = SingulrGuardrailPayload(request=request, input_type=input_type)
        return payload.model_dump(exclude_none=True)

    def _build_headers(self) -> dict[str, str]:
        return dict(
            (header, value)
            for header, value in (
                ("Content-Type", "application/json"),
                ("X-Singulr-Gateway-Token", self.api_key),
                (
                    "X-Singulr-Enforcement-Entity-Id",
                    self.enforcement_entity_id or "",
                ),
                ("X-Singulr-Guardrail-Id", self.guardrail_id or ""),
            )
            if value
        )

    async def _call_api(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        endpoint = f"{self.api_base}{_GUARD_ENDPOINT}"
        verbose_proxy_logger.debug("Singulr: %s", endpoint)

        try:
            response = await self.async_handler.post(
                url=endpoint,
                headers=self._build_headers(),
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            verbose_proxy_logger.debug("Singulr: result=%s", result)
            return result

        except httpx.HTTPStatusError as exc:
            verbose_proxy_logger.error(
                "Singulr API returned HTTP %s: %s",
                exc.response.status_code,
                str(exc),
            )
            if self.block_on_error:
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=(f"Singulr API returned HTTP {exc.response.status_code}: {exc.response.text}"),
                ) from exc
            return None

        except httpx.TransportError as exc:
            verbose_proxy_logger.error("Singulr API unreachable: %s", str(exc))
            if self.block_on_error:
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"Singulr API unreachable (block_on_error=True): {exc}",
                ) from exc
            return None

        except ValueError as exc:
            verbose_proxy_logger.error("Singulr API returned non-JSON response: %s", str(exc))
            if self.block_on_error:
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"Singulr API returned non-JSON response: {exc}",
                ) from exc
            return None

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: str,
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        payload = self._build_payload(cast(dict[str, Any], request_data), input_type)
        verbose_proxy_logger.debug("Singulr: payload=%s", payload)
        if not payload:
            return inputs

        result = await self._call_api(payload)
        if result is None:
            return inputs

        should_block = result.get("should_block", False)
        verbose_proxy_logger.debug(
            "Singulr: should_block=%s blocking_due_to=%s",
            should_block,
            result.get("blocking_due_to"),
        )

        if should_block:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Blocked by Singulr: {result.get('blocking_due_to', 'unknown')}",
            )

        return inputs
