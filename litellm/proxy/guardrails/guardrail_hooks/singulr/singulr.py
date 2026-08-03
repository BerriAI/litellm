import os
from typing import Any
from urllib.parse import urlparse

import httpx
import pydantic

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
    SingulrGuardrailResponse,
)
from litellm.types.utils import GenericGuardrailAPIInputs

_DEFAULT_API_BASE = "http://localhost:8003"
_GUARD_ENDPOINT = "/api/v1/ai-gateway/litellm"
_DEFAULT_TIMEOUT = 30.0


class SingulrGuardrail(CustomGuardrail):
    def __init__(
        self,
        singulr_api_key: str | None = None,
        singulr_api_base: str | None = None,
        singulr_application_id: str | None = None,
        singulr_guardrail_id: str | None = None,
        block_on_error: bool | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.singulr_api_key = singulr_api_key or os.environ.get("SINGULR_API_KEY")
        self.singulr_api_base = (singulr_api_base or os.environ.get("SINGULR_API_BASE") or _DEFAULT_API_BASE).rstrip(
            "/"
        )
        parsed = urlparse(self.singulr_api_base)
        if parsed.scheme == "http" and parsed.hostname not in (
            "localhost",
            "127.0.0.1",
        ):
            raise ValueError(
                f"Singulr: api_base {self.singulr_api_base} uses plain HTTP for a "
                "non-local endpoint. Guardrail payloads contain the API token, full "
                "conversation content, and the guardrail decision, so this endpoint "
                "must use HTTPS."
            )

        self.singulr_application_id = singulr_application_id or os.environ.get("SINGULR_ENFORCEMENT_ENTITY_ID")
        self.singulr_guardrail_id = singulr_guardrail_id or os.environ.get("SINGULR_GUARDRAIL_ID")

        if block_on_error is None:
            env = os.environ.get("SINGULR_BLOCK_ON_ERROR", "true")
            self.block_on_error = env.lower() in ("true", "1", "yes")
        else:
            self.block_on_error = block_on_error

        self.timeout = _DEFAULT_TIMEOUT if timeout is None else timeout

        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ]

        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.singulr import (
            SingulrGuardrailConfigModel,
        )

        return SingulrGuardrailConfigModel

    def _build_payload(
        self,
        request_data: dict[str, Any],
        inputs: GenericGuardrailAPIInputs,
        input_type: str,
    ) -> dict[str, Any]:
        if not request_data:
            texts = inputs.get("texts", [])

            payload = SingulrGuardrailPayload(
                input_type=input_type,
                is_playground_request=True,
                playground_text=texts[0] if texts else None,
            )
        else:
            response = request_data.get("response")
            singulr_req_object = SingulrGuardrailRequest(
                model=request_data.get("model"),
                messages=request_data.get("messages"),
                tools=request_data.get("tools"),
                model_response=response.model_dump(mode="json") if input_type == "response" and response else None,
                litellm_metadata=request_data.get("litellm_metadata"),
            )
            payload = SingulrGuardrailPayload(
                litellm_call_id=request_data.get("litellm_call_id"),
                request_data=singulr_req_object,
                input_type=input_type,
            )

        return payload.model_dump(mode="json")

    def _build_headers(self) -> dict[str, str]:
        return dict(
            (header, value)
            for header, value in (
                ("Content-Type", "application/json"),
                ("X-Singulr-Gateway-Token", self.singulr_api_key),
                (
                    "X-Singulr-Enforcement-Entity-Id",
                    self.singulr_application_id or "",
                ),
                ("X-Singulr-Guardrail-Id", self.singulr_guardrail_id or ""),
            )
            if value
        )

    async def _call_api(self, payload: dict[str, Any]) -> SingulrGuardrailResponse | None:
        endpoint = f"{self.singulr_api_base}{_GUARD_ENDPOINT}"
        verbose_proxy_logger.debug("Singulr: %s", endpoint)

        try:
            response = await self.async_handler.post(
                url=endpoint,
                headers=self._build_headers(),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = SingulrGuardrailResponse.model_validate(response.json())
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

        except (ValueError, pydantic.ValidationError) as exc:
            verbose_proxy_logger.error("Singulr API returned an invalid response: %s", str(exc))
            if self.block_on_error:
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"Singulr API returned an invalid response: {exc}",
                ) from exc
            return None

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: str,
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        payload = self._build_payload(request_data, inputs, input_type)
        if not payload:
            return inputs

        result = await self._call_api(payload)
        if result is None:
            return inputs

        verbose_proxy_logger.debug(
            "Singulr: should_block=%s blocking_due_to=%s",
            result.should_block,
            result.blocking_due_to,
        )

        if result.should_block:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Blocked by Singulr: {result.blocking_due_to or 'unknown'}",
            )

        return inputs
