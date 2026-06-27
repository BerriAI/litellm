from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import httpx
from fastapi import HTTPException
from httpx import Response as HttpxResponse
from typing_extensions import TypeGuard

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]
    httpxSpecialProvider,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

BYPASS_HEADER = "x-headroom-bypass"


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, list)


class HeadroomGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None = None,
        default_on: bool = False,
    ):
        self.headroom_api_base = (api_base or get_secret_str("HEADROOM_API_BASE") or "").rstrip("/")
        if not self.headroom_api_base:
            raise ValueError(
                "Headroom guardrail requires an API base URL. "
                "Set `api_base` in the guardrail config or HEADROOM_API_BASE env var."
            )
        self.headroom_api_key = api_key or get_secret_str("HEADROOM_API_KEY")
        self.headroom_model = model
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
        )

    def _should_bypass(self, request_data: dict) -> bool:
        psr = request_data.get("proxy_server_request")
        if not _is_str_object_dict(psr):
            return False
        headers = psr.get("headers")
        if not _is_str_object_dict(headers):
            return False
        value = headers.get(BYPASS_HEADER)
        return str(value).lower() == "true"

    async def _call_compress(
        self,
        messages: list[dict[str, object]],
        model: str | None,
    ) -> list[dict[str, object]]:
        payload: dict[str, object] = {"messages": messages}
        if model:
            payload["model"] = model

        request_headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.headroom_api_key:
            request_headers["Authorization"] = f"Bearer {self.headroom_api_key}"

        try:
            raw_response: HttpxResponse | None = await self.async_handler.post(  # pyright: ignore[reportUnknownMemberType]
                url=f"{self.headroom_api_base}/v1/compress",
                json=payload,
                headers=request_headers,
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError) as e:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "Headroom compression service unreachable",
                    "detail": str(e),
                },
            ) from e
        if raw_response is None:
            raise HTTPException(
                status_code=502,
                detail={"error": "Headroom compression service returned no response"},
            )
        response: HttpxResponse = raw_response

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "Headroom compression service returned an error",
                    "status_code": response.status_code,
                    "body": response.text,
                },
            )

        try:
            body: object = response.json()
        except Exception:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "Headroom compression service returned non-JSON response",
                    "body": response.text[:500],
                },
            )
        if not _is_str_object_dict(body):
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "Headroom compression service returned unexpected response shape",
                    "body": response.text[:500],
                },
            )

        compressed_messages = body.get("messages")
        if not _is_object_list(compressed_messages):
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "Headroom compression service response missing 'messages'",
                    "body": response.text,
                },
            )

        filtered = [item for item in compressed_messages if _is_str_object_dict(item)]
        if not filtered:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "Headroom compression service returned empty message list",
                    "body": response.text,
                },
            )

        verbose_proxy_logger.debug(
            "Headroom: compressed %s tokens -> %s tokens (ratio %.2f)",
            body.get("tokens_before", "?"),
            body.get("tokens_after", "?"),
            body.get("compression_ratio", 0),
        )
        return filtered

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type != "request":
            return inputs

        if self._should_bypass(request_data):
            verbose_proxy_logger.debug("Headroom: %s header set; skipping compression", BYPASS_HEADER)
            return inputs

        structured_messages = inputs.get("structured_messages")
        if not _is_object_list(structured_messages) or not structured_messages:
            return inputs

        messages = [m for m in structured_messages if _is_str_object_dict(m)]
        if not messages:
            return inputs

        model = self.headroom_model or request_data.get("model")
        compressed = await self._call_compress(
            messages=messages,
            model=model if isinstance(model, str) else None,
        )

        return {**inputs, "structured_messages": compressed}  # pyright: ignore[reportReturnType]

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel[object]] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.headroom import (
            HeadroomGuardrailConfigModel,
        )

        return HeadroomGuardrailConfigModel
