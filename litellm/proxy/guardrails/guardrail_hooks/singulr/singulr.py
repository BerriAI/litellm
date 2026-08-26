import asyncio
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Final
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
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.base import (
    GuardrailConfigModel,
)
from litellm.types.proxy.guardrails.guardrail_hooks.singulr import (
    AssistantMessage,
    SingulrGuardrailPayload,
    SingulrGuardrailResponse,
    SingulrMcpGuardrailPayload,
    ToolCall,
    ToolCallFunction,
)
from litellm.types.utils import (
    GenericGuardrailAPIInputs,
    GuardrailStatus,
    StandardLoggingGuardrailInformation,
)

_DEFAULT_API_BASE: Final = "http://localhost:8003"
_GUARD_ENDPOINT: Final = "/api/v1/ai-gateway/litellm"
_DEFAULT_TIMEOUT: Final = 30.0
_EMPTY_MAPPING: Final[Mapping[str, Any]] = MappingProxyType({})


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
        self.singulr_api_base = (
            (singulr_api_base or os.environ.get("SINGULR_API_BASE") or _DEFAULT_API_BASE).strip().rstrip("/")
        )
        parsed: Final = urlparse(self.singulr_api_base)
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
            env: Final = os.environ.get("SINGULR_BLOCK_ON_ERROR", "true")
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
                GuardrailEventHooks.logging_only,
                GuardrailEventHooks.pre_mcp_call,
                GuardrailEventHooks.post_mcp_call,
            ]

        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.singulr import (
            SingulrGuardrailConfigModel,
        )

        return SingulrGuardrailConfigModel

    @staticmethod
    def _metadata_containers(request_data: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        """Candidate metadata dicts to check, in priority order.

        Most call paths put metadata at the top level of ``request_data``
        (``litellm_metadata`` or ``metadata``). ``post_mcp_call`` instead hands
        us ``litellm_logging_obj.model_call_details``, which nests it under
        ``litellm_params`` instead, so that's checked as a fallback.
        """
        litellm_params: Final = request_data.get("litellm_params") or _EMPTY_MAPPING
        return tuple(
            container
            for container in (
                request_data.get("litellm_metadata"),
                request_data.get("metadata"),
                litellm_params.get("litellm_metadata") if litellm_params else None,
                litellm_params.get("metadata") if litellm_params else None,
            )
            if container
        )

    @classmethod
    def _resolve_metadata_value(cls, request_data: Mapping[str, Any], key: str) -> str | None:
        for container in cls._metadata_containers(request_data=request_data):
            value = container.get(key)
            if value:
                return value
        return None

    @classmethod
    def _resolve_user_role_from_request_data(cls, request_data: Mapping[str, Any]) -> str | None:
        for container in cls._metadata_containers(request_data=request_data):
            auth = container.get("user_api_key_auth")
            if isinstance(auth, UserAPIKeyAuth) and auth.user_role:
                return auth.user_role.value
        return None

    @classmethod
    def _build_metadata(cls, request_data: Mapping[str, Any]) -> Mapping[str, Any] | None:
        fields: Final = (
            "user_api_key_alias",
            "user_api_key_user_id",
            "user_api_key_user_email",
            "user_api_key_org_id",
            "user_api_key_org_alias",
            "user_api_key_team_id",
            "user_api_key_team_alias",
        )
        resolved: Final = (
            *((field, cls._resolve_metadata_value(request_data=request_data, key=field)) for field in fields),
            ("user_api_key_user_role", cls._resolve_user_role_from_request_data(request_data=request_data)),
        )
        if not any(value for _, value in resolved):
            return None
        return {key: value for key, value in resolved if value}  # mutable-ok: short-lived JSON payload dict

    @staticmethod
    def _build_user_message(text: str) -> Mapping[str, Any]:
        return {"role": "user", "content": text}  # mutable-ok: short-lived JSON payload dict

    def _build_headers(self) -> Mapping[str, str]:
        all_headers: Final = MappingProxyType(
            {
                "Content-Type": "application/json",
                "X-Singulr-Gateway-Token": self.singulr_api_key,
                "X-Singulr-Enforcement-Entity-Id": self.singulr_application_id,
                "X-Singulr-Guardrail-Id": self.singulr_guardrail_id,
            }
        )
        return MappingProxyType({header: value for header, value in all_headers.items() if value})

    async def _call_api(self, payload: Mapping[str, Any]) -> SingulrGuardrailResponse | None:
        endpoint: Final = f"{self.singulr_api_base}{_GUARD_ENDPOINT}"
        verbose_proxy_logger.debug("Singulr: %s", endpoint)

        try:
            response: Final = await self.async_handler.post(
                url=endpoint,
                headers=self._build_headers(),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result: Final = SingulrGuardrailResponse.model_validate(response.json())
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
                    message=f"Singulr API returned HTTP {exc.response.status_code}: {exc.response.text}",
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

    async def _apply_guardrail_on_request(
        self,
        inputs: GenericGuardrailAPIInputs,
        texts: Sequence[str],
        structured_messages: Sequence[Any],
        request_data: Mapping[str, Any],
    ) -> GenericGuardrailAPIInputs:
        messages: Final = (
            tuple(structured_messages)
            if structured_messages
            else tuple(self._build_user_message(text) for text in texts)
        )

        images: Final = inputs.get("images")
        tools: Final = inputs.get("tools")

        if not messages and not images and not tools:
            verbose_proxy_logger.debug("Singulr: No messages, images, or tools to check after filtering")
            return inputs

        metadata: Final = self._build_metadata(request_data=request_data)

        singulr_req_obj = SingulrGuardrailPayload(
            correlation_id=request_data.get("litellm_call_id"),
            model_name=inputs.get("model"),
            guardrail_scope="request",
            messages=messages,
            images=images,
            tools=tools,
            metadata=metadata,
        )
        payload = singulr_req_obj.model_dump(mode="json")
        guardrail_resp = await self._call_api(payload)

        if guardrail_resp is None:
            return inputs

        if guardrail_resp.should_block:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                status_code=400,
                message=f"Blocked by Singulr, Blocking due to {guardrail_resp.blocking_due_to or 'unknown'}",
            )
        return inputs

    async def _apply_guardrail_on_mcp_request(self, request_data: Mapping[str, Any]) -> None:
        metadata: Final = self._build_metadata(request_data=request_data)

        singulr_mcp_obj = SingulrMcpGuardrailPayload(
            guardrail_scope="mcp_request",
            tool_name=request_data.get("mcp_tool_name"),
            tool_arguments=request_data.get("mcp_arguments"),
            mcp_server_name=request_data.get("mcp_server_name"),
            metadata=metadata,
        )
        payload = singulr_mcp_obj.model_dump(mode="json")
        guardrail_resp = await self._call_api(payload)

        if guardrail_resp is None:
            return

        if guardrail_resp.should_block:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                status_code=400,
                message=f"Blocked by Singulr, Blocking due to {guardrail_resp.blocking_due_to or 'unknown'}",
            )

    async def _apply_guardrail_on_mcp_response(
        self, inputs: GenericGuardrailAPIInputs, texts: Sequence[str], request_data: Mapping[str, Any]
    ) -> GenericGuardrailAPIInputs:
        if not texts:
            return inputs

        metadata: Final = self._build_metadata(request_data=request_data)

        singulr_mcp_obj = SingulrMcpGuardrailPayload(
            model_name=request_data.get("model"),
            guardrail_scope="mcp_response",
            tool_result=texts,
            metadata=metadata,
        )
        payload = singulr_mcp_obj.model_dump(mode="json")
        guardrail_resp = await self._call_api(payload)

        if guardrail_resp is None:
            return inputs

        if guardrail_resp.should_block:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                status_code=400,
                message=f"Blocked by Singulr, Blocking due to {guardrail_resp.blocking_due_to or 'unknown'}",
            )

        return inputs

    @staticmethod
    def _build_tool_call(tool_call: Mapping[str, Any]) -> "ToolCall | None":
        tool_call_id: Final = tool_call.get("id")
        fun: Final = tool_call.get("function")
        if not tool_call_id or not fun:
            return None
        func_name: Final = fun.get("name")
        args: Final = fun.get("arguments")
        if not func_name or args is None:
            return None
        return ToolCall(
            id=tool_call_id,
            type=tool_call.get("type"),
            function=ToolCallFunction(name=func_name, arguments=args),
        )

    async def _apply_guardrail_on_response(
        self, inputs: GenericGuardrailAPIInputs, texts: Sequence[str], request_data: Mapping[str, Any]
    ) -> GenericGuardrailAPIInputs:
        combined_texts: Final = "\n".join(texts) if texts else None

        tool_calls: Final = inputs.get("tool_calls", ())
        tool_calls_res: Final = tuple(
            tool_call_res
            for tool_call_res in (self._build_tool_call(tool_call) for tool_call in tool_calls)
            if tool_call_res is not None
        )

        assistant_message: Final = AssistantMessage(
            role="assistant",
            content=combined_texts,
            tool_calls=tool_calls_res,
        )

        metadata: Final = self._build_metadata(request_data=request_data)

        singulr_resp_obj = SingulrGuardrailPayload(
            correlation_id=request_data.get("litellm_call_id"),
            guardrail_scope="response",
            messages=request_data.get("messages"),
            images=inputs.get("images"),
            response=assistant_message,
            metadata=metadata,
        )

        payload = singulr_resp_obj.model_dump(mode="json")
        guardrail_resp = await self._call_api(payload)

        if guardrail_resp is None:
            return inputs

        if guardrail_resp.should_block:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                status_code=400,
                message=f"Blocked by Singulr, Blocking due to {guardrail_resp.blocking_due_to or 'unknown'}",
            )
        return inputs

    async def async_logging_hook(
        self,
        kwargs: dict,  # mutable-ok: matches CustomLogger override; mutated via setdefault
        result: Any,  # noqa: ANN401  # required by CustomLogger.async_logging_hook override signature
        call_type: str,
    ) -> tuple[dict, Any]:
        start_time: Final = datetime.now(timezone.utc)
        guardrail_status: GuardrailStatus = "success"
        try:
            messages: Final = kwargs.get("messages") or ()
            if messages:
                request_metadata: Final = self._build_metadata(request_data=kwargs)
                singulr_req_obj = SingulrGuardrailPayload(
                    correlation_id=kwargs.get("litellm_call_id"),
                    model_name=kwargs.get("model"),
                    guardrail_scope="request",
                    messages=messages,
                    metadata=request_metadata,
                )
                payload_req = singulr_req_obj.model_dump(mode="json")
                await self._call_api(payload_req)

            if result:
                response_metadata: Final = self._build_metadata(request_data=kwargs)
                singulr_res_obj = SingulrGuardrailPayload(
                    correlation_id=kwargs.get("litellm_call_id"),
                    guardrail_scope="response",
                    response=result,
                    metadata=response_metadata,
                )
                try:
                    payload = singulr_res_obj.model_dump(mode="json")
                except Exception as exc:  # noqa: BLE001  # result can be any callback shape; fall back to a stringified report
                    verbose_proxy_logger.debug("Singulr: could not JSON-serialize response, falling back: %s", exc)
                    payload = {
                        "correlation_id": kwargs.get("litellm_call_id"),
                        "guardrail_scope": "response",
                        "response": str(result),
                        "metadata": response_metadata,
                    }
                await self._call_api(payload)
        except GuardrailRaisedException:
            guardrail_status = "guardrail_intervened"
        except Exception as exc:  # noqa: BLE001  # logging_only must never break the request
            verbose_proxy_logger.debug("Singulr: logging_only hook swallowed exception: %s", exc)
            return kwargs, result

        end_time: Final = datetime.now(timezone.utc)
        slg: Final = StandardLoggingGuardrailInformation(
            guardrail_name=self.guardrail_name or "singulr",
            guardrail_mode=GuardrailEventHooks.logging_only,
            guardrail_status=guardrail_status,
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            duration=(end_time - start_time).total_seconds(),
            masked_entity_count=None,
        )
        standard_logging_object: Final = kwargs.setdefault(
            "standard_logging_object",
            {},  # mutable-ok: shared, mutated accumulator
        )
        existing = standard_logging_object.get("guardrail_information")
        if isinstance(existing, list):
            existing.append(slg)
        else:
            standard_logging_object["guardrail_information"] = [slg]  # mutable-ok: shared accumulator

        return kwargs, result

    def logging_hook(
        self,
        kwargs: dict,  # mutable-ok: required by CustomLogger.logging_hook override signature
        result: Any,  # noqa: ANN401  # required by CustomLogger.logging_hook override signature
        call_type: str,
    ) -> tuple[dict, Any]:
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            if loop.is_running():
                verbose_proxy_logger.debug(
                    "Singulr: sync logging_hook called from a running loop; skipping logging_only report"
                )
                return kwargs, result
            loop.run_until_complete(self.async_logging_hook(kwargs=kwargs, result=result, call_type=call_type))
        except Exception as exc:  # noqa: BLE001  # logging_only must never break the request
            verbose_proxy_logger.debug("Singulr: sync logging_hook swallowed exception: %s", exc)
        return kwargs, result

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,  # mutable-ok: required by CustomGuardrail.apply_guardrail override signature
        input_type: str,
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = inputs.get("texts", ())
        structured_messages: Final = inputs.get("structured_messages", ())

        verbose_proxy_logger.debug(
            "Singulr Guardrail: apply_guardrail called with input_type=%s, texts=%d, structured_messages=%d",
            input_type,
            len(texts),
            len(structured_messages),
        )

        if input_type == "request":
            if request_data.get("mcp_tool_name"):
                await self._apply_guardrail_on_mcp_request(request_data=request_data)
                return inputs
            return await self._apply_guardrail_on_request(
                inputs=inputs, texts=texts, structured_messages=structured_messages, request_data=request_data
            )
        elif input_type == "response":
            if request_data.get("call_type") == "call_mcp_tool":
                return await self._apply_guardrail_on_mcp_response(
                    inputs=inputs, texts=texts, request_data=request_data
                )
            return await self._apply_guardrail_on_response(inputs=inputs, texts=texts, request_data=request_data)
        return inputs
