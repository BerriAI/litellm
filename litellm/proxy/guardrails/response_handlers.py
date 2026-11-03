"""Response policies for the standalone guardrail endpoint."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.types.guardrails import (
    ApplyGuardrailRequest,
    ApplyGuardrailResponse,
    GuardrailEventHooks,
    GuardrailResponseFormat,
    NativeGuardrailResponse,
)
from litellm.types.utils import GenericGuardrailAPIInputs

GuardrailEndpointResponse: TypeAlias = ApplyGuardrailResponse | NativeGuardrailResponse
_NATIVE_RESPONSE: Final[TypeAdapter[NativeGuardrailResponse]] = TypeAdapter(NativeGuardrailResponse)


class StandardGuardrailResponseHandler:
    """Preserve the endpoint's existing text response and hook behavior."""

    def prepare_request_data(self, request: ApplyGuardrailRequest) -> Mapping[str, object]:
        return request.model_dump(include={"messages", "metadata"}, exclude_none=True)

    def resolve_input_type(
        self, request: ApplyGuardrailRequest, guardrail: CustomGuardrail
    ) -> Literal["request", "response"]:
        if request.input_type == "request" and guardrail.event_hook in (GuardrailEventHooks.post_call, "post_call"):
            return "response"
        return "response" if request.input_type == "response" else "request"

    def build_response(
        self,
        request: ApplyGuardrailRequest,
        processed_inputs: GenericGuardrailAPIInputs,
        request_data: Mapping[str, object],
    ) -> GuardrailEndpointResponse:
        texts: Final = processed_inputs.get("texts", [])
        return ApplyGuardrailResponse(response_text=texts[0] if texts else request.text)

    def accept_hook_result(self, original: GuardrailEndpointResponse, modified: object) -> GuardrailEndpointResponse:
        return modified if isinstance(modified, ApplyGuardrailResponse) else original


class _NativeGuardrailRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    guardrail_name: str | None = None
    guardrail_response: JsonValue = None


class _NativeGuardrailMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    standard_logging_guardrail_information: tuple[_NativeGuardrailRecord, ...] = ()


class NativeGuardrailResponseHandler(StandardGuardrailResponseHandler):
    """Return structured findings recorded by this guardrail invocation."""

    def prepare_request_data(self, request: ApplyGuardrailRequest) -> Mapping[str, object]:
        return request.model_dump(
            include={"messages", "metadata"},
            exclude={"metadata": {"standard_logging_guardrail_information"}},
            exclude_none=True,
        )

    def resolve_input_type(
        self, request: ApplyGuardrailRequest, guardrail: CustomGuardrail
    ) -> Literal["request", "response"]:
        if "input_type" in request.model_fields_set:
            return "response" if request.input_type == "response" else "request"
        return super().resolve_input_type(request, guardrail)

    def build_response(
        self,
        request: ApplyGuardrailRequest,
        processed_inputs: GenericGuardrailAPIInputs,
        request_data: Mapping[str, object],
    ) -> GuardrailEndpointResponse:
        metadata: Final = request_data.get(get_metadata_variable_name_from_kwargs(request_data))
        recorded: Final = _NativeGuardrailMetadata.model_validate_json(
            safe_dumps(metadata if metadata is not None else {})
        )
        findings: Final = tuple(
            record.guardrail_response
            for record in recorded.standard_logging_guardrail_information
            if record.guardrail_name == request.guardrail_name and isinstance(record.guardrail_response, (dict, list))
        )
        if not findings:
            raise HTTPException(
                status_code=422,
                detail=f"Guardrail '{request.guardrail_name}' did not report structured native findings.",
            )
        return findings[0] if len(findings) == 1 else findings

    def accept_hook_result(self, original: GuardrailEndpointResponse, modified: object) -> GuardrailEndpointResponse:
        if modified is None:
            return original
        return _NATIVE_RESPONSE.validate_python(modified)


GUARDRAIL_RESPONSE_HANDLERS: Final[Mapping[GuardrailResponseFormat, StandardGuardrailResponseHandler]] = (
    MappingProxyType(
        {
            GuardrailResponseFormat.STANDARD: StandardGuardrailResponseHandler(),
            GuardrailResponseFormat.NATIVE: NativeGuardrailResponseHandler(),
        }
    )
)
