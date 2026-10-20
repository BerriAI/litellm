"""
CompactifAI chat completion transformation
"""

from collections.abc import Iterable, Mapping, Sequence, Sized
from typing import TYPE_CHECKING, Any, Final, Protocol

import httpx
from pydantic import InstanceOf, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.common_utils import OpenAIError
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_RESPONSE_BODY: Final = TypeAdapter(InstanceOf[dict[str, object]])
_JSON_VALUES: Final = TypeAdapter(Iterable[object])
_JSON_OBJECT: Final = TypeAdapter(InstanceOf[Mapping[str, object]])
_MESSAGE: Final = TypeAdapter(InstanceOf[dict[str, object]])
_SIZED: Final = TypeAdapter(InstanceOf[Sized])
_TOOL_CALLS: Final = TypeAdapter(InstanceOf[Sequence[object]])


class _JsonResponse(Protocol):
    def json(self) -> object: ...


def _json_body(response: _JsonResponse) -> object:
    return response.json()


class CompactifAIResponseFields(TypedDict, total=False):
    """The chat completion fields of a CompactifAI response body."""

    id: ReadOnly[str]
    choices: ReadOnly[Sequence[Mapping[str, object]]]
    created: ReadOnly[int]
    model: ReadOnly[str]
    system_fingerprint: ReadOnly[str | None]
    usage: ReadOnly[Mapping[str, object]]
    object: ReadOnly[str]


class CompactifAIChatConfig(OpenAIGPTConfig):
    """
    Configuration class for CompactifAI chat completions.
    Since CompactifAI is OpenAI-compatible, we extend OpenAIGPTConfig.
    """

    def _get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        """
        Get API base and key for CompactifAI provider.
        """
        api_base = api_base or "https://api.compactif.ai/v1"
        dynamic_api_key: Final = api_key or get_secret_str("COMPACTIFAI_API_KEY") or ""
        return api_base, dynamic_api_key

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: Mapping[str, object],
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        """
        Transform CompactifAI response to LiteLLM format.
        Since CompactifAI is OpenAI-compatible, we can use the standard OpenAI transformation.
        """
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        response_json: Final = _RESPONSE_BODY.validate_python(_json_body(raw_response))

        # Handle JSON mode if needed
        if json_mode:
            for choice in _JSON_VALUES.validate_python(response_json["choices"]):
                message = _JSON_OBJECT.validate_python(choice).get("message")
                message_fields = _MESSAGE.validate_python(message) if message else None
                if message_fields is not None and message_fields.get("tool_calls"):
                    # Convert tool calls to content for JSON mode
                    tool_calls = message_fields.get("tool_calls", [])
                    if len(_SIZED.validate_python(tool_calls)) == 1:
                        first_call = _JSON_OBJECT.validate_python(_TOOL_CALLS.validate_python(tool_calls)[0])
                        function = _JSON_OBJECT.validate_python(first_call["function"])
                        message_fields["content"] = function.get("arguments", "")
                        message_fields["tool_calls"] = None

        returned_response: Final = ModelResponse(**response_json)

        # Set model name with provider prefix
        returned_response.model = f"compactifai/{model}"

        return returned_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict[str, str] | httpx.Headers
    ) -> BaseLLMException:
        """
        Get the appropriate error class for CompactifAI errors.
        Since CompactifAI is OpenAI-compatible, we use OpenAI error handling.
        """
        return OpenAIError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
