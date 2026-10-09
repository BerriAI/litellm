"""Databricks ResponsesAgent endpoints.

Model Serving: POST {workspace}/serving-endpoints/{endpoint}/invocations
Model Serving, unified: POST {workspace}/serving-endpoints/responses with the endpoint as the body "model"
Databricks Apps: POST {app}/responses
"""

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Final
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.databricks.agent.responses_output import (
    AgentResponse,
    DatabricksAgentResponsesIterator,
    output_text,
)
from litellm.llms.databricks.common_utils import DatabricksException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

PROVIDER_NAME: Final = "databricks_agent"
_MODEL_PREFIX: Final = f"{PROVIDER_NAME}/"
_ENDPOINT_URL_SUFFIXES: Final = ("/invocations", "/responses")
_UNIFIED_RESPONSES_SUFFIX: Final = "/serving-endpoints/responses"
_INPUT_ROLES: Final = frozenset({"system", "developer", "user", "assistant"})
_PASSTHROUGH_PARAMS: Final = ("custom_inputs", "context", "databricks_options")


def endpoint_name_from_model(model: str) -> str:
    return model.removeprefix(_MODEL_PREFIX).strip()


def _normalized_base(api_base: str) -> str:
    return api_base.strip().rstrip("/")


def _is_unified_responses_url(api_base: str) -> bool:
    return _normalized_base(api_base).endswith(_UNIFIED_RESPONSES_SUFFIX)


def _required_endpoint_name(model: str) -> str:
    endpoint: Final = endpoint_name_from_model(model)
    if not endpoint:
        raise DatabricksException(
            status_code=400,
            message=(
                "model must name the Databricks serving endpoint (model: databricks_agent/<endpoint>) "
                "unless api_base is the endpoint's /invocations URL or a Databricks App /responses URL"
            ),
        )
    return endpoint


def resolve_endpoint_url(api_base: str, model: str) -> str:
    base: Final = _normalized_base(api_base)
    if _is_unified_responses_url(base):
        _required_endpoint_name(model)
        return base
    if base.endswith(_ENDPOINT_URL_SUFFIXES):
        return base
    workspace: Final = base.removesuffix("/serving-endpoints")
    return f"{workspace}/serving-endpoints/{quote(_required_endpoint_name(model), safe='')}/invocations"


def _body_model_field(api_base: object, model: str) -> Mapping[str, str]:
    if not isinstance(api_base, str) or not _is_unified_responses_url(api_base):
        return {}
    return {"model": _required_endpoint_name(model)}


class _InputContentPart(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = "unknown"


class _InputMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str | tuple[_InputContentPart, ...] | None = None


def _drop_params_enabled(litellm_params: Mapping[str, object]) -> bool:
    return litellm.drop_params is True or litellm_params.get("drop_params") is True


def _input_item(message: AllMessageValues, drop_unsupported_content: bool) -> Mapping[str, str]:
    role: Final = message["role"]
    if role not in _INPUT_ROLES:
        raise DatabricksException(
            status_code=400,
            message=(
                f"message role '{role}' is not supported by a Databricks ResponsesAgent; "
                "send system, developer, user, or assistant messages"
            ),
        )
    unsupported: Final = _non_text_content_types(message)
    if unsupported and not drop_unsupported_content:
        raise DatabricksException(
            status_code=400,
            message=(
                f"databricks_agent sends text content only; remove the {', '.join(unsupported)} content "
                "part(s) from the request, or set drop_params: true to drop them"
            ),
        )
    return {"role": role, "content": convert_content_list_to_str(message)}


def _non_text_content_types(message: AllMessageValues) -> tuple[str, ...]:
    content: Final = _InputMessage.model_validate(message).content
    if not isinstance(content, tuple):
        return ()
    return tuple(sorted({part.type for part in content} - {"text"}))


def _has_authorization(headers: Mapping[str, object]) -> bool:
    return any(name.lower() == "authorization" for name in headers)


class DatabricksAgentConfig(BaseConfig):
    @staticmethod
    def resolve_api_base_and_key(api_base: str | None, api_key: str | None) -> tuple[str | None, str | None]:
        return (
            api_base or get_secret_str("DATABRICKS_API_BASE"),
            api_key or get_secret_str("DATABRICKS_API_KEY"),
        )

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: mirrors override contract
        return ["stream"]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: mirrors override contract
        return dict(optional_params)

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: mirrors override contract
        if _has_authorization(headers):
            return {**headers, "Content-Type": "application/json"}
        if not api_key:
            raise DatabricksException(
                status_code=400,
                message=(
                    "Missing Databricks credentials: set api_key to a personal access token, set the "
                    "DATABRICKS_API_KEY environment variable, or register the agent with OAuth client credentials"
                ),
            )
        return {**headers, "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        if not api_base:
            raise DatabricksException(
                status_code=400,
                message=(
                    "api_base is required for databricks_agent: the workspace URL, the Databricks App URL, "
                    "or the full endpoint URL (set it or the DATABRICKS_API_BASE environment variable)"
                ),
            )
        return resolve_endpoint_url(api_base, model)

    def transform_request(
        self,
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: mirrors override contract
        passthrough: Final = {key: optional_params[key] for key in _PASSTHROUGH_PARAMS if key in optional_params}
        stream: Final = {"stream": True} if optional_params.get("stream") is True else {}
        drop_unsupported_content: Final = _drop_params_enabled(litellm_params)
        return {
            **_body_model_field(litellm_params.get("api_base"), model),
            "input": [_input_item(message, drop_unsupported_content) for message in messages],
            **stream,
            **passthrough,
        }

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: Mapping[str, object],
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )
        try:
            agent_response: Final = AgentResponse.model_validate_json(raw_response.content)
        except ValidationError as e:
            raise DatabricksException(
                status_code=502,
                message=f"Databricks agent returned a response that is not a ResponsesAgent response: {e}",
            )

        content: Final = output_text(agent_response.output)
        message: Final = Message(
            content=content,
            role="assistant",
            provider_specific_fields=(
                {"custom_outputs": agent_response.custom_outputs} if agent_response.custom_outputs is not None else None
            ),
        )
        return ModelResponse(
            id=model_response.id,
            created=model_response.created,
            model=model,
            choices=[Choices(finish_reason="stop", index=0, message=message)],
            usage=agent_response.usage.to_usage()
            if agent_response.usage is not None
            else self._estimated_usage(model, messages, content),
        )

    @staticmethod
    def _estimated_usage(model: str, messages: Sequence[AllMessageValues], content: str) -> Usage:
        from litellm.utils import token_counter

        prompt_tokens: Final = token_counter(model=model, messages=list(messages))
        completion_tokens: Final = token_counter(model=model, text=content, count_response_tokens=True)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> DatabricksAgentResponsesIterator:
        return DatabricksAgentResponsesIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers
    ) -> BaseLLMException:
        return DatabricksException(status_code=status_code, message=error_message)
