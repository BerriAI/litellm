"""Databricks ResponsesAgent endpoints.

Model Serving: POST {workspace}/serving-endpoints/{endpoint}/invocations
Model Serving, unified: POST {workspace}/serving-endpoints/responses with the endpoint as the body "model"
Databricks Apps: POST {app}/responses
"""

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Final
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, StrictStr, TypeAdapter, ValidationError

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
from litellm.llms.databricks.common_utils import DatabricksBase, DatabricksException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

PROVIDER_NAME: Final = "databricks_agent"
_MODEL_PREFIX: Final = f"{PROVIDER_NAME}/"
_INVOCATIONS_SUFFIX: Final = "/invocations"
_APP_RESPONSES_SUFFIX: Final = "/responses"
_ENDPOINT_URL_SUFFIXES: Final = (_INVOCATIONS_SUFFIX, _APP_RESPONSES_SUFFIX)
_SERVING_ENDPOINTS_SEGMENT: Final = "/serving-endpoints/"
_UNIFIED_RESPONSES_SUFFIX: Final = "/serving-endpoints/responses"
_APP_HOST_SUFFIX: Final = ".databricksapps.com"
_INPUT_ROLES: Final = frozenset({"system", "developer", "user", "assistant"})
_BODY_PARAMS: Final = (
    "max_output_tokens",
    "temperature",
    "top_p",
    "user",
    "custom_inputs",
    "context",
    "databricks_options",
)
_SAMPLING_PARAMS: Final = ("temperature", "top_p", "user")
_OUTPUT_TOKEN_PARAMS: Final = ("max_completion_tokens", "max_tokens")
_STRING_METADATA: Final = TypeAdapter(Mapping[str, StrictStr])
_METADATA: Final = TypeAdapter(Mapping[str, object])
_PINNED_BODY_FIELDS: Final = ("model", "input")


def endpoint_name_from_model(model: str) -> str:
    return model.removeprefix(_MODEL_PREFIX).strip()


def _normalized_base(api_base: str) -> str:
    return api_base.strip().rstrip("/")


def _is_unified_responses_url(api_base: str) -> bool:
    return _normalized_base(api_base).endswith(_UNIFIED_RESPONSES_SUFFIX)


def _is_databricks_app_host(api_base: str) -> bool:
    return (urlsplit(_normalized_base(api_base)).hostname or "").endswith(_APP_HOST_SUFFIX)


def is_databricks_app_url(api_base: str) -> bool:
    if _is_unified_responses_url(api_base):
        return False
    return _is_databricks_app_host(api_base) or _normalized_base(api_base).endswith(_APP_RESPONSES_SUFFIX)


def workspace_url_from_api_base(api_base: str | None) -> str | None:
    if not api_base or not api_base.strip() or is_databricks_app_url(api_base):
        return None
    parts: Final = urlsplit(_normalized_base(api_base))
    if not parts.scheme or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}"


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


def _app_endpoint_url(app_base: str) -> str:
    return f"{app_base}{_APP_RESPONSES_SUFFIX}" if urlsplit(app_base).path in ("", "/") else app_base


def resolve_endpoint_url(api_base: str, model: str) -> str:
    base: Final = _normalized_base(api_base)
    if _is_unified_responses_url(base):
        _required_endpoint_name(model)
        return base
    if base.endswith(_ENDPOINT_URL_SUFFIXES):
        return base
    if _is_databricks_app_host(base):
        return _app_endpoint_url(base)
    if _SERVING_ENDPOINTS_SEGMENT in base:
        return f"{base}{_INVOCATIONS_SUFFIX}"
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
    tool_calls: tuple[object, ...] | None = None


def _drop_params_enabled(litellm_params: Mapping[str, object]) -> bool:
    return litellm.drop_params is True or litellm_params.get("drop_params") is True


def _input_item(message: AllMessageValues, drop_unsupported: bool) -> Mapping[str, str] | None:
    role: Final = message["role"]
    if role not in _INPUT_ROLES:
        if drop_unsupported:
            return None
        raise DatabricksException(
            status_code=400,
            message=(
                f"message role '{role}' is not supported by a Databricks ResponsesAgent; "
                "send system, developer, user, or assistant messages, or set drop_params: true to drop it"
            ),
        )
    unsupported: Final = _unsupported_message_parts(message)
    if unsupported and not drop_unsupported:
        raise DatabricksException(
            status_code=400,
            message=(
                f"databricks_agent sends text content only; remove {', '.join(unsupported)} from the "
                "message, or set drop_params: true to drop them"
            ),
        )
    content: Final = convert_content_list_to_str(message)
    if unsupported and not content:
        return None
    return {"role": role, "content": content}


def _input_items(messages: Sequence[AllMessageValues], drop_unsupported: bool) -> Sequence[Mapping[str, str]]:
    items: Final = (_input_item(message, drop_unsupported) for message in messages)
    return [item for item in items if item is not None]


def _unsupported_message_parts(message: AllMessageValues) -> tuple[str, ...]:
    parsed: Final = _InputMessage.model_validate(message)
    content_parts: Final = parsed.content if isinstance(parsed.content, tuple) else ()
    content_types: Final[frozenset[str]] = frozenset(part.type for part in content_parts) - {"text"}
    tool_calls: Final[frozenset[str]] = frozenset({"tool_calls"}) if parsed.tool_calls else frozenset()
    return tuple(sorted(content_types | tool_calls))


def _has_authorization(headers: Mapping[str, object]) -> bool:
    return any(name.lower() == "authorization" for name in headers)


def _without_authorization(headers: Mapping[str, object]) -> Mapping[str, object]:
    return {name: value for name, value in headers.items() if name.lower() != "authorization"}


def _env_oauth_m2m_token(api_base: str | None) -> str | None:
    client_id: Final = get_secret_str("DATABRICKS_CLIENT_ID")
    client_secret: Final = get_secret_str("DATABRICKS_CLIENT_SECRET")
    if not (client_id and client_secret):
        return None
    workspace: Final = get_secret_str("DATABRICKS_HOST") or workspace_url_from_api_base(api_base)
    if workspace is None:
        raise DatabricksException(
            status_code=400,
            message=(
                "DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET are set but the OAuth token endpoint is unknown: "
                "set DATABRICKS_HOST to the workspace URL (a Databricks App URL is not a workspace)"
            ),
        )
    return DatabricksBase().get_cached_oauth_m2m_token(workspace, client_id, client_secret)


def _resolved_env_reference(value: str | None) -> str | None:
    if value is not None and value.startswith("os.environ/"):
        return get_secret_str(value)
    return value


def _max_output_tokens(non_default_params: Mapping[str, object]) -> Mapping[str, object]:
    limits: Final = (non_default_params[key] for key in _OUTPUT_TOKEN_PARAMS if non_default_params.get(key) is not None)
    limit: Final = next(limits, None)
    return {} if limit is None else {"max_output_tokens": limit}


def _string_metadata_entries(metadata: object) -> Mapping[str, str]:
    try:
        entries: Final = _METADATA.validate_python(metadata)
    except ValidationError:
        return {}
    return {key: value for key, value in entries.items() if isinstance(value, str)}


def _has_string_values_only(metadata: object) -> bool:
    try:
        _STRING_METADATA.validate_python(metadata)
    except ValidationError:
        return False
    return True


def _without_pinned_fields(extra_body: Mapping[str, object], drop_params: bool) -> Mapping[str, object]:
    pinned: Final = tuple(key for key in _PINNED_BODY_FIELDS if key in extra_body)
    if not pinned:
        return extra_body
    if not drop_params:
        raise DatabricksException(
            status_code=400,
            message=(
                f"extra_body cannot set {', '.join(pinned)} on a Databricks ResponsesAgent request, "
                "since the agent's endpoint and the messages set them; remove them, "
                "or set drop_params: true to drop them"
            ),
        )
    return {key: value for key, value in extra_body.items() if key not in _PINNED_BODY_FIELDS}


def _with_string_metadata(extra_body: Mapping[str, object], drop_params: bool) -> Mapping[str, object]:
    if "metadata" not in extra_body or _has_string_values_only(extra_body["metadata"]):
        return extra_body
    if not drop_params:
        raise DatabricksException(
            status_code=400,
            message=(
                "metadata sent to a Databricks ResponsesAgent must map names to string values; "
                "send string values only, or set drop_params: true to drop the others"
            ),
        )
    kept: Final = _string_metadata_entries(extra_body["metadata"])
    rest: Final = {key: value for key, value in extra_body.items() if key != "metadata"}
    return {**rest, **({"metadata": kept} if kept else {})}


class DatabricksAgentConfig(BaseConfig):
    @staticmethod
    def resolve_api_base_and_key(api_base: str | None, api_key: str | None) -> tuple[str | None, str | None]:
        return (
            _resolved_env_reference(api_base) or get_secret_str("DATABRICKS_API_BASE"),
            _resolved_env_reference(api_key),
        )

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: mirrors override contract
        return ["stream", *_OUTPUT_TOKEN_PARAMS, *_SAMPLING_PARAMS]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: mirrors override contract
        sampling: Final = {key: non_default_params[key] for key in _SAMPLING_PARAMS if key in non_default_params}
        return {**optional_params, **sampling, **_max_output_tokens(non_default_params)}

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
        bearer: Final = api_key or _env_oauth_m2m_token(api_base) or get_secret_str("DATABRICKS_API_KEY")
        if bearer:
            return {
                **_without_authorization(headers),
                "Content-Type": "application/json",
                "Authorization": f"Bearer {bearer}",
            }
        if _has_authorization(headers):
            return {**headers, "Content-Type": "application/json"}
        raise DatabricksException(
            status_code=400,
            message=(
                "Missing Databricks credentials: set api_key to a personal access token, set the "
                "DATABRICKS_API_KEY environment variable, set DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET "
                "for OAuth M2M, or register the agent with OAuth client credentials"
            ),
        )

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
        body_params: Final = {key: optional_params[key] for key in _BODY_PARAMS if key in optional_params}
        stream: Final = {"stream": True} if optional_params.get("stream") is True else {}
        return {
            **_body_model_field(litellm_params.get("api_base"), model),
            "input": _input_items(messages, _drop_params_enabled(litellm_params)),
            **stream,
            **body_params,
        }

    def transform_extra_body(
        self,
        extra_body: Mapping[str, object],
        request: Mapping[str, object],
        model: str,
        litellm_params: Mapping[str, object],
    ) -> Mapping[str, object]:
        drop_params: Final = _drop_params_enabled(litellm_params)
        return _with_string_metadata(_without_pinned_fields(extra_body, drop_params), drop_params)

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
