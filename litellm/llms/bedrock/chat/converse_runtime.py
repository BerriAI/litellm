"""Bedrock Runtime transport for the Converse route, driven by the shared ``BaseLLMHTTPHandler``.

``AmazonConverseConfig`` stays transport-free because the ``converse_like`` route reuses it against
endpoints that are not AWS and are never SigV4-signed.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import httpx

from litellm.litellm_core_utils.get_litellm_params import stored_control_options, with_aws_credential_fallbacks
from litellm.litellm_core_utils.logging_utils import track_llm_api_timing
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, SignsRequestsWithAWS
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.bedrock.chat.invoke_handler import make_call, make_sync_call
from litellm.llms.bedrock.common_utils import get_all_bedrock_regions
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.bedrock import AWS_AUTH_PARAM_KEYS
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

_AWS: Final = BaseAWSLLM()
_ROUTE_PREFIXES: Final = ("bedrock/converse/", "bedrock/", "converse/")
_MODEL_SPEC_PREFIXES: Final = ("nova-2/", "nova/")
_PARAMS_SENT_OUTSIDE_THE_BODY: Final = frozenset(
    {*AWS_AUTH_PARAM_KEYS, "aws_region_name", "aws_bedrock_runtime_endpoint", "model_id", "stream", "fake_stream"}
)


@dataclass(frozen=True, slots=True)
class _ModelPath:
    model_id: str
    region: str | None


def _model_path(model: str) -> _ModelPath:
    without_route: Final = next((model[len(prefix) :] for prefix in _ROUTE_PREFIXES if model.startswith(prefix)), model)
    head, separator, tail = without_route.partition("/")
    region: Final = head if separator and head in get_all_bedrock_regions() else None
    after_region: Final = without_route if region is None else tail
    model_id: Final = model if region is None else tail
    spec: Final = next((prefix for prefix in _MODEL_SPEC_PREFIXES if after_region.startswith(prefix)), None)
    return _ModelPath(model_id=model_id if spec is None else model_id.replace(spec, "", 1), region=region)


def _explicit_model_id(params: Mapping[str, object]) -> str | None:
    model_id: Final = params.get("model_id")
    return model_id if isinstance(model_id, str) else None


def _region(model: str, params: Mapping[str, object]) -> str:
    model_id: Final = _explicit_model_id(params)
    path_region: Final = None if model_id is not None else _model_path(model).region
    region_params: Final = (
        {**params, "aws_region_name": path_region}
        if path_region is not None and "aws_region_name" not in params
        else params
    )
    return _AWS._get_aws_region_name(optional_params=dict(region_params), model=model, model_id=model_id)


def _body_params(optional_params: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in optional_params.items() if key not in _PARAMS_SENT_OUTSIDE_THE_BODY}


def _signed_body_text(data: Mapping[str, object], signed_json_body: bytes | None) -> str:
    return json.dumps(data) if signed_json_body is None else signed_json_body.decode()


class AmazonConverseRuntimeConfig(AmazonConverseConfig, SignsRequestsWithAWS):
    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: bool | None = None,
    ) -> str:
        params: Final = with_aws_credential_fallbacks(optional_params, litellm_params)
        runtime_endpoint: Final = params.get("aws_bedrock_runtime_endpoint")
        _, endpoint_url = _AWS.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=runtime_endpoint if isinstance(runtime_endpoint, str) else None,
            aws_region_name=_region(model, params),
        )
        model_id: Final = BaseAWSLLM.encode_model_id(_explicit_model_id(params) or _model_path(model).model_id)
        action: Final = "converse-stream" if stream is True else "converse"
        return join_api_base(endpoint_url, f"model/{model_id}/{action}")

    def sign_request(
        self,
        headers: dict[str, object],
        optional_params: dict[str, object],
        request_data: dict[str, object],
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict[str, object], bytes | None]:
        return _AWS._sign_request(
            service_name="bedrock",
            headers=headers,
            optional_params={**optional_params, "aws_region_name": _region(model or "", optional_params)},
            request_data=request_data,
            api_base=api_base,
            api_key=api_key,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        headers: dict[str, object],
    ) -> dict[str, object]:
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=_body_params(optional_params),
            litellm_params=litellm_params,
            headers=headers,
        )

    async def async_transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        headers: dict[str, object],
    ) -> dict[str, object]:
        return cast(
            "dict[str, object]",
            await self._async_transform_request(
                model=model,
                messages=messages,
                optional_params=_body_params(optional_params),
                litellm_params=litellm_params,
                headers=headers,
            ),
        )

    @property
    def uses_async_transform_request(self) -> bool:
        return True

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict[str, object],
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        response: Final = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
        response.set_provider_response_headers(raw_response.headers)
        return response

    @property
    def has_custom_stream_wrapper(self) -> bool:
        return True

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        return False

    @track_llm_api_timing()
    async def get_async_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: "LiteLLMLoggingObj",
        api_base: str,
        headers: dict[str, object],
        data: dict[str, object],
        messages: list[AllMessageValues],
        client: AsyncHTTPHandler | None = None,
        json_mode: bool | None = None,
        signed_json_body: bytes | None = None,
        *,
        litellm_params: Mapping[str, object],
        timeout: float | httpx.Timeout | None = None,
    ) -> CustomStreamWrapper:
        completion_stream, response_headers = await make_call(
            client=client,
            api_base=api_base,
            headers=headers,
            data=_signed_body_text(data, signed_json_body),
            model=model,
            messages=messages,
            logging_obj=logging_obj,
            json_mode=json_mode,
            stream_chunk_size=stored_control_options(litellm_params).stream_chunk_size,
            timeout=timeout,
        )
        return CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
            _response_headers=response_headers,
        )

    @track_llm_api_timing()
    def get_sync_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: "LiteLLMLoggingObj",
        api_base: str,
        headers: dict[str, object],
        data: dict[str, object],
        messages: list[AllMessageValues],
        client: HTTPHandler | AsyncHTTPHandler | None = None,
        json_mode: bool | None = None,
        signed_json_body: bytes | None = None,
        *,
        litellm_params: Mapping[str, object],
        timeout: float | httpx.Timeout | None = None,
    ) -> CustomStreamWrapper:
        completion_stream, response_headers = make_sync_call(
            client=client if isinstance(client, HTTPHandler) else None,
            api_base=api_base,
            headers=headers,
            data=_signed_body_text(data, signed_json_body),
            signed_json_body=signed_json_body,
            model=model,
            messages=messages,
            logging_obj=logging_obj,
            json_mode=json_mode,
            stream_chunk_size=stored_control_options(litellm_params).stream_chunk_size,
            timeout=timeout,
        )
        return CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
            _response_headers=response_headers,
        )
