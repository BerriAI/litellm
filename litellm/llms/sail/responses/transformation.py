from collections.abc import Mapping
from typing import Final

from litellm.llms.openai_like.dynamic_config import create_responses_config_class
from litellm.llms.sail.common_utils import (
    extra_body_for_sail,
    json_body,
    responses_params_with_completion_window,
    sail_provider_config,
    without_keys,
)
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams


class SailResponsesAPIConfig(create_responses_config_class(sail_provider_config())):
    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: return type fixed by the base interface
        params: Final = responses_params_with_completion_window(
            super().map_openai_params(
                response_api_optional_params=response_api_optional_params, model=model, drop_params=drop_params
            ),
            model=model,
            drop_params=drop_params,
        )
        return json_body(params)

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict,  # mutable-ok: signature fixed by the base interface
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: signature fixed by the base interface
    ) -> dict:  # mutable-ok: return type fixed by the base interface
        request: Final[Mapping[str, object]] = super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        return json_body(without_keys(request, frozenset({"service_tier"})))

    def transform_extra_body(
        self,
        extra_body: Mapping[str, object],
        request: Mapping[str, object],
        model: str,
        litellm_params: GenericLiteLLMParams,
    ) -> Mapping[str, object]:
        return extra_body_for_sail(
            extra_body, request.get("metadata"), model=model, drop_params=bool(litellm_params.drop_params)
        )
