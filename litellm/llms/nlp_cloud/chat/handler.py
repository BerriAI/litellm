import json
from typing import Callable, Optional, Union

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
)
from litellm.utils import ModelResponse
from litellm.llms.base import BaseLLM

from .transformation import NLPCloudConfig

nlp_config = NLPCloudConfig()



class NLPCloudChatHandler(BaseLLM):
    def __init__(self):
        super().__init__()

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        litellm_params: dict,
        logger_fn=None,
        default_max_tokens_to_sample=None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        headers={},
    ):
        headers = nlp_config.validate_environment(
            api_key=api_key,
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        ## Load Config
        config = litellm.NLPCloudConfig.get_config()
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > togetherai_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        completion_url_fragment_1 = api_base
        completion_url_fragment_2 = "/generation"
        model = model

        completion_url = completion_url_fragment_1 + model + completion_url_fragment_2
        data = nlp_config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=None,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "headers": headers,
                "api_base": completion_url,
            },
        )
        ## COMPLETION CALL
        if client is None:
            client = _get_httpx_client()

        response = client.post(
            completion_url,
            headers=headers,
            data=json.dumps(data),
            stream=optional_params["stream"] if "stream" in optional_params else False,
        )

        if "stream" in optional_params and optional_params["stream"] is True:
            from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

            return CustomStreamWrapper(
                completion_stream=response.iter_lines(),
                model=model,
                custom_llm_provider="nlp_cloud",
                logging_obj=logging_obj,
            )
        else:
            return nlp_config.transform_response(
                model=model,
                raw_response=response,
                model_response=model_response,
                logging_obj=logging_obj,
                api_key=api_key,
                request_data=data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=encoding,
            )

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        litellm_params: dict,
        logger_fn=None,
        default_max_tokens_to_sample=None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        headers={},
    ):
        headers = nlp_config.validate_environment(
            api_key=api_key,
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        ## Load Config
        config = litellm.NLPCloudConfig.get_config()
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > togetherai_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        completion_url_fragment_1 = api_base
        completion_url_fragment_2 = "/generation"
        model = model

        completion_url = completion_url_fragment_1 + model + completion_url_fragment_2
        data = nlp_config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=None,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "headers": headers,
                "api_base": completion_url,
            },
        )
        ## COMPLETION CALL
        if client is None:
            client = AsyncHTTPHandler()

        response = await client.post(
            completion_url,
            headers=headers,
            data=json.dumps(data),
            stream=optional_params["stream"] if "stream" in optional_params else False,
        )

        if "stream" in optional_params and optional_params["stream"] is True:
            from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

            return CustomStreamWrapper(
                completion_stream=response.aiter_lines(),
                model=model,
                custom_llm_provider="nlp_cloud",
                logging_obj=logging_obj,
            )
        else:
            return nlp_config.transform_response(
                model=model,
                raw_response=response,
                model_response=model_response,
                logging_obj=logging_obj,
                api_key=api_key,
                request_data=data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=encoding,
            )

    def embedding(
        self,
        model: str,
        input: list,
        api_key: str,
        api_base: str,
        logging_obj,
        model_response: Optional[ModelResponse] = None,
        optional_params: dict = {},
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ):
        headers = nlp_config.validate_environment(
            api_key=api_key,
            headers={},
            model=model,
            messages=[],
            optional_params=optional_params,
            litellm_params={},
        )
        
        # URL format: https://api.nlpcloud.io/v1/{model}/embeddings
        completion_url = api_base + model + "/embeddings"
        
        data = {
            "sentences": input
        }

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "headers": headers,
                "api_base": completion_url,
            },
        )

        if client is None:
            client = _get_httpx_client()

        response = client.post(
            completion_url,
            headers=headers,
            data=json.dumps(data),
        )

        if response.status_code != 200:
            raise nlp_config.get_error_class(
                error_message=response.text,
                status_code=response.status_code,
                headers=response.headers,
            )

        # Response format: {"embeddings": [[...], [...]]}
        try:
            response_json = response.json()
        except Exception:
             raise nlp_config.get_error_class(
                error_message=response.text,
                status_code=response.status_code,
                headers=response.headers,
            )

        embeddings = response_json.get("embeddings", [])
        
        output_data = []
        for idx, embedding_val in enumerate(embeddings):
            output_data.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding_val,
                }
            )
        
        if model_response is None:
            model_response = litellm.EmbeddingResponse(
                model=model,
                data=output_data,
            )
        else:
            model_response.data = output_data # type: ignore
            model_response.model = model
        
        return model_response
