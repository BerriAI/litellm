from typing import Callable, Optional, Any
from .base import BaseLLM
from litellm.utils import (
    ModelResponse,
    CustomStreamWrapper,
    convert_to_model_response_object,
)
import litellm
import httpx
from zhipuai import ZhipuAI


class ZhipuAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
    ):
        self.status_code = status_code
        self.message = message
        if request:
            self.request = request
        else:
            self.request = httpx.Request(method="POST", url="https://open.bigmodel.cn/api/paas/v4")
        if response:
            self.response = response
        else:
            self.response = httpx.Response(
                status_code=status_code, request=self.request
            )
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class ZhipuAICompletion(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        api_key: str,
        api_base: str,
        print_verbose: Callable,
        timeout,
        logging_obj,
        optional_params,
        litellm_params,
        logger_fn,
        acompletion: bool = False,
        headers: Optional[dict] = None,
        client=None,
    ):
        super().completion()
        exception_mapping_worked = False
        try:
            if model is None or messages is None:
                raise ZhipuAIError(
                    status_code=422, message=f"Missing model or messages"
                )

            max_retries = optional_params.pop("max_retries", 2)

            del optional_params["user"]
            data = {
                "model": model,  # type: ignore
                "messages": messages,
                **optional_params,
            }

            if "stream" in optional_params and optional_params["stream"] == True:
                return self.streaming(
                    logging_obj=logging_obj,
                    api_base=api_base,
                    data=data,
                    model=model,
                    api_key=api_key,
                    timeout=timeout,
                    client=client,
                )
            else:
                ## LOGGING
                logging_obj.pre_call(
                    input=messages,
                    api_key=api_key,
                    additional_args={
                        "headers": {
                            "api_key": api_key,
                        },
                        "api_base": api_base,
                        "complete_input_dict": data,
                    },
                )
                if not isinstance(max_retries, int):
                    raise ZhipuAIError(
                        status_code=422, message="max retries must be an int"
                    )
                # init ZhipuAI Client
                zhipuai_client_params = {
                    "base_url": api_base,
                    "http_client": litellm.client_session,
                    "max_retries": max_retries,
                    "timeout": timeout,
                }
                if api_key is not None:
                    zhipuai_client_params["api_key"] = api_key
                if client is None:
                    zhipuai_client = ZhipuAI(**zhipuai_client_params)
                else:
                    zhipuai_client = client
                response = zhipuai_client.chat.completions.create(**data, timeout=timeout)  # type: ignore
                stringified_response = response.model_dump()
                ## LOGGING
                logging_obj.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=stringified_response,
                    additional_args={
                        "headers": headers,
                        "api_base": api_base,
                    },
                )
                return convert_to_model_response_object(
                    response_object=stringified_response,
                    model_response_object=model_response,
                )
        except ZhipuAIError as e:
            exception_mapping_worked = True
            raise e
        except Exception as e:
            if hasattr(e, "status_code"):
                raise ZhipuAIError(status_code=e.status_code, message=str(e))
            else:
                raise ZhipuAIError(status_code=500, message=str(e))

    def streaming(
        self,
        logging_obj,
        api_base: str,
        api_key: str,
        data: dict,
        model: str,
        timeout: Any,
        client=None,
    ):
        max_retries = data.pop("max_retries", 2)
        if not isinstance(max_retries, int):
            raise ZhipuAIError(
                status_code=422, message="max retries must be an int"
            )
        # init ZhipuAI Client
        zhipuai_client_params = {
            "base_url": api_base,
            "http_client": litellm.client_session,
            "max_retries": max_retries,
            "timeout": timeout,
        }
        if api_key is not None:
            zhipuai_client_params["api_key"] = api_key
        if client is None:
            zhipuai_client = ZhipuAI(**zhipuai_client_params)
        else:
            zhipuai_client = client
        ## LOGGING
        logging_obj.pre_call(
            input=data["messages"],
            api_key=zhipuai_client.api_key,
            additional_args={
                "headers": {"api_key": api_key},
                "api_base": api_base,
                "acompletion": True,
                "complete_input_dict": data,
            },
        )
        response = zhipuai_client.chat.completions.create(**data, timeout=timeout)
        streamwrapper = CustomStreamWrapper(
            completion_stream=response,
            model=model,
            custom_llm_provider="zhipuai",
            logging_obj=logging_obj,
        )
        return streamwrapper

    def embedding(
        self,
        model: str,
        input: list,
        api_key: str,
        api_base: str,
        timeout: float,
        logging_obj=None,
        model_response=None,
        optional_params=None,
        client=None,
        aembedding=None,
    ):
        super().embedding()
        exception_mapping_worked = False
        if self._client_session is None:
            self._client_session = self.create_client_session()
        try:
            data = {"model": model, "input": input, **optional_params}
            max_retries = data.pop("max_retries", 2)
            if not isinstance(max_retries, int):
                raise ZhipuAIError(
                    status_code=422, message="max retries must be an int"
                )

            # init ZhipuAI Client
            zhipuai_client_params = {
                "base_url": api_base,
                "http_client": litellm.client_session,
                "max_retries": max_retries,
                "timeout": timeout,
            }
            if api_key is not None:
                zhipuai_client_params["api_key"] = api_key

            ## LOGGING
            logging_obj.pre_call(
                input=input,
                api_key=api_key,
                additional_args={
                    "complete_input_dict": data,
                    "headers": {"api_key": api_key},
                },
            )

            if client is None:
                zhipuai_client = ZhipuAI(**zhipuai_client_params)
            else:
                zhipuai_client = client
            ## COMPLETION CALL
            response = zhipuai_client.embeddings.create(**data, timeout=timeout)  # type: ignore
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=response,
            )

            return convert_to_model_response_object(response_object=response.model_dump(), model_response_object=model_response, response_type="embedding")  # type: ignore
        except ZhipuAIError as e:
            exception_mapping_worked = True
            raise e
        except Exception as e:
            if hasattr(e, "status_code"):
                raise ZhipuAIError(status_code=e.status_code, message=str(e))
            else:
                raise ZhipuAIError(status_code=500, message=str(e))

    def image_generation(
        self,
        prompt: str,
        timeout: float,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_response: Optional[litellm.utils.ImageResponse] = None,
        logging_obj=None,
        client=None,
        aimg_generation=None,
    ):
        exception_mapping_worked = False
        try:
            if model and len(model) > 0:
                model = model
            else:
                model = None
            data = {"model": model, "prompt": prompt}
            max_retries = data.pop("max_retries", 2)
            if not isinstance(max_retries, int):
                raise ZhipuAIError(
                    status_code=422, message="max retries must be an int"
                )

            # init ZhipuAI Client
            zhipuai_client_params = {
                "base_url": api_base,
                "http_client": litellm.client_session,
                "max_retries": max_retries,
                "timeout": timeout,
            }
            if api_key is not None:
                zhipuai_client_params["api_key"] = api_key

            if client is None:
                zhipuai_client = ZhipuAI(**zhipuai_client_params)
            else:
                zhipuai_client = client

            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=zhipuai_client.api_key,
                additional_args={
                    "headers": {"api_key": api_key},
                    "api_base": api_base,
                    "acompletion": False,
                    "complete_input_dict": data,
                },
            )

            ## COMPLETION CALL
            response = zhipuai_client.images.generations(**data, timeout=timeout)  # type: ignore
            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=response,
            )
            # return response
            return convert_to_model_response_object(response_object=response.model_dump(), model_response_object=model_response, response_type="image_generation")  # type: ignore
        except ZhipuAIError as e:
            exception_mapping_worked = True
            raise e
        except Exception as e:
            if hasattr(e, "status_code"):
                raise ZhipuAIError(status_code=e.status_code, message=str(e))
            else:
                raise ZhipuAIError(status_code=500, message=str(e))
