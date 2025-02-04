## Uses the huggingface text generation inference API
import dataclasses
import json
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Union,
)

from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
from litellm.llms.huggingface.chat.transformation import HFChatConfig
from litellm.types.llms.openai import AllMessageValues, OpenAITextCompletionUserMessage
from litellm.types.utils import TextCompletionResponse

from ...base import BaseLLM
from ..common_utils import HuggingFaceError

if TYPE_CHECKING:
    from huggingface_hub import AsyncInferenceClient, InferenceClient


class HFCompletion(BaseLLM):
    config = HFChatConfig()

    def __init__(self) -> None:
        super().__init__()

    def completion(
        self,
        api_key: str,
        model: str,
        messages: Union[List[AllMessageValues], List[OpenAITextCompletionUserMessage]],
        timeout: float,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        api_base: Optional[str] = None,
        acompletion: bool = False,
        litellm_params=None,
        client=None,
        headers: Optional[dict] = None,
    ):
        super().completion()

        try:
            request_parameters = self.config._prepare_request(
                model=model,
                messages=messages,
                api_base=api_base,
                api_key=api_key,
                headers=headers or {},
                optional_params=optional_params,
                litellm_params=litellm_params,  # type: ignore
            )
            ## LOGGING
            logging_obj.pre_call(
                input=messages,
                api_key=api_key,
                additional_args={
                    "headers": request_parameters.headers,
                    "api_base": request_parameters.base_url,
                    "complete_input_dict": request_parameters.inputs,
                },
            )
            if acompletion is True:
                if optional_params.get("stream", False):
                    return self.async_streaming(
                        api_key=api_key,
                        data=request_parameters.inputs,  # type: ignore
                        headers=request_parameters.headers,
                        timeout=timeout,
                        api_base=request_parameters.base_url,
                        client=client,
                        provider=request_parameters.provider,
                    )
                else:
                    return self.acompletion(
                        logging_obj=logging_obj,
                        api_base=request_parameters.base_url,  # type: ignore
                        data=request_parameters.inputs,  # type: ignore
                        headers=request_parameters.headers,
                        api_key=api_key,
                        timeout=timeout,
                        client=client,
                        provider=request_parameters.provider,
                    )  # type: ignore
            elif optional_params.get("stream", False):
                return self.streaming(
                    api_key=api_key,
                    data=request_parameters.inputs,  # type: ignore
                    headers=request_parameters.headers,
                    timeout=timeout,
                    api_base=request_parameters.base_url,
                    client=client,
                    provider=request_parameters.provider,
                )
            else:
                if client is None:
                    try:
                        from huggingface_hub import InferenceClient
                    except ImportError:
                        raise ImportError(
                        "To use the default Hugging Face's InferenceClient client, please install `huggingface-hub` "
                        "with `pip install huggingface-hub` or `poetry add huggingface-hub`."
                    )
                    hf_client = InferenceClient(
                        api_key=api_key,
                        timeout=timeout,
                        provider=request_parameters.provider,
                        base_url=request_parameters.base_url,
                        headers=request_parameters.headers,
                    )
                else:
                    hf_client = client

                response = hf_client.chat.completions.create(**request_parameters.inputs)  # type: ignore
                response_dict = dataclasses.asdict(response)

                ## LOGGING
                logging_obj.post_call(
                    api_key=api_key,
                    original_response=response_dict,
                    additional_args={
                        "headers": headers,
                        "api_base": api_base,
                    },
                )

                return TextCompletionResponse(**response_dict)
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise HuggingFaceError(
                status_code=status_code, message=error_text, headers=error_headers
            )

    async def acompletion(
        self,
        logging_obj,
        api_base: str,
        data: dict,
        headers: dict,
        api_key: str,
        timeout: float,
        client=None,
        provider: Optional[str] = None,
    ):
        try:
            if client is None:
                try:
                    from huggingface_hub import AsyncInferenceClient
                except ImportError:
                    raise ImportError(
                    "To use the default Hugging Face's AsyncInferenceClient client, please install `huggingface-hub` "
                    "with `pip install huggingface-hub` or `poetry add huggingface-hub`."
                )
                hf_client = AsyncInferenceClient(
                    api_key=api_key,
                    timeout=timeout,
                    base_url=api_base,
                    provider=provider,
                    headers=headers,
                )

            else:
                hf_client = client

            response = await hf_client.chat.completions.create(**data)
            response_dict = dataclasses.asdict(response)

            ## LOGGING
            logging_obj.post_call(
                api_key=api_key,
                original_response=response_dict,
                additional_args={
                    "headers": headers,
                    "api_base": api_base,
                },
            )
            ## RESPONSE OBJECT
            response_obj = TextCompletionResponse(**response_dict)
            response_obj._hidden_params.original_response = json.dumps(response_dict)
            return response_obj
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise HuggingFaceError(
                status_code=status_code, message=error_text, headers=error_headers
            )

    def streaming(
        self,
        api_key: str,
        data: dict,
        headers: dict,
        timeout: float,
        api_base: Optional[str] = None,
        client=None,
        provider=None,
    ):
        if client is None:
            try:
                from huggingface_hub import InferenceClient
            except ImportError:
                raise ImportError(
                    "To use the default Hugging Face's InferenceClient client, please install `huggingface-hub` "
                    "with `pip install huggingface-hub` or `poetry add huggingface-hub`."
                )
            hf_client = InferenceClient(
                api_key=api_key,
                base_url=api_base,
                headers=headers,
                timeout=timeout,
                provider=provider,
            )
        else:
            hf_client = client

        try:
            response_stream = hf_client.chat.completions.create(**data)

        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise HuggingFaceError(
                status_code=status_code, message=error_text, headers=error_headers
            )

        try:
            for chunk in response_stream:
                transformed_chunk = self.config.transform_stream_chunk(chunk)
                yield transformed_chunk
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise HuggingFaceError(
                status_code=status_code, message=error_text, headers=error_headers
            )

    async def async_streaming(
        self,
        api_key: str,
        data: dict,
        headers: dict,
        timeout: float,
        api_base: Optional[str] = None,
        client=None,
        provider=None,
    ):
        if client is None:
            try:
                    from huggingface_hub import AsyncInferenceClient
            except ImportError:
                raise ImportError(
                    "To use the default Hugging Face's AsyncInferenceClient client, please install `huggingface-hub` "
                    "with `pip install huggingface-hub` or `poetry add huggingface-hub`."
                )
            hf_client = AsyncInferenceClient(
                api_key=api_key,
                base_url=api_base,
                headers=headers,
                timeout=timeout,
                provider=provider,
            )
        else:
            hf_client = client

        response_stream = await hf_client.chat.completions.create(**data)

        try:
            async for chunk in response_stream:
                transformed_chunk = self.config.transform_stream_chunk(chunk)
                yield transformed_chunk
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise HuggingFaceError(
                status_code=status_code, message=error_text, headers=error_headers
            )
        finally:
            # Close the client if we created it
            if hf_client is not None and client is None:
                await hf_client.close()
