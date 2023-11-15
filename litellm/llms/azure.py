from typing import Optional, Union
import types, requests
from .base import BaseLLM
from litellm.utils import ModelResponse, Choices, Message, CustomStreamWrapper, convert_to_model_response_object
from typing import Callable, Optional
from litellm import OpenAIConfig
import httpx

class AzureOpenAIError(Exception):
    def __init__(self, status_code, message, request: Optional[httpx.Request]=None, response: Optional[httpx.Response]=None):
        self.status_code = status_code
        self.message = message
        if request:
            self.request = request
        else:
            self.request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        if response:
            self.response = response
        else:
            self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class AzureOpenAIConfig(OpenAIConfig):
    """
    Reference: https://platform.openai.com/docs/api-reference/chat/create

    The class `AzureOpenAIConfig` provides configuration for the OpenAI's Chat API interface, for use with Azure. It inherits from `OpenAIConfig`. Below are the parameters::

    - `frequency_penalty` (number or null): Defaults to 0. Allows a value between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, thereby minimizing repetition.

    - `function_call` (string or object): This optional parameter controls how the model calls functions.

    - `functions` (array): An optional parameter. It is a list of functions for which the model may generate JSON inputs.

    - `logit_bias` (map): This optional parameter modifies the likelihood of specified tokens appearing in the completion.

    - `max_tokens` (integer or null): This optional parameter helps to set the maximum number of tokens to generate in the chat completion.

    - `n` (integer or null): This optional parameter helps to set how many chat completion choices to generate for each input message.

    - `presence_penalty` (number or null): Defaults to 0. It penalizes new tokens based on if they appear in the text so far, hence increasing the model's likelihood to talk about new topics.

    - `stop` (string / array / null): Specifies up to 4 sequences where the API will stop generating further tokens.

    - `temperature` (number or null): Defines the sampling temperature to use, varying between 0 and 2.

    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling. 
    """

    def __init__(self, 
                 frequency_penalty: Optional[int] = None, 
                 function_call: Optional[Union[str, dict]]= None, 
                 functions: Optional[list]= None, 
                 logit_bias: Optional[dict]= None, 
                 max_tokens: Optional[int]= None, 
                 n: Optional[int]= None, 
                 presence_penalty: Optional[int]= None, 
                 stop: Optional[Union[str,list]]=None, 
                 temperature: Optional[int]= None, 
                 top_p: Optional[int]= None) -> None:
        super().__init__(frequency_penalty, 
                         function_call, 
                         functions, 
                         logit_bias, 
                         max_tokens, 
                         n, 
                         presence_penalty, 
                         stop, 
                         temperature, 
                         top_p)

class AzureChatCompletion(BaseLLM):
    _client_session: Optional[httpx.Client] = None
    _aclient_session: Optional[httpx.AsyncClient] = None

    def __init__(self) -> None:
        super().__init__()
    
    def validate_environment(self, api_key, azure_ad_token):
        headers = {
            "content-type": "application/json",
        }
        if api_key is not None:
            headers["api-key"] = api_key
        if azure_ad_token is not None:
            headers["Authorization"] = f"Bearer {azure_ad_token}"
        return headers

    def completion(self, 
               model: str,
               messages: list,
               model_response: ModelResponse,
               api_key: str,
               api_base: str,
               api_version: str,
               api_type: str,
               azure_ad_token: str,
               print_verbose: Callable,
               logging_obj,
               optional_params,
               litellm_params,
               logger_fn,
               acompletion: bool = False,
               headers: Optional[dict]=None):
        super().completion()
        if self._client_session is None: 
            self._client_session = self.create_client_session()
        exception_mapping_worked = False
        try:
            if headers is None:
                headers = self.validate_environment(api_key=api_key, azure_ad_token=azure_ad_token)

            if model is None or messages is None:
                raise AzureOpenAIError(status_code=422, message=f"Missing model or messages")
            # Ensure api_base ends with a trailing slash
            if not api_base.endswith('/'):
                api_base += '/'

            api_base = api_base + f"openai/deployments/{model}/chat/completions?api-version={api_version}"
            data = {
                "messages": messages, 
                **optional_params
            }
            ## LOGGING
            logging_obj.pre_call(
                input=messages,
                api_key=api_key,
                additional_args={
                    "headers": headers,
                    "api_version": api_version,
                    "api_base": api_base,
                    "complete_input_dict": data,
                },
            )
            if acompletion is True: 
                if optional_params.get("stream", False):
                    return self.async_streaming(logging_obj=logging_obj, api_base=api_base, data=data, headers=headers, model_response=model_response, model=model)
                else:
                    return self.acompletion(api_base=api_base, data=data, headers=headers, model_response=model_response)
            elif "stream" in optional_params and optional_params["stream"] == True:
                return self.streaming(logging_obj=logging_obj, api_base=api_base, data=data, headers=headers, model_response=model_response, model=model)
            else:
                response = self._client_session.post(
                    url=api_base,
                    json=data,
                    headers=headers,
                )
                if response.status_code != 200:
                    raise AzureOpenAIError(status_code=response.status_code, message=response.text)
                    
                ## RESPONSE OBJECT
                return convert_to_model_response_object(response_object=response.json(), model_response_object=model_response)
        except AzureOpenAIError as e: 
            exception_mapping_worked = True
            raise e
        except Exception as e: 
            raise e
    
    async def acompletion(self, api_base: str, data: dict, headers: dict, model_response: ModelResponse): 
       if self._aclient_session is None:
           self._aclient_session = self.create_aclient_session()
       client = self._aclient_session
       try:
            response = await client.post(api_base, json=data, headers=headers) 
            response_json = response.json()
            if response.status_code != 200:
                raise AzureOpenAIError(status_code=response.status_code, message=response.text, request=response.request, response=response)
            
            ## RESPONSE OBJECT
            return convert_to_model_response_object(response_object=response_json, model_response_object=model_response)
       except Exception as e: 
           if isinstance(e,httpx.TimeoutException):
                raise AzureOpenAIError(status_code=500, message="Request Timeout Error")
           elif response and hasattr(response, "text"):
                raise AzureOpenAIError(status_code=500, message=f"{str(e)}\n\nOriginal Response: {response.text}")
           else: 
                raise AzureOpenAIError(status_code=500, message=f"{str(e)}")

    def streaming(self,
                  logging_obj,
                  api_base: str, 
                  data: dict, 
                  headers: dict, 
                  model_response: ModelResponse, 
                  model: str
    ):
        if self._client_session is None:
            self._client_session = self.create_client_session()
        with self._client_session.stream(
                    url=f"{api_base}",
                    json=data,
                    headers=headers,
                    method="POST"
                ) as response: 
                    if response.status_code != 200:
                        raise AzureOpenAIError(status_code=response.status_code, message=response.text)
                    
                    completion_stream = response.iter_lines()
                    streamwrapper = CustomStreamWrapper(completion_stream=completion_stream, model=model, custom_llm_provider="azure",logging_obj=logging_obj)
                    for transformed_chunk in streamwrapper:
                        yield transformed_chunk

    async def async_streaming(self, 
                          logging_obj,
                          api_base: str, 
                          data: dict, 
                          headers: dict, 
                          model_response: ModelResponse, 
                          model: str):
        if self._aclient_session is None:
           self._aclient_session = self.create_aclient_session()
        client = self._aclient_session
        async with client.stream(
                    url=f"{api_base}",
                    json=data,
                    headers=headers,
                    method="POST"
                ) as response: 
            if response.status_code != 200:
                raise AzureOpenAIError(status_code=response.status_code, message=response.text)
            
            streamwrapper = CustomStreamWrapper(completion_stream=response.aiter_lines(), model=model, custom_llm_provider="azure",logging_obj=logging_obj)
            async for transformed_chunk in streamwrapper:
                yield transformed_chunk

    def embedding(self,
                model: str,
                input: list,
                api_key: str,
                api_base: str,
                azure_ad_token: str,
                api_version: str,
                logging_obj=None,
                model_response=None,
                optional_params=None,):
        super().embedding()
        exception_mapping_worked = False
        if self._client_session is None:
            self._client_session = self.create_client_session()
        try: 
            headers = self.validate_environment(api_key, azure_ad_token=azure_ad_token)
            # Ensure api_base ends with a trailing slash
            if not api_base.endswith('/'):
                api_base += '/'

            api_base = api_base + f"openai/deployments/{model}/embeddings?api-version={api_version}"
            model = model
            data = {
                "model": model,
                "input": input,
                **optional_params
            }

            ## LOGGING
            logging_obj.pre_call(
                    input=input,
                    api_key=api_key,
                    additional_args={"complete_input_dict": data},
                )
            ## COMPLETION CALL
            response = self._client_session.post(
                api_base, headers=headers, json=data
            )
            ## LOGGING
            logging_obj.post_call(
                    input=input,
                    api_key=api_key,
                    additional_args={"complete_input_dict": data},
                    original_response=response,
                )

            if response.status_code!=200:
                raise AzureOpenAIError(message=response.text, status_code=response.status_code)
            embedding_response = response.json() 
            output_data = []
            for idx, embedding in enumerate(embedding_response["data"]):
                output_data.append(
                    {
                        "object": embedding["object"],
                        "index": embedding["index"],
                        "embedding": embedding["embedding"]
                    }
                )
            model_response["object"] = "list"
            model_response["data"] = output_data
            model_response["model"] = model
            model_response["usage"] = embedding_response["usage"]
            return model_response
        except AzureOpenAIError as e: 
            exception_mapping_worked = True
            raise e
        except Exception as e: 
            if exception_mapping_worked: 
                raise e
            else: 
                import traceback
                raise AzureOpenAIError(status_code=500, message=traceback.format_exc())