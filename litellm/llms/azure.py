from typing import Optional, Union
import types, requests
from .base import BaseLLM
from litellm.utils import ModelResponse, Choices, Message, CustomStreamWrapper
from typing import Callable, Optional
from litellm import OpenAIConfig
import aiohttp

class AzureOpenAIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
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
    _client_session: requests.Session

    def __init__(self) -> None:
        super().__init__()
        self._client_session = self.create_client_session()
    
    def validate_environment(self, api_key):
        headers = {
            "content-type": "application/json",
        }
        if api_key:
            headers["api-key"] = api_key
        return headers
    
    def convert_to_model_response_object(self, response_object: Optional[dict]=None, model_response_object: Optional[ModelResponse]=None):
        try: 
            if response_object is None or model_response_object is None:
                raise AzureOpenAIError(status_code=500, message="Error in response object format")
            choice_list=[]
            for idx, choice in enumerate(response_object["choices"]): 
                message = Message(content=choice["message"]["content"], role=choice["message"]["role"])
                choice = Choices(finish_reason=choice["finish_reason"], index=idx, message=message)
                choice_list.append(choice)
            model_response_object.choices = choice_list

            if "usage" in response_object: 
                model_response_object.usage = response_object["usage"]
            
            if "id" in response_object: 
                model_response_object.id = response_object["id"]
            
            if "model" in response_object: 
                model_response_object.model = response_object["model"]
            return model_response_object
        except: 
            AzureOpenAIError(status_code=500, message="Invalid response object.")

    def completion(self, 
               model: str,
               messages: list,
               model_response: ModelResponse,
               api_key: str,
               api_base: str,
               api_version: str,
               api_type: str,
               print_verbose: Callable,
               logging_obj,
               optional_params,
               litellm_params,
               logger_fn,
               acompletion: bool = False,
               headers: Optional[dict]=None):
        super().completion()
        exception_mapping_worked = False
        try:
            if headers is None:
                headers = self.validate_environment(api_key=api_key)

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
                },
            )
            if acompletion is True: 
                if optional_params.get("stream", False):
                    return self.async_streaming(logging_obj=logging_obj, api_base=api_base, data=data, headers=headers, model_response=model_response, model=model)
                else:
                    return self.acompletion(api_base=api_base, data=data, headers=headers, model_response=model_response)
            elif "stream" in optional_params and optional_params["stream"] == True:
                response = self._client_session.post(
                    url=api_base,
                    json=data,
                    headers=headers,
                    stream=optional_params["stream"]
                )
                if response.status_code != 200:
                    raise AzureOpenAIError(status_code=response.status_code, message=response.text)
                    
                ## RESPONSE OBJECT
                return response.iter_lines()
            else:
                response = self._client_session.post(
                    url=api_base,
                    json=data,
                    headers=headers,
                )
                if response.status_code != 200:
                    raise AzureOpenAIError(status_code=response.status_code, message=response.text)
                    
                ## RESPONSE OBJECT
                return self.convert_to_model_response_object(response_object=response.json(), model_response_object=model_response)
        except AzureOpenAIError as e: 
            exception_mapping_worked = True
            raise e
        except Exception as e: 
            if exception_mapping_worked: 
                raise e
            else: 
                import traceback
                raise AzureOpenAIError(status_code=500, message=traceback.format_exc())
    
    async def acompletion(self, api_base: str, data: dict, headers: dict, model_response: ModelResponse): 
        async with aiohttp.ClientSession() as session:
            async with session.post(api_base, json=data, headers=headers) as response:
                response_json = await response.json()
                if response.status != 200:
                    raise AzureOpenAIError(status_code=response.status, message=response.text)
                

                ## RESPONSE OBJECT
                return self.convert_to_model_response_object(response_object=response_json, model_response_object=model_response)

    async def async_streaming(self, 
                          logging_obj,
                          api_base: str, 
                          data: dict, headers: dict, 
                          model_response: ModelResponse, 
                          model: str):
        async with aiohttp.ClientSession() as session:
            async with session.post(api_base, json=data, headers=headers) as response:
                # Check if the request was successful (status code 200)
                if response.status != 200:
                    raise AzureOpenAIError(status_code=response.status, message=await response.text())
                
                # Handle the streamed response
                stream_wrapper = CustomStreamWrapper(completion_stream=response, model=model, custom_llm_provider="azure",logging_obj=logging_obj)
                async for transformed_chunk in stream_wrapper:
                    yield transformed_chunk

    def embedding(self,
                model: str,
                input: list,
                api_key: str,
                api_base: str,
                api_version: str,
                logging_obj=None,
                model_response=None,
                optional_params=None,):
        super().embedding()
        exception_mapping_worked = False
        try: 
            headers = self.validate_environment(api_key)
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