from typing import Optional, Union, Any
import types, requests
from .base import BaseLLM
from litellm.utils import ModelResponse, Choices, Message, CustomStreamWrapper, convert_to_model_response_object
from typing import Callable, Optional
from litellm import OpenAIConfig
import litellm, json
import httpx
from openai import AzureOpenAI, AsyncAzureOpenAI

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

    def __init__(self) -> None:
        super().__init__()

    def validate_environment(self, api_key, azure_ad_token):
        headers = {
            "content-type": "application/json",
        }
        if api_key is not None:
            headers["api-key"] = api_key
        elif azure_ad_token is not None:
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
               timeout,
               logging_obj,
               optional_params,
               litellm_params,
               logger_fn,
               acompletion: bool = False,
               headers: Optional[dict]=None,
               client = None,
               ):
        super().completion()
        exception_mapping_worked = False
        try:

            if model is None or messages is None:
                raise AzureOpenAIError(status_code=422, message=f"Missing model or messages")
            
            max_retries = optional_params.pop("max_retries", 2)

            ### CHECK IF CLOUDFLARE AI GATEWAY ###
            ### if so - set the model as part of the base url 
            if "gateway.ai.cloudflare.com" in api_base: 
                ## build base url - assume api base includes resource name
                if client is None:
                    if not api_base.endswith("/"): 
                        api_base += "/"
                    api_base += f"{model}"
                    
                    azure_client_params = {
                        "api_version": api_version,
                        "base_url": f"{api_base}",
                        "http_client": litellm.client_session,
                        "max_retries": max_retries,
                        "timeout": timeout
                    }
                    if api_key is not None:
                        azure_client_params["api_key"] = api_key
                    elif azure_ad_token is not None:
                        azure_client_params["azure_ad_token"] = azure_ad_token

                    if acompletion is True:
                        client = AsyncAzureOpenAI(**azure_client_params)
                    else:
                        client = AzureOpenAI(**azure_client_params)
                
                data = {
                    "model": None,
                    "messages": messages, 
                    **optional_params
                }
            else: 
                data = {
                    "model": model, # type: ignore
                    "messages": messages, 
                    **optional_params
                }
            
            if acompletion is True: 
                if optional_params.get("stream", False):
                    return self.async_streaming(logging_obj=logging_obj, api_base=api_base, data=data, model=model, api_key=api_key, api_version=api_version, azure_ad_token=azure_ad_token, timeout=timeout, client=client)
                else:
                    return self.acompletion(api_base=api_base, data=data, model_response=model_response, api_key=api_key, api_version=api_version, model=model, azure_ad_token=azure_ad_token, timeout=timeout, client=client, logging_obj=logging_obj)
            elif "stream" in optional_params and optional_params["stream"] == True:
                return self.streaming(logging_obj=logging_obj, api_base=api_base, data=data, model=model, api_key=api_key, api_version=api_version, azure_ad_token=azure_ad_token, timeout=timeout, client=client)
            else:
                ## LOGGING
                logging_obj.pre_call(
                    input=messages,
                    api_key=api_key,
                    additional_args={
                        "headers": {
                            "api_key": api_key, 
                            "azure_ad_token": azure_ad_token
                        },
                        "api_version": api_version,
                        "api_base": api_base,
                        "complete_input_dict": data,
                    },
                )
                if not isinstance(max_retries, int): 
                    raise AzureOpenAIError(status_code=422, message="max retries must be an int")
                # init AzureOpenAI Client
                azure_client_params = {
                    "api_version": api_version,
                    "azure_endpoint": api_base,
                    "azure_deployment": model,
                    "http_client": litellm.client_session,
                    "max_retries": max_retries,
                    "timeout": timeout
                }
                if api_key is not None:
                    azure_client_params["api_key"] = api_key
                elif azure_ad_token is not None:
                    azure_client_params["azure_ad_token"] = azure_ad_token
                if client is None:
                    azure_client = AzureOpenAI(**azure_client_params)
                else:
                    azure_client = client
                response = azure_client.chat.completions.create(**data) # type: ignore
                stringified_response = response.model_dump_json()
                ## LOGGING
                logging_obj.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=stringified_response,
                    additional_args={
                        "headers": headers,
                        "api_version": api_version,
                        "api_base": api_base,
                    },
                )
                return convert_to_model_response_object(response_object=json.loads(stringified_response), model_response_object=model_response)
        except AzureOpenAIError as e: 
            exception_mapping_worked = True
            raise e
        except Exception as e: 
            raise e
    
    async def acompletion(self, 
                          api_key: str, 
                          api_version: str, 
                          model: str, 
                          api_base: str, 
                          data: dict, 
                          timeout: Any,
                          model_response: ModelResponse,
                          azure_ad_token: Optional[str]=None, 
                          client = None, # this is the AsyncAzureOpenAI
                          logging_obj=None,
                          ): 
       response = None
       try:
            max_retries = data.pop("max_retries", 2)
            if not isinstance(max_retries, int): 
                raise AzureOpenAIError(status_code=422, message="max retries must be an int")
            # init AzureOpenAI Client
            azure_client_params = {
                "api_version": api_version,
                "azure_endpoint": api_base,
                "azure_deployment": model,
                "http_client": litellm.client_session,
                "max_retries": max_retries,
                "timeout": timeout
            }
            if api_key is not None:
                azure_client_params["api_key"] = api_key
            elif azure_ad_token is not None:
                azure_client_params["azure_ad_token"] = azure_ad_token
            if client is None:
                azure_client = AsyncAzureOpenAI(**azure_client_params)
            else:
                azure_client = client
            ## LOGGING
            logging_obj.pre_call(
                input=data['messages'],
                api_key=azure_client.api_key,
                additional_args={"headers": {"Authorization": f"Bearer {azure_client.api_key}"}, "api_base": azure_client._base_url._uri_reference, "acompletion": True, "complete_input_dict": data},
            )
            response = await azure_client.chat.completions.create(**data) 
            return convert_to_model_response_object(response_object=json.loads(response.model_dump_json()), model_response_object=model_response)
       except AzureOpenAIError as e: 
            exception_mapping_worked = True
            raise e
       except Exception as e: 
            raise AzureOpenAIError(status_code=500, message=str(e))

    def streaming(self,
                  logging_obj,
                  api_base: str, 
                  api_key: str,
                  api_version: str, 
                  data: dict, 
                  model: str,
                  timeout: Any,
                  azure_ad_token: Optional[str]=None, 
                  client=None,
    ): 
        max_retries = data.pop("max_retries", 2)
        if not isinstance(max_retries, int): 
            raise AzureOpenAIError(status_code=422, message="max retries must be an int")
        # init AzureOpenAI Client
        azure_client_params = {
            "api_version": api_version,
            "azure_endpoint": api_base,
            "azure_deployment": model,
            "http_client": litellm.client_session,
            "max_retries": max_retries,
            "timeout": timeout
        }
        if api_key is not None:
            azure_client_params["api_key"] = api_key
        elif azure_ad_token is not None:
            azure_client_params["azure_ad_token"] = azure_ad_token
        if client is None:
            azure_client = AzureOpenAI(**azure_client_params)
        else:
            azure_client = client
        ## LOGGING
        logging_obj.pre_call(
            input=data['messages'],
            api_key=azure_client.api_key,
            additional_args={"headers": {"Authorization": f"Bearer {azure_client.api_key}"}, "api_base": azure_client._base_url._uri_reference, "acompletion": True, "complete_input_dict": data},
        )
        response = azure_client.chat.completions.create(**data)
        streamwrapper = CustomStreamWrapper(completion_stream=response, model=model, custom_llm_provider="azure",logging_obj=logging_obj)
        return streamwrapper

    async def async_streaming(self, 
                          logging_obj,
                          api_base: str, 
                          api_key: str, 
                          api_version: str, 
                          data: dict, 
                          model: str,
                          timeout: Any,
                          azure_ad_token: Optional[str]=None,
                          client = None,
                          ):
        # init AzureOpenAI Client
        azure_client_params = {
            "api_version": api_version,
            "azure_endpoint": api_base,
            "azure_deployment": model,
            "http_client": litellm.client_session,
            "max_retries": data.pop("max_retries", 2),
            "timeout": timeout
        }
        if api_key is not None:
            azure_client_params["api_key"] = api_key
        elif azure_ad_token is not None:
            azure_client_params["azure_ad_token"] = azure_ad_token
        if client is None:
                azure_client = AsyncAzureOpenAI(**azure_client_params)
        else:
            azure_client = client
        ## LOGGING
        logging_obj.pre_call(
            input=data['messages'],
            api_key=azure_client.api_key,
            additional_args={"headers": {"Authorization": f"Bearer {azure_client.api_key}"}, "api_base": azure_client._base_url._uri_reference, "acompletion": True, "complete_input_dict": data},
        )
        response = await azure_client.chat.completions.create(**data)
        streamwrapper = CustomStreamWrapper(completion_stream=response, model=model, custom_llm_provider="azure",logging_obj=logging_obj)
        async for transformed_chunk in streamwrapper:
            yield transformed_chunk

    async def aembedding(
        self, 
        data: dict, 
        model_response: ModelResponse, 
        azure_client_params: dict,
        api_key: str, 
        input: list, 
        client=None,
        logging_obj=None
    ): 
        response = None
        try: 
            if client is None:
                openai_aclient = AsyncAzureOpenAI(**azure_client_params)
            else:
                openai_aclient = client
            response = await openai_aclient.embeddings.create(**data)
            stringified_response = response.model_dump_json()
            ## LOGGING
            logging_obj.post_call(
                    input=input,
                    api_key=api_key,
                    additional_args={"complete_input_dict": data},
                    original_response=stringified_response,
                )
            return convert_to_model_response_object(response_object=json.loads(stringified_response), model_response_object=model_response, response_type="embedding")
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                    input=input,
                    api_key=api_key,
                    additional_args={"complete_input_dict": data},
                    original_response=str(e),
                )
            raise e

    def embedding(self,
                model: str,
                input: list,
                api_key: str,
                api_base: str,
                api_version: str,
                timeout: float, 
                logging_obj=None,
                model_response=None,
                optional_params=None,
                azure_ad_token: Optional[str]=None,
                client = None,
                aembedding=None,
                ):
        super().embedding()
        exception_mapping_worked = False
        if self._client_session is None:
            self._client_session = self.create_client_session()
        try: 
            data = {
                "model": model,
                "input": input,
                **optional_params
            }
            max_retries = data.pop("max_retries", 2)
            if not isinstance(max_retries, int): 
                raise AzureOpenAIError(status_code=422, message="max retries must be an int")
            
            # init AzureOpenAI Client
            azure_client_params = {
                "api_version": api_version,
                "azure_endpoint": api_base,
                "azure_deployment": model,
                "http_client": litellm.client_session,
                "max_retries": max_retries,
                "timeout": timeout
            }
            if api_key is not None:
                azure_client_params["api_key"] = api_key
            elif azure_ad_token is not None:
                azure_client_params["azure_ad_token"] = azure_ad_token

            ## LOGGING
            logging_obj.pre_call(
                    input=input,
                    api_key=api_key,
                    additional_args={
                        "complete_input_dict": data, 
                        "headers": {
                            "api_key": api_key, 
                            "azure_ad_token": azure_ad_token
                        }
                    },
                )
            
            if aembedding == True:
                response =  self.aembedding(data=data, input=input, logging_obj=logging_obj, api_key=api_key, model_response=model_response, azure_client_params=azure_client_params)
                return response
            if client is None:
                azure_client = AzureOpenAI(**azure_client_params) # type: ignore
            else:
                azure_client = client
            ## COMPLETION CALL            
            response = azure_client.embeddings.create(**data) # type: ignore
            ## LOGGING
            logging_obj.post_call(
                    input=input,
                    api_key=api_key,
                    additional_args={"complete_input_dict": data, "api_base": api_base},
                    original_response=response,
                )


            return convert_to_model_response_object(response_object=json.loads(response.model_dump_json()), model_response_object=model_response, response_type="embedding") # type: ignore
        except AzureOpenAIError as e: 
            exception_mapping_worked = True
            raise e
        except Exception as e: 
            if exception_mapping_worked: 
                raise e
            else: 
                import traceback
                raise AzureOpenAIError(status_code=500, message=traceback.format_exc())

    def image_generation(self,
                prompt: list,
                timeout: float, 
                model: Optional[str]=None,
                api_key: Optional[str] = None,
                api_base: Optional[str] = None,
                model_response: Optional[litellm.utils.ImageResponse] = None,
                logging_obj=None,
                optional_params=None,
                client=None,
                aimg_generation=None,
                ):
        exception_mapping_worked = False
        try: 
            model = model
            data = {
                # "model": model,
                "prompt": prompt,
                **optional_params
            }
            max_retries = data.pop("max_retries", 2)
            if not isinstance(max_retries, int): 
                raise AzureOpenAIError(status_code=422, message="max retries must be an int")
            
            # if aembedding == True:
            #     response =  self.aembedding(data=data, input=input, logging_obj=logging_obj, model_response=model_response, api_base=api_base, api_key=api_key, timeout=timeout, client=client, max_retries=max_retries) # type: ignore
            #     return response
            
            if client is None:
                azure_client = AzureOpenAI(api_key=api_key, base_url=api_base, http_client=litellm.client_session, timeout=timeout, max_retries=max_retries)  # type: ignore 
            else:
                azure_client = client
            
            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=azure_client.api_key,
                additional_args={"headers": {"Authorization": f"Bearer {azure_client.api_key}"}, "api_base": azure_client._base_url._uri_reference, "acompletion": False, "complete_input_dict": data},
            )
            
            ## COMPLETION CALL
            response = azure_client.images.generate(**data) # type: ignore
            ## LOGGING
            logging_obj.post_call(
                    input=input,
                    api_key=api_key,
                    additional_args={"complete_input_dict": data},
                    original_response=response,
                )
            # return response
            return convert_to_model_response_object(response_object=json.loads(response.model_dump_json()), model_response_object=model_response, response_type="image_generation") # type: ignore
        except AzureOpenAIError as e: 
            exception_mapping_worked = True
            raise e
        except Exception as e: 
            if exception_mapping_worked: 
                raise e
            else: 
                import traceback
                raise AzureOpenAIError(status_code=500, message=traceback.format_exc())