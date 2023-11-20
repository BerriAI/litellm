from typing import Optional, Union, Any
import types, time, json
import httpx
from .base import BaseLLM
from litellm.utils import ModelResponse, Choices, Message, CustomStreamWrapper, convert_to_model_response_object, Usage
from typing import Callable, Optional
import aiohttp, requests
import litellm
from openai import OpenAI, AsyncOpenAI

class OpenAIError(Exception):
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


class OpenAIConfig():
    """
    Reference: https://platform.openai.com/docs/api-reference/chat/create

    The class `OpenAIConfig` provides configuration for the OpenAI's Chat API interface. Below are the parameters:

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
    frequency_penalty: Optional[int]=None
    function_call: Optional[Union[str, dict]]=None
    functions: Optional[list]=None
    logit_bias: Optional[dict]=None
    max_tokens: Optional[int]=None
    n: Optional[int]=None
    presence_penalty: Optional[int]=None
    stop: Optional[Union[str, list]]=None
    temperature: Optional[int]=None
    top_p: Optional[int]=None

    def __init__(self,
                 frequency_penalty: Optional[int]=None,
                 function_call: Optional[Union[str, dict]]=None,
                 functions: Optional[list]=None,
                 logit_bias: Optional[dict]=None,
                 max_tokens: Optional[int]=None,
                 n: Optional[int]=None,
                 presence_penalty: Optional[int]=None,
                 stop: Optional[Union[str, list]]=None,
                 temperature: Optional[int]=None,
                 top_p: Optional[int]=None,) -> None:
        
        locals_ = locals()
        for key, value in locals_.items():
            if key != 'self' and value is not None:
                setattr(self.__class__, key, value)
   
    @classmethod
    def get_config(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') 
                and not isinstance(v, (types.FunctionType, types.BuiltinFunctionType, classmethod, staticmethod)) 
                and v is not None}

class OpenAITextCompletionConfig():
    """
    Reference: https://platform.openai.com/docs/api-reference/completions/create

    The class `OpenAITextCompletionConfig` provides configuration for the OpenAI's text completion API interface. Below are the parameters:

    - `best_of` (integer or null): This optional parameter generates server-side completions and returns the one with the highest log probability per token.

    - `echo` (boolean or null): This optional parameter will echo back the prompt in addition to the completion.

    - `frequency_penalty` (number or null): Defaults to 0. It is a numbers from -2.0 to 2.0, where positive values decrease the model's likelihood to repeat the same line.

    - `logit_bias` (map): This optional parameter modifies the likelihood of specified tokens appearing in the completion.
    
    - `logprobs` (integer or null): This optional parameter includes the log probabilities on the most likely tokens as well as the chosen tokens.

    - `max_tokens` (integer or null): This optional parameter sets the maximum number of tokens to generate in the completion.

    - `n` (integer or null): This optional parameter sets how many completions to generate for each prompt.

    - `presence_penalty` (number or null): Defaults to 0 and can be between -2.0 and 2.0. Positive values increase the model's likelihood to talk about new topics.
    
    - `stop` (string / array / null): Specifies up to 4 sequences where the API will stop generating further tokens.
    
    - `suffix` (string or null): Defines the suffix that comes after a completion of inserted text.

    - `temperature` (number or null): This optional parameter defines the sampling temperature to use.
    
    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling.
    """
    best_of: Optional[int]=None
    echo: Optional[bool]=None
    frequency_penalty: Optional[int]=None
    logit_bias: Optional[dict]=None
    logprobs: Optional[int]=None
    max_tokens: Optional[int]=None
    n: Optional[int]=None
    presence_penalty: Optional[int]=None
    stop: Optional[Union[str, list]]=None
    suffix: Optional[str]=None
    temperature: Optional[float]=None
    top_p: Optional[float]=None

    def __init__(self,
                 best_of: Optional[int]=None,
                 echo: Optional[bool]=None,
                 frequency_penalty: Optional[int]=None,
                 logit_bias: Optional[dict]=None,
                 logprobs: Optional[int]=None,
                 max_tokens: Optional[int]=None,
                 n: Optional[int]=None,
                 presence_penalty: Optional[int]=None,
                 stop: Optional[Union[str, list]]=None,
                 suffix: Optional[str]=None,
                 temperature: Optional[float]=None,
                 top_p: Optional[float]=None) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != 'self' and value is not None:
                setattr(self.__class__, key, value)
   
    @classmethod
    def get_config(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') 
                and not isinstance(v, (types.FunctionType, types.BuiltinFunctionType, classmethod, staticmethod)) 
                and v is not None}

class OpenAIChatCompletion(BaseLLM):

    def __init__(self) -> None:
        super().__init__()

    def completion(self, 
                model_response: ModelResponse,
                timeout: float, 
                model: Optional[str]=None,
                messages: Optional[list]=None,
                print_verbose: Optional[Callable]=None,
                api_key: Optional[str]=None,
                api_base: Optional[str]=None,
                acompletion: bool = False,
                logging_obj=None,
                optional_params=None,
                litellm_params=None,
                logger_fn=None,
                headers: Optional[dict]=None):
        super().completion()
        exception_mapping_worked = False
        try: 
            if headers: 
                optional_params["extra_headers"] = headers
            if model is None or messages is None:
                raise OpenAIError(status_code=422, message=f"Missing model or messages")
            
            if not isinstance(timeout, float):
                raise OpenAIError(status_code=422, message=f"Timeout needs to be a float")

            for _ in range(2): # if call fails due to alternating messages, retry with reformatted message
                data = {
                    "model": model,
                    "messages": messages, 
                    **optional_params
                }
                
                ## LOGGING
                logging_obj.pre_call(
                    input=messages,
                    api_key=api_key,
                    additional_args={"headers": headers, "api_base": api_base, "acompletion": acompletion, "complete_input_dict": data},
                )
                
                try: 
                    if acompletion is True: 
                        if optional_params.get("stream", False):
                            return self.async_streaming(logging_obj=logging_obj, data=data, model=model, api_base=api_base, api_key=api_key, timeout=timeout)
                        else:
                            return self.acompletion(data=data, model_response=model_response, api_base=api_base, api_key=api_key, timeout=timeout)
                    elif optional_params.get("stream", False):
                        return self.streaming(logging_obj=logging_obj, data=data, model=model, api_base=api_base, api_key=api_key, timeout=timeout)
                    else:
                        openai_client = OpenAI(api_key=api_key, base_url=api_base, http_client=litellm.client_session, timeout=timeout)
                        response = openai_client.chat.completions.create(**data) # type: ignore
                        logging_obj.post_call(
                                input=None,
                                api_key=api_key,
                                original_response=response,
                                additional_args={"complete_input_dict": data},
                            )
                        return convert_to_model_response_object(response_object=json.loads(response.model_dump_json()), model_response_object=model_response)
                except Exception as e:
                    if "Conversation roles must alternate user/assistant" in str(e) or "user and assistant roles should be alternating" in str(e): 
                        # reformat messages to ensure user/assistant are alternating, if there's either 2 consecutive 'user' messages or 2 consecutive 'assistant' message, add a blank 'user' or 'assistant' message to ensure compatibility
                        new_messages = []
                        for i in range(len(messages)-1): 
                            new_messages.append(messages[i])
                            if messages[i]["role"] == messages[i+1]["role"]:
                                if messages[i]["role"] == "user":
                                    new_messages.append({"role": "assistant", "content": ""})
                                else:
                                    new_messages.append({"role": "user", "content": ""})
                        new_messages.append(messages[-1])
                        messages = new_messages
                    elif "Last message must have role `user`" in str(e):
                        new_messages = messages
                        new_messages.append({"role": "user", "content": ""})
                        messages = new_messages
                    else:
                        raise e
        except OpenAIError as e: 
            exception_mapping_worked = True
            raise e
        except Exception as e: 
            raise e
    
    async def acompletion(self, 
                          data: dict, 
                          model_response: ModelResponse, 
                          timeout: float,
                          api_key: Optional[str]=None,
                          api_base: Optional[str]=None): 
        response = None
        try: 
            openai_aclient = AsyncOpenAI(api_key=api_key, base_url=api_base, http_client=litellm.aclient_session, timeout=timeout)
            response = await openai_aclient.chat.completions.create(**data)
            return convert_to_model_response_object(response_object=json.loads(response.model_dump_json()), model_response_object=model_response)
        except Exception as e: 
            if response and hasattr(response, "text"):
                raise OpenAIError(status_code=500, message=f"{str(e)}\n\nOriginal Response: {response.text}")
            else: 
                if type(e).__name__ == "ReadTimeout": 
                    raise OpenAIError(status_code=408, message=f"{type(e).__name__}")
                else:
                    raise OpenAIError(status_code=500, message=f"{str(e)}")

    def streaming(self,
                  logging_obj,
                  timeout: float,
                  data: dict, 
                  model: str,
                  api_key: Optional[str]=None,
                  api_base: Optional[str]=None
    ):
        openai_client = OpenAI(api_key=api_key, base_url=api_base, http_client=litellm.client_session, timeout=timeout)
        response = openai_client.chat.completions.create(**data)
        streamwrapper = CustomStreamWrapper(completion_stream=response, model=model, custom_llm_provider="openai",logging_obj=logging_obj)
        for transformed_chunk in streamwrapper:
            yield transformed_chunk

    async def async_streaming(self, 
                          logging_obj,
                          timeout: float,
                          data: dict, 
                          model: str,
                          api_key: Optional[str]=None,
                          api_base: Optional[str]=None):
        response = None
        try: 
            openai_aclient = AsyncOpenAI(api_key=api_key, base_url=api_base, http_client=litellm.aclient_session, timeout=timeout)
            response = await openai_aclient.chat.completions.create(**data)
            streamwrapper = CustomStreamWrapper(completion_stream=response, model=model, custom_llm_provider="openai",logging_obj=logging_obj)
            async for transformed_chunk in streamwrapper:
                yield transformed_chunk
        except Exception as e: # need to exception handle here. async exceptions don't get caught in sync functions. 
            if response is not None and hasattr(response, "text"):
                raise OpenAIError(status_code=500, message=f"{str(e)}\n\nOriginal Response: {response.text}")
            else:
                if type(e).__name__ == "ReadTimeout": 
                    raise OpenAIError(status_code=408, message=f"{type(e).__name__}")
                else:
                    raise OpenAIError(status_code=500, message=f"{str(e)}")
                
    def embedding(self,
                model: str,
                input: list,
                api_key: Optional[str] = None,
                api_base: Optional[str] = None,
                logging_obj=None,
                model_response=None,
                optional_params=None,):
        super().embedding()
        exception_mapping_worked = False
        try: 
            openai_client = OpenAI(api_key=api_key, base_url=api_base, http_client=litellm.client_session)
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
            response = openai_client.embeddings.create(**data) # type: ignore
            ## LOGGING
            logging_obj.post_call(
                    input=input,
                    api_key=api_key,
                    additional_args={"complete_input_dict": data},
                    original_response=response,
                )
            
            embedding_response = json.loads(response.model_dump_json())
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
        except OpenAIError as e: 
            exception_mapping_worked = True
            raise e
        except Exception as e: 
            if exception_mapping_worked: 
                raise e
            else: 
                import traceback
                raise OpenAIError(status_code=500, message=traceback.format_exc())


class OpenAITextCompletion(BaseLLM):
    _client_session: httpx.Client

    def __init__(self) -> None:
        super().__init__()
        self._client_session = self.create_client_session()
    
    def validate_environment(self, api_key):
        headers = {
            "content-type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers
    
    def convert_to_model_response_object(self, response_object: Optional[dict]=None, model_response_object: Optional[ModelResponse]=None):
        try: 
            ## RESPONSE OBJECT
            if response_object is None or model_response_object is None:
                raise ValueError("Error in response object format")
            choice_list=[]
            for idx, choice in enumerate(response_object["choices"]): 
                message = Message(content=choice["text"], role="assistant")
                choice = Choices(finish_reason=choice["finish_reason"], index=idx, message=message)
                choice_list.append(choice)
            model_response_object.choices = choice_list

            if "usage" in response_object: 
                model_response_object.usage = response_object["usage"]
            
            if "id" in response_object: 
                model_response_object.id = response_object["id"]
            
            if "model" in response_object: 
                model_response_object.model = response_object["model"]
            
            model_response_object._hidden_params["original_response"] = response_object # track original response, if users make a litellm.text_completion() request, we can return the original response
            return model_response_object
        except Exception as e: 
            raise e

    def completion(self, 
               model_response: ModelResponse,    
               api_key: str,
               model: str,
               messages: list,
               print_verbose: Optional[Callable]=None,
               api_base: Optional[str]=None,
               logging_obj=None,
               acompletion: bool = False,
               optional_params=None,
               litellm_params=None,
               logger_fn=None,
               headers: Optional[dict]=None):
        super().completion()
        exception_mapping_worked = False
        try: 
            if headers is None:
                headers = self.validate_environment(api_key=api_key)
            if model is None or messages is None:
                raise OpenAIError(status_code=422, message=f"Missing model or messages")
            
            api_base = f"{api_base}/completions"

            if len(messages)>0 and "content" in messages[0] and type(messages[0]["content"]) == list: 
                prompt = messages[0]["content"]
            else:
                prompt = " ".join([message["content"] for message in messages]) # type: ignore

            data = {
                "model": model,
                "prompt": prompt, 
                **optional_params
            }
            
            ## LOGGING
            logging_obj.pre_call(
                input=messages,
                api_key=api_key,
                additional_args={"headers": headers, "api_base": api_base, "complete_input_dict": data},
            )
            if acompletion == True: 
                if optional_params.get("stream", False):
                    return self.async_streaming(logging_obj=logging_obj, api_base=api_base, data=data, headers=headers, model_response=model_response, model=model)
                else:
                    return self.acompletion(api_base=api_base, data=data, headers=headers, model_response=model_response, prompt=prompt, api_key=api_key, logging_obj=logging_obj, model=model) # type: ignore
            elif optional_params.get("stream", False):
                return self.streaming(logging_obj=logging_obj, api_base=api_base, data=data, headers=headers, model_response=model_response, model=model)
            else:
                response = httpx.post(
                    url=f"{api_base}",
                    json=data,
                    headers=headers,
                )
                if response.status_code != 200:
                    raise OpenAIError(status_code=response.status_code, message=response.text)
                
                ## LOGGING
                logging_obj.post_call(
                    input=prompt,
                    api_key=api_key,
                    original_response=response,
                    additional_args={
                        "headers": headers,
                        "api_base": api_base,
                    },
                )

                ## RESPONSE OBJECT
                return self.convert_to_model_response_object(response_object=response.json(), model_response_object=model_response)
        except Exception as e: 
            raise e
    
    async def acompletion(self, 
                        logging_obj,
                        api_base: str, 
                        data: dict, 
                        headers: dict, 
                        model_response: ModelResponse, 
                        prompt: str, 
                        api_key: str, 
                        model: str): 
        async with httpx.AsyncClient() as client:
            response = await client.post(api_base, json=data, headers=headers, timeout=litellm.request_timeout) 
            response_json = response.json()
            if response.status_code != 200:
                raise OpenAIError(status_code=response.status_code, message=response.text)
                
            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                original_response=response,
                additional_args={
                    "headers": headers,
                    "api_base": api_base,
                },
            )

            ## RESPONSE OBJECT
            return self.convert_to_model_response_object(response_object=response_json, model_response_object=model_response)

    def streaming(self,
                  logging_obj,
                  api_base: str, 
                  data: dict, 
                  headers: dict, 
                  model_response: ModelResponse, 
                  model: str
    ):
        with httpx.stream(
                    url=f"{api_base}",
                    json=data,
                    headers=headers,
                    method="POST",
                    timeout=litellm.request_timeout
                ) as response: 
                    if response.status_code != 200:
                        raise OpenAIError(status_code=response.status_code, message=response.text) 
                    
                    streamwrapper = CustomStreamWrapper(completion_stream=response.iter_lines(), model=model, custom_llm_provider="text-completion-openai",logging_obj=logging_obj)
                    for transformed_chunk in streamwrapper:
                        yield transformed_chunk

    async def async_streaming(self, 
                          logging_obj,
                          api_base: str, 
                          data: dict, 
                          headers: dict, 
                          model_response: ModelResponse, 
                          model: str):
        client = httpx.AsyncClient()
        async with client.stream(
                    url=f"{api_base}",
                    json=data,
                    headers=headers,
                    method="POST",
                    timeout=litellm.request_timeout
                ) as response: 
            if response.status_code != 200:
                raise OpenAIError(status_code=response.status_code, message=response.text)
            
            streamwrapper = CustomStreamWrapper(completion_stream=response.aiter_lines(), model=model, custom_llm_provider="text-completion-openai",logging_obj=logging_obj)
            async for transformed_chunk in streamwrapper:
                yield transformed_chunk