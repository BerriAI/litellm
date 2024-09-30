"""
Handler for Snowflake's Cortex Complete endpoint
"""

import json
import time
from typing import Callable, Optional

import httpx

from litellm.llms.OpenAI.openai import OpenAIConfig
from litellm.llms.snowflake.snowflake_utils import ModelResponseIterator
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import CustomStreamWrapper, ModelResponse, Usage

from ..base import BaseLLM



class SnowflakeError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://docs.snowflake.com/")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

# makes headers for API call
def validate_environment(api_key, user_headers):
    if api_key is None:
        raise ValueError(
            "Missing Snowflake API Key - A call is being made to Snowflake but no key is set either in the environment variables or via params"
        )

    headers = {
        "Accept": "application/json,text/stream",
        "Content-Type": "application/json",
        "Authorization": f'Snowflake Token="{api_key}"',
    }

    if user_headers is not None and isinstance(user_headers, dict):
        headers = {**headers, **user_headers}
    return headers


class SnowflakeTextCompletion(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def _process_response(
        self, model_response: ModelResponse, response, encoding, prompt: str, model: str
    ):
        ## RESPONSE OBJECT
        try:
            completion_response = self._process_snowflake_complete_response(
                response.content
            )
        except:
            raise SnowflakeError(
                message=response.text, status_code=response.status_code
            )
        if "error" in completion_response:
            raise SnowflakeError(
                message=str(completion_response["error"]),
                status_code=response.status_code,
            )
        else:
            if len(completion_response["completion"]) > 0:
                model_response.choices[0].message.content = completion_response[
                    "completion"
                ]

            model_response.choices[0].finish_reason = completion_response["stop_reason"]

        ## CALCULATING USAGE
        prompt_tokens = len(encoding.encode(prompt[0]["content"]))
        completion_tokens = len(encoding.encode(completion_response["completion"]))

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        setattr(model_response, "usage", usage)

        return model_response

    def _process_snowflake_complete_response(self, data: bytes):

        # Decode the bytes object to a string
        data_str = data.decode("utf-8")

        # Split the data into individual JSON objects
        json_objects = data_str.split("\ndata: ")

        # Initialize an empty list to store the JSON objects
        json_list = []

        # Iterate over each JSON object
        for obj in json_objects:
            obj = obj.strip()
            if obj:
                # Remove the 'data: ' prefix if it exists
                if obj.startswith("data: "):
                    obj = obj[6:]
                # Load the JSON object into a Python dictionary
                json_dict = json.loads(str(obj))
                # Append the JSON dictionary to the list
                json_list.append(json_dict)

        completion = ""
        model = ""
        choices = {}
        for chunk in json_list:

            model = chunk["model"]
            choices = chunk["choices"][0]

            if "content" in choices["delta"].keys():

                completion += choices["delta"]["content"]

        processed_response = {
            "model": model,
            "completion": completion,
            "stop_reason": choices["finish_reason"],
        }

        return ModelResponse(**processed_response)

    async def async_completion(
        self,
        model: str,
        model_response: ModelResponse,
        api_base: str,
        logging_obj,
        encoding,
        headers: dict,
        data: dict,
        client=None,
    ):
        if client is None:
            client = AsyncHTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))

        response = await client.post(api_base, headers=headers, data=json.dumps(data))

        if response.status_code != 200:
            raise SnowflakeError(
                status_code=response.status_code, message=response.text
            )

        ## LOGGING
        logging_obj.post_call(
            input=data["messages"],
            api_key=headers.get("Authorization"),
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )

        response = self._process_response(
            model_response=model_response,
            response=response,
            encoding=encoding,
            prompt=data["messages"],
            model=model,
        )
        return response

    async def async_streaming(
        self,
        model: str,
        api_base: str,
        logging_obj,
        headers: dict,
        data: Optional[dict],
        client=None,
    ):
        if client is None:
            client = AsyncHTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))

        response = await client.post(api_base, headers=headers, data=json.dumps(data))

        if response.status_code != 200:
            raise SnowflakeError(
                status_code=response.status_code, message=response.text
            )

        completion_stream = response.aiter_lines()

        streamwrapper = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="snowflake",
            logging_obj=logging_obj,
        )
        return streamwrapper

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        acompletion: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        client=None,
    ):
        headers = validate_environment(api_key, headers)

        ## Load Config
        # config = SnowflakeTextConfig.get_config()
        config = OpenAIConfig.get_config()
        for k, v in config.items():
             if k not in optional_params:
                 optional_params[k] = v

        data = {
            "model": model,
            "messages": messages,
            **optional_params,
        }

        ## LOGGING
        logging_obj.pre_call(
            input=messages[0]['content'],
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        ## COMPLETION CALL
        if "stream" in optional_params and optional_params["stream"] == True:
            if acompletion == True:
                return self.async_streaming(
                    model=model,
                    api_base=api_base,
                    logging_obj=logging_obj,
                    headers=headers,
                    data=data,
                    client=None,
                )

            if client is None:
                client = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))

            response = client.post(api_base, headers=headers, data=json.dumps(data))

            if response.status_code != 200:
                raise SnowflakeError(
                    status_code=response.status_code, message=response.text
                )

            completion_stream = ModelResponseIterator(
                streaming_response=response.iter_lines(), sync_stream=True
            )

            stream_response = CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider="snowflake",
                logging_obj=logging_obj,
            )
            return stream_response
        elif acompletion == True:
            return self.async_completion(
                model=model,
                model_response=model_response,
                api_base=api_base,
                logging_obj=logging_obj,
                encoding=encoding,
                headers=headers,
                data=data,
                client=client,
            )
        else:
            if client is None:
                client = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))
            response = client.post(api_base, headers=headers, data=json.dumps(data))

            if response.status_code != 200:
                raise SnowflakeError(
                    status_code=response.status_code, message=response.text
                )

            ## LOGGING
            logging_obj.post_call(
                input=data["messages"],
                api_key=api_key,
                original_response=response.text,
                additional_args={"complete_input_dict": data},
            )
            print_verbose(f"raw model_response: {response.text}")

            response = self._process_response(
                model_response=model_response,
                response=response,
                encoding=encoding,
                prompt=data["messages"],
                model=model,
            )
            return response

    def embedding(self):
        # logic for parsing in - calling - parsing out model embedding calls
        pass
