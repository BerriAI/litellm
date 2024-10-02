"""
Handler for Snowflake's Cortex Complete endpoint
"""

from typing import Callable
import httpx
from litellm.utils import ModelResponse
from litellm.llms.databricks.chat import DatabricksChatCompletion
from litellm.types.utils import Choices, Message
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
        "Accept": "text/stream",
        "Content-Type": "application/json",
        "Authorization": f'Snowflake Token="{api_key}"',
    }

    if user_headers is not None and isinstance(user_headers, dict):
        headers = {**headers, **user_headers}
    return headers


class SnowflakeTextCompletion(BaseLLM):

    def __init__(self) -> None:
        super().__init__()

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
        openai_like_chat_completions = DatabricksChatCompletion()

        # Non Streaming Mode Currently Not Enabled in Preview
        streaming = optional_params["stream"] if "stream" in optional_params else False
        optional_params["stream"] = True

        response = openai_like_chat_completions.completion(
            model=model,
            messages=messages,
            api_base=api_base,
            api_key=None,
            headers=headers,
            custom_prompt_dict=custom_prompt_dict,
            model_response=model_response,
            print_verbose=print_verbose,
            logging_obj=logging_obj,
            optional_params=optional_params,
            acompletion=acompletion,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            client=client,
            encoding=encoding,
            custom_llm_provider="snowflake",
            custom_endpoint=True,
        )

        if streaming:
            return response
        else:

            completion = ""
            for part in response:
                completion += part.choices[0]["delta"]["content"] or ""
            choices = Choices(message=Message(content=completion))

            return ModelResponse(choices=[choices])
