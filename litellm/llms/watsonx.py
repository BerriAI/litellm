import os
from typing import Dict, List, Optional, Union

import httpx

import litellm
from litellm.utils import EmbeddingResponse, Message, ModelResponse

from .base import BaseLLM


class WatsonxError(Exception):
    def __init__(
        self,
        status_code,
        message,
        url,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
    ):
        self.status_code = status_code
        self.message = message
        if request is not None:
            self.request = request
        else:
            self.request = httpx.Request(
                method="POST",
                url=url,
            )
        if response is not None:
            self.response = response
        else:
            self.response = httpx.Response(
                status_code=status_code, request=self.request
            )
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class TextGenLengthPenalty:
    def __init__(
        self,
        length_penalty: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ):
        self.length_penalty = length_penalty
        self.min_length = min_length
        self.max_length = max_length


class ParameterTruncateInputTokens:
    def __init__(
        self,
        truncate_input_tokens: Optional[bool] = None,
        max_input_tokens: Optional[int] = None,
    ):
        self.truncate_input_tokens = truncate_input_tokens
        self.max_input_tokens = max_input_tokens


class ReturnOptionProperties:
    def __init__(
        self, return_scores: Optional[bool] = None, return_text: Optional[bool] = None
    ):
        self.return_scores = return_scores
        self.return_text = return_text


class TextGenParameters:
    def __init__(
        self,
        decoding_method: Optional[str] = None,
        length_penalty: Optional[TextGenLengthPenalty] = None,
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        random_seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        time_limit: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        truncate_input_tokens: Optional[ParameterTruncateInputTokens] = None,
        return_options: Optional[ReturnOptionProperties] = None,
        include_stop_sequence: Optional[bool] = None,
    ):
        self.decoding_method = decoding_method
        self.length_penalty = length_penalty
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.random_seed = random_seed
        self.stop_sequences = stop_sequences or []
        self.temperature = temperature
        self.time_limit = time_limit
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.truncate_input_tokens = truncate_input_tokens
        self.return_options = return_options
        self.include_stop_sequence = include_stop_sequence

        # Validation
        if self.decoding_method and self.decoding_method not in ["sample", "greedy"]:
            raise ValueError(
                "Invalid value for decoding_method. Must be 'sample' or 'greedy'."
            )
        if self.max_new_tokens is not None and self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer.")
        if self.min_new_tokens is not None and self.min_new_tokens < 0:
            raise ValueError("min_new_tokens must be a non-negative integer.")
        if self.temperature is not None and (
            self.temperature < 0 or self.temperature > 2
        ):
            raise ValueError("temperature must be between 0 and 2.")
        if self.top_p is not None and (self.top_p <= 0 or self.top_p > 1):
            raise ValueError("top_p must be between 0 and 1, exclusive.")
        if self.repetition_penalty is not None and (
            self.repetition_penalty < 1 or self.repetition_penalty > 2
        ):
            raise ValueError("repetition_penalty must be between 1 and 2.")
        if self.stop_sequences and len(self.stop_sequences) > 6:
            raise ValueError("stop_sequences can have at most 6 items.")
        if self.top_k is not None and (self.top_k < 1 or self.top_k > 100):
            raise ValueError("top_k must be between 1 and 100.")


class WatsonxConfig:
    """
    - `api_url` (str)  base URL of the API endpoint
    - `api_version` (str)  version of the API to use
    - `api_key` (str)  API authentication key
    - `model_id` (str)  ID of the model to use for text generation
    - `project_id` (str)  ID of the project
    - `input_text` (str)  input text or prompt for the text generation
    - `params` (Optional[TextGenParameters]) parameters for text generation
    """

    def __init__(
        self,
        api_url: str,  # The base URL of the API endpoint
        api_version: str,  # The version of the API to use
        api_key: str,  # The API authentication key
        model_id: str,  # The ID of the model to use for text generation
        project_id: str,  # The ID of the project
        input_text: str,  # The input text or prompt for the text generation
        # Parameters for text generation
        params: Optional[TextGenParameters] = None,
    ):
        self.api_url = api_url
        self.api_version = api_version
        self.api_key = api_key
        self.model_id = model_id
        self.project_id = project_id
        self.input_text = input_text
        self.params = params or TextGenParameters()

    @property
    def url(self) -> str:
        url = f"{self.api_url}"

        if self.api_version:
            url += f"?version={self.api_version}"

        return url

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @property
    def data(self) -> Dict[str, Union[str, Dict]]:
        return {
            "model_id": self.model_id,
            "input": self.input_text,
            "project_id": self.project_id,
            "parameters": self.params,
        }


class Watsonx(BaseLLM):
    _client_session: Optional[httpx.Client] = None
    _aclient_session: Optional[httpx.AsyncClient] = None

    PROJECT_ID_ENV_NAME = "WATSONX_PROJECT_ID"
    API_KEY_ENV_NAME = "WATSONX_API_KEY"
    URL_ENV_NAME = "WATSONX_URL"

    def create_client_session(self) -> httpx.Client:
        if litellm.client_session:
            _client_session = litellm.client_session
        else:
            _client_session = httpx.Client()

        return _client_session

    def create_aclient_session(self) -> httpx.AsyncClient:
        if litellm.aclient_session:
            _aclient_session = litellm.aclient_session
        else:
            _aclient_session = httpx.AsyncClient()

        return _aclient_session

    def __exit__(self) -> None:
        if hasattr(self, "_client_session") and self._client_session:
            self._client_session.close()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if hasattr(self, "_aclient_session") and self._aclient_session:
            await self._aclient_session.aclose()

    def convert_to_model_response_object(
        self,
        completion_response,
        model_response: ModelResponse,
    ):
        for i, response in enumerate(completion_response):
            if len(response["generated_text"]) > 0:
                model_response["choices"][i]["message"]["content"] = response[
                    "generated_text"
                ]

        model_response["model"] = completion_response["model_id"]
        model_response["created"] = completion_response["created_at"]

        return model_response

    def validate_environment(self) -> None:
        pass

    def get_headers(self, api_key: str) -> Dict[str, str]:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def completion(
        self,
        model: str,
        model_response: ModelResponse,
        messages: List[Message],
        print_verbose: Callable,
        encoding: Any,
        logging_obj: Any,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
    ) -> ModelResponse:
        """
        https://cloud.ibm.com/apidocs/watsonx-ai#text-generation
        """
        try:
            project_id = project_id or os.environ.get(self.PROJECT_ID_ENV_NAME)
            api_key = api_key or os.environ.get(self.API_KEY_ENV_NAME)
            api_base = api_base or os.environ.get(self.URL_ENV_NAME)

            # TODO
            assert project_id
            assert api_key
            assert api_base

            headers = self.get_headers(api_key)

            data = {
                "model": model,
                "prompt": prompt,
                **optional_params,
            }

            with httpx.Client() as client:
                response = client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                )

                response_json = response.json()

                if response.status_code != 200:
                    if "error" in response_json:
                        raise WatsonxError(
                            status_code=response.status_code,
                            message=response_json["error"],
                            url=api_base,
                            request=response.request,
                            response=response,
                        )
                    else:
                        raise WatsonxError(
                            status_code=response.status_code,
                            message=response.text,
                            url=api_base,
                            request=response.request,
                            response=response,
                        )

                return self.convert_to_model_response_object(
                    response_json,
                    model_response,
                )
        except Exception as e:
            if isinstance(e, WatsonxError):
                raise e
            if isinstance(e, httpx.TimeoutException):
                raise WatsonxError(
                    status_code=500,
                    message="Request Timeout Error",
                    url=api_base,
                )
            if response is not None and hasattr(response, "text"):
                raise WatsonxError(
                    status_code=500,
                    message=f"{str(e)}\n\nOriginal Response: {response.text}",
                    url=api_base,
                )
            raise WatsonxError(status_code=500, message=f"{str(e)}", url=api_base)

    def embedding(self, *args, **kwargs) -> EmbeddingResponse:
        """
        TODO

        https://cloud.ibm.com/apidocs/watsonx-ai#text-embeddings
        """
        pass
