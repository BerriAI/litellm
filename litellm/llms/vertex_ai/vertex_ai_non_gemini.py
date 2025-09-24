import json
import os
import time
from typing import Any, Callable, Optional, cast

import google.auth
import httpx
from google.auth.transport import requests as google_auth_requests
from google.cloud.aiplatform import constants

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.bedrock.common_utils import ModelResponseIterator
from litellm.llms.custom_httpx.http_handler import (
    _DEFAULT_TTL_FOR_HTTPX_CLIENTS,
    get_async_httpx_client,
)
from litellm.types.llms.vertex_ai import *
from litellm.utils import CustomStreamWrapper, ModelResponse, Usage


class VertexAIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url=" https://cloud.google.com/vertex-ai/"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


def get_google_auth_session():
    """
    Get Google authenticated session for making HTTP requests to Vertex AI.
    """
    credentials, project = google.auth.default()
    credentials._scopes = constants.base.DEFAULT_AUTHED_SCOPES
    return google_auth_requests.AuthorizedSession(credentials)


def get_project_number(project_id: str) -> Optional[str]:
    """
    Get the Google Cloud project number from the project ID using direct HTTP request.

    Args:
        project_id: The Google Cloud project ID

    Returns:
        The project number as a string, or None if not found
    """
    try:
        # Get authenticated session
        session = get_google_auth_session()

        # Make direct HTTP request to Cloud Resource Manager API
        url = f"https://cloudresourcemanager.googleapis.com/v1/projects/{project_id}"
        response = session.get(url)

        if response.status_code == 200:
            project_data = response.json()
            project_number = project_data.get("projectNumber")
            return str(project_number) if project_number else None
        else:
            from litellm._logging import verbose_logger

            verbose_logger.warning(
                f"Could not retrieve project number for {project_id}: HTTP {response.status_code} - {response.text}"
            )
            return None

    except Exception as e:
        # Log the error but don't raise it to avoid breaking the main flow
        from litellm._logging import verbose_logger

        verbose_logger.warning(
            f"Could not retrieve project number [project_id=REDACTED]: {e}"
        )
        return None


def get_dedicated_endpoint_url(
    endpoint_name: str,
    vertex_project: str,
    vertex_location: str,
    project_number: Optional[str] = None,
) -> str:
    """
    Get the URL for a dedicated endpoint prediction request.
    For dedicated endpoints, we need to use the endpoint's dedicated DNS.
    """
    # For dedicated endpoints, the endpoint_name should contain the full endpoint ID
    # Format: projects/{project}/locations/{location}/endpoints/{endpoint_id}
    if not endpoint_name.startswith("projects/"):
        endpoint_id = endpoint_name
        endpoint_name = f"projects/{vertex_project}/locations/{vertex_location}/endpoints/{endpoint_id}"

    # Extract endpoint_id from the full resource name
    endpoint_id = endpoint_name.split("/")[-1]

    # For dedicated endpoints, use the custom DNS format:
    # https://{ENDPOINT_ID}.{LOCATION}-{PROJECT_NUMBER}.prediction.vertexai.goog/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict
    # Note: This requires the project number, not project ID
    if project_number is None:
        # Try to get the project number from the project ID
        project_number = get_project_number(vertex_project)
        if project_number is None:
            raise ValueError(f"Project number not found for {vertex_project}")

    return f"https://{endpoint_id}.{vertex_location}-{project_number}.prediction.vertexai.goog/v1/projects/{vertex_project}/locations/{vertex_location}/endpoints/{endpoint_id}:predict"


def make_vertex_ai_prediction_request(
    endpoint_name: str,
    instances: list,
    vertex_project: str,
    vertex_location: str,
    parameters: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> dict:
    """
    Make a direct HTTP request to Vertex AI prediction endpoint.
    """
    # Get authenticated session
    session = get_google_auth_session()

    # Construct URL
    url = get_dedicated_endpoint_url(endpoint_name, vertex_project, vertex_location)

    # Prepare request body
    request_body: dict = {"instances": instances}
    if parameters:
        request_body["parameters"] = parameters

    # Set headers
    headers = {"Content-Type": "application/json"}

    # Make the request
    response = session.post(
        url=url, data=json.dumps(request_body), headers=headers, timeout=timeout
    )

    if response.status_code != 200:
        raise VertexAIError(
            status_code=response.status_code,
            message=f"Failed to make prediction request. Status code: {response.status_code}, response: {response.text}.",
        )

    return response.json()


async def make_vertex_ai_prediction_request_async(
    endpoint_name: str,
    instances: list,
    vertex_project: str,
    vertex_location: str,
    parameters: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> dict:
    """
    Make an async HTTP request to Vertex AI prediction endpoint.
    """
    # Get async HTTP client
    http_client = get_async_httpx_client("vertex_ai")  # type: ignore

    # Construct URL
    url = get_dedicated_endpoint_url(endpoint_name, vertex_project, vertex_location)

    # Prepare request body
    request_body: dict = {"instances": instances}
    if parameters:
        request_body["parameters"] = parameters

    # Set headers
    headers = {"Content-Type": "application/json"}

    # Make the request
    response = await http_client.post(
        url=url, json=request_body, headers=headers, timeout=timeout
    )

    if response.status_code != 200:
        raise VertexAIError(
            status_code=response.status_code,
            message=f"Failed to make prediction request. Status code: {response.status_code}, response: {response.text}.",
        )

    return response.json()


class TextStreamer:
    """
    Fake streaming iterator for Vertex AI Model Garden calls
    """

    def __init__(self, text):
        self.text = text.split()  # let's assume words as a streaming unit
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.text):
            result = self.text[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.text):
            result = self.text[self.index]
            self.index += 1
            return result
        else:
            raise StopAsyncIteration  # once we run out of data to stream, we raise this error


def _get_client_cache_key(
    model: str, vertex_project: Optional[str], vertex_location: Optional[str]
):
    _cache_key = f"{model}-{vertex_project}-{vertex_location}"
    return _cache_key


def _get_client_from_cache(client_cache_key: str):
    return litellm.in_memory_llm_clients_cache.get_cache(client_cache_key)


def _set_client_in_cache(client_cache_key: str, vertex_llm_model: Any):
    litellm.in_memory_llm_clients_cache.set_cache(
        key=client_cache_key,
        value=vertex_llm_model,
        ttl=_DEFAULT_TTL_FOR_HTTPX_CLIENTS,
    )


def completion(  # noqa: PLR0915
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    logging_obj,
    optional_params: dict,
    vertex_project=None,
    vertex_location=None,
    vertex_credentials=None,
    litellm_params=None,
    logger_fn=None,
    acompletion: bool = False,
):
    """
    NON-GEMINI/ANTHROPIC CALLS.

    This is the handler for OLDER PALM MODELS and VERTEX AI MODEL GARDEN

    For Vertex AI Anthropic: `vertex_anthropic.py`
    For Gemini: `vertex_httpx.py`
    """
    try:
        import vertexai
    except Exception:
        raise VertexAIError(
            status_code=400,
            message="vertexai import failed please run `pip install google-cloud-aiplatform`. This is required for the 'vertex_ai/' route on LiteLLM",
        )

    if not (
        hasattr(vertexai, "preview") or hasattr(vertexai.preview, "language_models")
    ):
        raise VertexAIError(
            status_code=400,
            message="""Upgrade vertex ai. Run `pip install "google-cloud-aiplatform>=1.38"`""",
        )
    try:
        import google.auth  # type: ignore
        from google.cloud import aiplatform  # type: ignore
        from google.cloud.aiplatform_v1beta1.types import (
            content as gapic_content_types,  # type: ignore
        )
        from google.protobuf import json_format  # type: ignore
        from google.protobuf.struct_pb2 import Value  # type: ignore
        from vertexai.language_models import CodeGenerationModel, TextGenerationModel
        from vertexai.preview.generative_models import GenerativeModel
        from vertexai.preview.language_models import ChatModel, CodeChatModel

        ## Load credentials with the correct quota project ref: https://github.com/googleapis/python-aiplatform/issues/2557#issuecomment-1709284744
        print_verbose(
            f"VERTEX AI: vertex_project={vertex_project}; vertex_location={vertex_location}"
        )

        _cache_key = _get_client_cache_key(
            model=model, vertex_project=vertex_project, vertex_location=vertex_location
        )
        _vertex_llm_model_object = _get_client_from_cache(client_cache_key=_cache_key)

        if _vertex_llm_model_object is None:
            from google.auth.credentials import Credentials

            if vertex_credentials is not None and isinstance(vertex_credentials, str):
                import google.oauth2.service_account

                json_obj = json.loads(vertex_credentials)

                creds = (
                    google.oauth2.service_account.Credentials.from_service_account_info(
                        json_obj,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                )
            else:
                creds, _ = google.auth.default(quota_project_id=vertex_project)
            print_verbose(
                f"VERTEX AI: creds={creds}; google application credentials: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}"
            )
            vertexai.init(
                project=vertex_project,
                location=vertex_location,
                credentials=cast(Credentials, creds),
            )

        ## Load Config
        config = litellm.VertexAIConfig.get_config()
        for k, v in config.items():
            if k not in optional_params:
                optional_params[k] = v

        ## Process safety settings into format expected by vertex AI
        safety_settings = None
        if "safety_settings" in optional_params:
            safety_settings = optional_params.pop("safety_settings")
            if not isinstance(safety_settings, list):
                raise ValueError("safety_settings must be a list")
            if len(safety_settings) > 0 and not isinstance(safety_settings[0], dict):
                raise ValueError("safety_settings must be a list of dicts")
            safety_settings = [
                gapic_content_types.SafetySetting(x) for x in safety_settings
            ]

        # vertexai does not use an API key, it looks for credentials.json in the environment

        prompt = " ".join(
            [
                message.get("content")
                for message in messages
                if isinstance(message.get("content", None), str)
            ]
        )

        mode = ""

        request_str = ""
        response_obj = None
        instances = None
        client_options = {
            "api_endpoint": f"{vertex_location}-aiplatform.googleapis.com"
        }
        fake_stream = False
        if (
            model in litellm.vertex_language_models
            or model in litellm.vertex_vision_models
        ):
            llm_model: Any = _vertex_llm_model_object or GenerativeModel(model)
            mode = "vision"
            request_str += f"llm_model = GenerativeModel({model})\n"
        elif model in litellm.vertex_chat_models:
            llm_model = _vertex_llm_model_object or ChatModel.from_pretrained(model)
            mode = "chat"
            request_str += f"llm_model = ChatModel.from_pretrained({model})\n"
        elif model in litellm.vertex_text_models:
            llm_model = _vertex_llm_model_object or TextGenerationModel.from_pretrained(
                model
            )
            mode = "text"
            request_str += f"llm_model = TextGenerationModel.from_pretrained({model})\n"
        elif model in litellm.vertex_code_text_models:
            llm_model = _vertex_llm_model_object or CodeGenerationModel.from_pretrained(
                model
            )
            mode = "text"
            request_str += f"llm_model = CodeGenerationModel.from_pretrained({model})\n"
            fake_stream = True
        elif model in litellm.vertex_code_chat_models:  # vertex_code_llm_models
            llm_model = _vertex_llm_model_object or CodeChatModel.from_pretrained(model)
            mode = "chat"
            request_str += f"llm_model = CodeChatModel.from_pretrained({model})\n"
        elif model == "private":
            mode = "private"
            model = optional_params.pop("model_id", None)
            # private endpoint requires a dict instead of JSON
            instances = [optional_params.copy()]
            instances[0]["prompt"] = prompt
            llm_model = aiplatform.PrivateEndpoint(
                endpoint_name=model,
                project=vertex_project,
                location=vertex_location,
            )
            request_str += f"llm_model = aiplatform.PrivateEndpoint(endpoint_name={model}, project={vertex_project}, location={vertex_location})\n"
        elif model == "dedicated" or (
            optional_params.get("model_id")
            and "mg-endpoint-" in optional_params.get("model_id", "")
        ):
            mode = "dedicated"
            model = optional_params.pop("model_id", model)
            instances = [optional_params.copy()]
            instances[0]["prompt"] = prompt
            print_verbose(
                f"endpoint_name: {model}, project: {vertex_project}, location: {vertex_location}"
            )
            # No need to initialize aiplatform.Endpoint - we'll use direct HTTP requests
            llm_model = None  # We'll use the HTTP client instead
            request_str += (
                "# Using direct HTTP requests instead of aiplatform.Endpoint\n"
            )
        else:  # assume vertex model garden on public endpoint
            mode = "custom"

            instances = [optional_params.copy()]
            instances[0]["prompt"] = prompt
            instances = [
                json_format.ParseDict(instance_dict, Value())
                for instance_dict in instances
            ]
            # Will determine the API used based on async parameter
            llm_model = None

        # NOTE: async prediction and streaming under "private" mode isn't supported by aiplatform right now
        if acompletion is True:
            data = {
                "llm_model": llm_model,
                "mode": mode,
                "prompt": prompt,
                "logging_obj": logging_obj,
                "request_str": request_str,
                "model": model,
                "model_response": model_response,
                "encoding": encoding,
                "messages": messages,
                "print_verbose": print_verbose,
                "client_options": client_options,
                "instances": instances,
                "vertex_location": vertex_location,
                "vertex_project": vertex_project,
                "safety_settings": safety_settings,
                **optional_params,
            }
            if optional_params.get("stream", False) is True:
                # async streaming
                return async_streaming(**data)

            return async_completion(**data)

        completion_response = None

        stream = optional_params.pop(
            "stream", None
        )  # See note above on handling streaming for vertex ai
        if mode == "chat":
            chat = llm_model.start_chat()
            request_str += "chat = llm_model.start_chat()\n"

            if fake_stream is not True and stream is True:
                # NOTE: VertexAI does not accept stream=True as a param and raises an error,
                # we handle this by removing 'stream' from optional params and sending the request
                # after we get the response we add optional_params["stream"] = True, since main.py needs to know it's a streaming response to then transform it for the OpenAI format
                optional_params.pop(
                    "stream", None
                )  # vertex ai raises an error when passing stream in optional params

                request_str += (
                    f"chat.send_message_streaming({prompt}, **{optional_params})\n"
                )
                ## LOGGING
                logging_obj.pre_call(
                    input=prompt,
                    api_key=None,
                    additional_args={
                        "complete_input_dict": optional_params,
                        "request_str": request_str,
                    },
                )

                model_response = chat.send_message_streaming(prompt, **optional_params)

                return model_response

            request_str += f"chat.send_message({prompt}, **{optional_params}).text\n"
            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )
            completion_response = chat.send_message(prompt, **optional_params).text
        elif mode == "text":
            if fake_stream is not True and stream is True:
                request_str += (
                    f"llm_model.predict_streaming({prompt}, **{optional_params})\n"
                )
                ## LOGGING
                logging_obj.pre_call(
                    input=prompt,
                    api_key=None,
                    additional_args={
                        "complete_input_dict": optional_params,
                        "request_str": request_str,
                    },
                )
                model_response = llm_model.predict_streaming(prompt, **optional_params)

                return model_response

            request_str += f"llm_model.predict({prompt}, **{optional_params}).text\n"
            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )
            completion_response = llm_model.predict(prompt, **optional_params).text
        elif mode == "custom":
            """
            Vertex AI Model Garden
            """

            if vertex_project is None or vertex_location is None:
                raise ValueError(
                    "Vertex project and location are required for custom endpoint"
                )

            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )
            llm_model = aiplatform.gapic.PredictionServiceClient(
                client_options=client_options
            )
            request_str += f"llm_model = aiplatform.gapic.PredictionServiceClient(client_options={client_options})\n"
            endpoint_path = llm_model.endpoint_path(
                project=vertex_project, location=vertex_location, endpoint=model
            )
            request_str += (
                f"llm_model.predict(endpoint={endpoint_path}, instances={instances})\n"
            )
            response = llm_model.predict(
                endpoint=endpoint_path, instances=instances
            ).predictions

            completion_response = response[0]
            if (
                isinstance(completion_response, str)
                and "\nOutput:\n" in completion_response
            ):
                completion_response = completion_response.split("\nOutput:\n", 1)[1]
            if stream is True:
                response = TextStreamer(completion_response)
                return response
        elif mode == "private":
            """
            Vertex AI Model Garden deployed on private endpoint
            """
            if instances is None:
                raise ValueError("instances are required for private endpoint")
            if llm_model is None:
                raise ValueError("Unable to pick client for private endpoint")
            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )
            request_str += f"llm_model.predict(instances={instances})\n"
            response = llm_model.predict(instances=instances).predictions

            completion_response = response[0]
            if (
                isinstance(completion_response, str)
                and "\nOutput:\n" in completion_response
            ):
                completion_response = completion_response.split("\nOutput:\n", 1)[1]
            if stream is True:
                response = TextStreamer(completion_response)
                return response
        elif mode == "dedicated":
            """
            Vertex AI Model Garden deployed on dedicated endpoint
            """
            if instances is None:
                raise ValueError("instances are required for dedicated endpoint")

            # Prepare request details for curl logging BEFORE logging call
            url = get_dedicated_endpoint_url(model, vertex_project, vertex_location)
            request_body = {"instances": [instances[0]]}
            headers = {"Content-Type": "application/json"}

            # Update request_str for curl logging
            request_str = f"curl -X POST \\\n{url} \\\n"
            for k, v in headers.items():
                request_str += f"-H '{k}: {v}' \\\n"
            request_str += f"-d '{json.dumps(request_body)}'\n"

            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )

            # Use direct HTTP request instead of aiplatform.Endpoint.predict()
            response = make_vertex_ai_prediction_request(
                endpoint_name=model,
                instances=[instances[0]],  # Convert back from dict format
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                parameters={},
            )

            completion_response = response["predictions"][0]
            if (
                isinstance(completion_response, str)
                and "\nOutput:\n" in completion_response
            ):
                completion_response = completion_response.split("\nOutput:\n", 1)[1]
            if stream is True:
                response = TextStreamer(completion_response)
                return response

        ## LOGGING
        logging_obj.post_call(
            input=prompt, api_key=None, original_response=completion_response
        )

        ## RESPONSE OBJECT
        if isinstance(completion_response, litellm.Message):
            model_response.choices[0].message = completion_response  # type: ignore
        elif len(str(completion_response)) > 0:
            model_response.choices[0].message.content = str(completion_response)  # type: ignore
        model_response.created = int(time.time())
        model_response.model = model
        ## CALCULATING USAGE
        if model in litellm.vertex_language_models and response_obj is not None:
            model_response.choices[0].finish_reason = map_finish_reason(
                response_obj.candidates[0].finish_reason.name
            )
            usage = Usage(
                prompt_tokens=response_obj.usage_metadata.prompt_token_count,
                completion_tokens=response_obj.usage_metadata.candidates_token_count,
                total_tokens=response_obj.usage_metadata.total_token_count,
            )
        else:
            # init prompt tokens
            # this block attempts to get usage from response_obj if it exists, if not it uses the litellm token counter
            prompt_tokens, completion_tokens, _ = 0, 0, 0
            if response_obj is not None:
                if hasattr(response_obj, "usage_metadata") and hasattr(
                    response_obj.usage_metadata, "prompt_token_count"
                ):
                    prompt_tokens = response_obj.usage_metadata.prompt_token_count
                    completion_tokens = (
                        response_obj.usage_metadata.candidates_token_count
                    )
            else:
                prompt_tokens = len(encoding.encode(prompt))
                completion_tokens = len(
                    encoding.encode(
                        model_response["choices"][0]["message"].get("content", "")
                    )
                )

            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        setattr(model_response, "usage", usage)

        if fake_stream is True and stream is True:
            return ModelResponseIterator(model_response)
        return model_response
    except Exception as e:
        if isinstance(e, VertexAIError):
            raise e
        raise litellm.APIConnectionError(
            message=str(e), llm_provider="vertex_ai", model=model
        )


async def async_completion(  # noqa: PLR0915
    llm_model,
    mode: str,
    prompt: str,
    model: str,
    messages: list,
    model_response: ModelResponse,
    request_str: str,
    print_verbose: Callable,
    logging_obj,
    encoding,
    client_options=None,
    instances=None,
    vertex_project=None,
    vertex_location=None,
    safety_settings=None,
    **optional_params,
):
    """
    Add support for acompletion calls for gemini-pro
    """
    try:
        response_obj = None
        completion_response = None
        if mode == "chat":
            # chat-bison etc.
            chat = llm_model.start_chat()
            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )
            response_obj = await chat.send_message_async(prompt, **optional_params)
            completion_response = response_obj.text
        elif mode == "text":
            # gecko etc.
            request_str += f"llm_model.predict({prompt}, **{optional_params}).text\n"
            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )
            response_obj = await llm_model.predict_async(prompt, **optional_params)
            completion_response = response_obj.text
        elif mode == "custom":
            """
            Vertex AI Model Garden
            """
            from google.cloud import aiplatform  # type: ignore

            if vertex_project is None or vertex_location is None:
                raise ValueError(
                    "Vertex project and location are required for custom endpoint"
                )

            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )

            llm_model = aiplatform.gapic.PredictionServiceAsyncClient(
                client_options=client_options
            )
            request_str += f"llm_model = aiplatform.gapic.PredictionServiceAsyncClient(client_options={client_options})\n"
            endpoint_path = llm_model.endpoint_path(
                project=vertex_project, location=vertex_location, endpoint=model
            )
            request_str += (
                f"llm_model.predict(endpoint={endpoint_path}, instances={instances})\n"
            )
            response_obj = await llm_model.predict(
                endpoint=endpoint_path,
                instances=instances,
            )
            response = response_obj.predictions
            completion_response = response[0]
            if (
                isinstance(completion_response, str)
                and "\nOutput:\n" in completion_response
            ):
                completion_response = completion_response.split("\nOutput:\n", 1)[1]

        elif mode == "private":
            request_str += f"llm_model.predict_async(instances={instances})\n"
            response_obj = await llm_model.predict_async(
                instances=instances,
            )

            response = response_obj.predictions
            completion_response = response[0]
            if (
                isinstance(completion_response, str)
                and "\nOutput:\n" in completion_response
            ):
                completion_response = completion_response.split("\nOutput:\n", 1)[1]

        elif mode == "dedicated":
            # Use async HTTP requests for dedicated endpoints
            if instances is None:
                raise ValueError("instances are required for dedicated endpoint")

            # Prepare request details for curl logging BEFORE logging call
            url = get_dedicated_endpoint_url(model, vertex_project, vertex_location)
            request_body = {"instances": [instances[0]]}
            headers = {"Content-Type": "application/json"}

            # Update request_str for curl logging
            request_str = f"curl -X POST \\\n{url} \\\n"
            for k, v in headers.items():
                request_str += f"-H '{k}: {v}' \\\n"
            request_str += f"-d '{json.dumps(request_body)}'\n"

            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )

            # Use async HTTP request instead of aiplatform.Endpoint.predict()
            response = await make_vertex_ai_prediction_request_async(
                endpoint_name=model,
                instances=[instances[0]],  # Convert back from dict format
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                parameters={},
            )

            completion_response = response["predictions"][0]
            if (
                isinstance(completion_response, str)
                and "\nOutput:\n" in completion_response
            ):
                completion_response = completion_response.split("\nOutput:\n", 1)[1]

            # Return the response directly for async completion
            return completion_response

        ## LOGGING
        logging_obj.post_call(
            input=prompt, api_key=None, original_response=completion_response
        )

        ## RESPONSE OBJECT
        if isinstance(completion_response, litellm.Message):
            model_response.choices[0].message = completion_response  # type: ignore
        elif len(str(completion_response)) > 0:
            model_response.choices[0].message.content = str(  # type: ignore
                completion_response
            )
        model_response.created = int(time.time())
        model_response.model = model
        ## CALCULATING USAGE
        if model in litellm.vertex_language_models and response_obj is not None:
            model_response.choices[0].finish_reason = map_finish_reason(
                response_obj.candidates[0].finish_reason.name
            )
            usage = Usage(
                prompt_tokens=response_obj.usage_metadata.prompt_token_count,
                completion_tokens=response_obj.usage_metadata.candidates_token_count,
                total_tokens=response_obj.usage_metadata.total_token_count,
            )
        else:
            # init prompt tokens
            # this block attempts to get usage from response_obj if it exists, if not it uses the litellm token counter
            prompt_tokens, completion_tokens, _ = 0, 0, 0
            if response_obj is not None and (
                hasattr(response_obj, "usage_metadata")
                and hasattr(response_obj.usage_metadata, "prompt_token_count")
            ):
                prompt_tokens = response_obj.usage_metadata.prompt_token_count
                completion_tokens = response_obj.usage_metadata.candidates_token_count
            else:
                prompt_tokens = len(encoding.encode(prompt))
                completion_tokens = len(
                    encoding.encode(
                        model_response["choices"][0]["message"].get("content", "")
                    )
                )

            # set usage
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        setattr(model_response, "usage", usage)
        return model_response
    except Exception as e:
        raise VertexAIError(status_code=500, message=str(e))


async def async_streaming(  # noqa: PLR0915
    llm_model,
    mode: str,
    prompt: str,
    model: str,
    model_response: ModelResponse,
    messages: list,
    print_verbose: Callable,
    logging_obj,
    request_str: str,
    encoding=None,
    client_options=None,
    instances=None,
    vertex_project=None,
    vertex_location=None,
    safety_settings=None,
    **optional_params,
):
    """
    Add support for async streaming calls for gemini-pro
    """
    response: Any = None
    if mode == "chat":
        chat = llm_model.start_chat()
        optional_params.pop(
            "stream", None
        )  # vertex ai raises an error when passing stream in optional params
        request_str += (
            f"chat.send_message_streaming_async({prompt}, **{optional_params})\n"
        )
        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
                "request_str": request_str,
            },
        )
        response = chat.send_message_streaming_async(prompt, **optional_params)

    elif mode == "text":
        optional_params.pop(
            "stream", None
        )  # See note above on handling streaming for vertex ai
        request_str += (
            f"llm_model.predict_streaming_async({prompt}, **{optional_params})\n"
        )
        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
                "request_str": request_str,
            },
        )
        response = llm_model.predict_streaming_async(prompt, **optional_params)
    elif mode == "custom":
        from google.cloud import aiplatform  # type: ignore

        if vertex_project is None or vertex_location is None:
            raise ValueError(
                "Vertex project and location are required for custom endpoint"
            )

        stream = optional_params.pop("stream", None)

        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
                "request_str": request_str,
            },
        )
        llm_model = aiplatform.gapic.PredictionServiceAsyncClient(
            client_options=client_options
        )
        request_str += f"llm_model = aiplatform.gapic.PredictionServiceAsyncClient(client_options={client_options})\n"
        endpoint_path = llm_model.endpoint_path(
            project=vertex_project, location=vertex_location, endpoint=model
        )
        request_str += (
            f"client.predict(endpoint={endpoint_path}, instances={instances})\n"
        )
        response_obj = await llm_model.predict(
            endpoint=endpoint_path,
            instances=instances,
        )

        response = response_obj.predictions
        completion_response = response[0]
        if (
            isinstance(completion_response, str)
            and "\nOutput:\n" in completion_response
        ):
            completion_response = completion_response.split("\nOutput:\n", 1)[1]
        if stream:
            response = TextStreamer(completion_response)

    elif mode == "private":
        if instances is None:
            raise ValueError("Instances are required for private endpoint")
        stream = optional_params.pop("stream", None)
        _ = instances[0].pop("stream", None)
        request_str += f"llm_model.predict_async(instances={instances})\n"
        response_obj = await llm_model.predict_async(
            instances=instances,
        )
        response = response_obj.predictions
        completion_response = response[0]
        if (
            isinstance(completion_response, str)
            and "\nOutput:\n" in completion_response
        ):
            completion_response = completion_response.split("\nOutput:\n", 1)[1]
        if stream:
            response = TextStreamer(completion_response)

    elif mode == "dedicated":
        # Use HTTP requests for dedicated endpoints
        if instances is None:
            raise ValueError("Instances are required for dedicated endpoint")

        stream = optional_params.pop("stream", None)

        # Prepare request details for curl logging BEFORE logging call
        url = get_dedicated_endpoint_url(model, vertex_project, vertex_location)
        request_body = {"instances": [instances[0]]}
        headers = {"Content-Type": "application/json"}

        # Update request_str for curl logging
        request_str = f"curl -X POST \\\n{url} \\\n"
        for k, v in headers.items():
            request_str += f"-H '{k}: {v}' \\\n"
        request_str += f"-d '{json.dumps(request_body)}'\n"

        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
                "request_str": request_str,
            },
        )

        # Use direct HTTP request instead of aiplatform.Endpoint.predict()
        response = make_vertex_ai_prediction_request(
            endpoint_name=model,
            instances=[instances[0]],  # Convert back from dict format
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            parameters={},
        )

        completion_response = response["predictions"][0]
        if (
            isinstance(completion_response, str)
            and "\nOutput:\n" in completion_response
        ):
            completion_response = completion_response.split("\nOutput:\n", 1)[1]

        # Use TextStreamer for fake streaming
        if stream:
            response = TextStreamer(completion_response)
        else:
            response = completion_response

    if response is None:
        raise ValueError("Unable to generate response")

    logging_obj.post_call(input=prompt, api_key=None, original_response=response)

    streamwrapper = CustomStreamWrapper(
        completion_stream=response,
        model=model,
        custom_llm_provider="vertex_ai",
        logging_obj=logging_obj,
    )

    return streamwrapper
