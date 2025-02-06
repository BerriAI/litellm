import json
import os
import time
from typing import Any, Callable, Optional, cast

import httpx

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.bedrock.common_utils import ModelResponseIterator
from litellm.llms.custom_httpx.http_handler import _DEFAULT_TTL_FOR_HTTPX_CLIENTS
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
