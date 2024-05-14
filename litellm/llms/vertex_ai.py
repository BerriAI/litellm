import os, types
import json
from enum import Enum
import requests  # type: ignore
import time
from typing import Callable, Optional, Union, List
from litellm.utils import ModelResponse, Usage, CustomStreamWrapper, map_finish_reason
import litellm, uuid
import httpx, inspect  # type: ignore


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


class ExtendedGenerationConfig(dict):
    """Extended parameters for the generation."""

    def __init__(
        self,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        candidate_count: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        response_mime_type: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ):
        super().__init__(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            candidate_count=candidate_count,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences,
            response_mime_type=response_mime_type,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )


class VertexAIConfig:
    """
    Reference: https://cloud.google.com/vertex-ai/docs/generative-ai/chat/test-chat-prompts
    Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference

    The class `VertexAIConfig` provides configuration for the VertexAI's API interface. Below are the parameters:

    - `temperature` (float): This controls the degree of randomness in token selection.

    - `max_output_tokens` (integer): This sets the limitation for the maximum amount of token in the text output. In this case, the default value is 256.

    - `top_p` (float): The tokens are selected from the most probable to the least probable until the sum of their probabilities equals the `top_p` value. Default is 0.95.

    - `top_k` (integer): The value of `top_k` determines how many of the most probable tokens are considered in the selection. For example, a `top_k` of 1 means the selected token is the most probable among all tokens. The default value is 40.

    - `response_mime_type` (str): The MIME type of the response. The default value is 'text/plain'.

    - `candidate_count` (int): Number of generated responses to return.

    - `stop_sequences` (List[str]): The set of character sequences (up to 5) that will stop output generation. If specified, the API will stop at the first appearance of a stop sequence. The stop sequence will not be included as part of the response.

    - `frequency_penalty` (float): This parameter is used to penalize the model from repeating the same output. The default value is 0.0.

    - `presence_penalty` (float): This parameter is used to penalize the model from generating the same output as the input. The default value is 0.0.

    Note: Please make sure to modify the default parameters as required for your use case.
    """

    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    response_mime_type: Optional[str] = None
    candidate_count: Optional[int] = None
    stop_sequences: Optional[list] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        response_mime_type: Optional[str] = None,
        candidate_count: Optional[int] = None,
        stop_sequences: Optional[list] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self):
        return [
            "temperature",
            "top_p",
            "max_tokens",
            "stream",
            "tools",
            "tool_choice",
            "response_format",
            "n",
            "stop",
        ]

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if (
                param == "stream" and value == True
            ):  # sending stream = False, can cause it to get passed unchecked and raise issues
                optional_params["stream"] = value
            if param == "n":
                optional_params["candidate_count"] = value
            if param == "stop":
                if isinstance(value, str):
                    optional_params["stop_sequences"] = [value]
                elif isinstance(value, list):
                    optional_params["stop_sequences"] = value
            if param == "max_tokens":
                optional_params["max_output_tokens"] = value
            if param == "response_format" and value["type"] == "json_object":
                optional_params["response_mime_type"] = "application/json"
            if param == "frequency_penalty":
                optional_params["frequency_penalty"] = value
            if param == "presence_penalty":
                optional_params["presence_penalty"] = value
            if param == "tools" and isinstance(value, list):
                from vertexai.preview import generative_models

                gtool_func_declarations = []
                for tool in value:
                    gtool_func_declaration = generative_models.FunctionDeclaration(
                        name=tool["function"]["name"],
                        description=tool["function"].get("description", ""),
                        parameters=tool["function"].get("parameters", {}),
                    )
                    gtool_func_declarations.append(gtool_func_declaration)
                optional_params["tools"] = [
                    generative_models.Tool(
                        function_declarations=gtool_func_declarations
                    )
                ]
            if param == "tool_choice" and (
                isinstance(value, str) or isinstance(value, dict)
            ):
                pass
        return optional_params

    def get_mapped_special_auth_params(self) -> dict:
        """
        Common auth params across bedrock/vertex_ai/azure/watsonx
        """
        return {"project": "vertex_project", "region_name": "vertex_location"}

    def map_special_auth_params(self, non_default_params: dict, optional_params: dict):
        mapped_params = self.get_mapped_special_auth_params()

        for param, value in non_default_params.items():
            if param in mapped_params:
                optional_params[mapped_params[param]] = value
        return optional_params

    def get_eu_regions(self) -> List[str]:
        """
        Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
        """
        return [
            "europe-central2",
            "europe-north1",
            "europe-southwest1",
            "europe-west1",
            "europe-west2",
            "europe-west3",
            "europe-west4",
            "europe-west6",
            "europe-west8",
            "europe-west9",
        ]


import asyncio


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


def _get_image_bytes_from_url(image_url: str) -> bytes:
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
        image_bytes = response.content
        return image_bytes
    except requests.exceptions.RequestException as e:
        raise Exception(f"An exception occurs with this image - {str(e)}")


def _load_image_from_url(image_url: str):
    """
    Loads an image from a URL.

    Args:
        image_url (str): The URL of the image.

    Returns:
        Image: The loaded image.
    """
    from vertexai.preview.generative_models import (
        GenerativeModel,
        Part,
        GenerationConfig,
        Image,
    )

    image_bytes = _get_image_bytes_from_url(image_url)

    return Image.from_bytes(data=image_bytes)


def _gemini_vision_convert_messages(messages: list):
    """
    Converts given messages for GPT-4 Vision to Gemini format.

    Args:
        messages (list): The messages to convert. Each message can be a dictionary with a "content" key. The content can be a string or a list of elements. If it is a string, it will be concatenated to the prompt. If it is a list, each element will be processed based on its type:
            - If the element is a dictionary with a "type" key equal to "text", its "text" value will be concatenated to the prompt.
            - If the element is a dictionary with a "type" key equal to "image_url", its "image_url" value will be added to the list of images.

    Returns:
        tuple: A tuple containing the prompt (a string) and the processed images (a list of objects representing the images).

    Raises:
        VertexAIError: If the import of the 'vertexai' module fails, indicating that 'google-cloud-aiplatform' needs to be installed.
        Exception: If any other exception occurs during the execution of the function.

    Note:
        This function is based on the code from the 'gemini/getting-started/intro_gemini_python.ipynb' notebook in the 'generative-ai' repository on GitHub.
        The supported MIME types for images include 'image/png' and 'image/jpeg'.

    Examples:
        >>> messages = [
        ...     {"content": "Hello, world!"},
        ...     {"content": [{"type": "text", "text": "This is a text message."}, {"type": "image_url", "image_url": "example.com/image.png"}]},
        ... ]
        >>> _gemini_vision_convert_messages(messages)
        ('Hello, world!This is a text message.', [<Part object>, <Part object>])
    """
    try:
        import vertexai
    except:
        raise VertexAIError(
            status_code=400,
            message="vertexai import failed please run `pip install google-cloud-aiplatform`",
        )
    try:
        from vertexai.preview.language_models import (
            ChatModel,
            CodeChatModel,
            InputOutputTextPair,
        )
        from vertexai.language_models import TextGenerationModel, CodeGenerationModel
        from vertexai.preview.generative_models import (
            GenerativeModel,
            Part,
            GenerationConfig,
            Image,
        )

        # given messages for gpt-4 vision, convert them for gemini
        # https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_python.ipynb
        prompt = ""
        images = []
        for message in messages:
            if isinstance(message["content"], str):
                prompt += message["content"]
            elif isinstance(message["content"], list):
                # see https://docs.litellm.ai/docs/providers/openai#openai-vision-models
                for element in message["content"]:
                    if isinstance(element, dict):
                        if element["type"] == "text":
                            prompt += element["text"]
                        elif element["type"] == "image_url":
                            image_url = element["image_url"]["url"]
                            images.append(image_url)
        # processing images passed to gemini
        processed_images = []
        for img in images:
            if "gs://" in img:
                # Case 1: Images with Cloud Storage URIs
                # The supported MIME types for images include image/png and image/jpeg.
                part_mime = "image/png" if "png" in img else "image/jpeg"
                google_clooud_part = Part.from_uri(img, mime_type=part_mime)
                processed_images.append(google_clooud_part)
            elif "https:/" in img:
                # Case 2: Images with direct links
                image = _load_image_from_url(img)
                processed_images.append(image)
            elif ".mp4" in img and "gs://" in img:
                # Case 3: Videos with Cloud Storage URIs
                part_mime = "video/mp4"
                google_clooud_part = Part.from_uri(img, mime_type=part_mime)
                processed_images.append(google_clooud_part)
            elif "base64" in img:
                # Case 4: Images with base64 encoding
                import base64, re

                # base 64 is passed as data:image/jpeg;base64,<base-64-encoded-image>
                image_metadata, img_without_base_64 = img.split(",")

                # read mime_type from img_without_base_64=data:image/jpeg;base64
                # Extract MIME type using regular expression
                mime_type_match = re.match(r"data:(.*?);base64", image_metadata)

                if mime_type_match:
                    mime_type = mime_type_match.group(1)
                else:
                    mime_type = "image/jpeg"
                decoded_img = base64.b64decode(img_without_base_64)
                processed_image = Part.from_data(data=decoded_img, mime_type=mime_type)
                processed_images.append(processed_image)
        return prompt, processed_images
    except Exception as e:
        raise e


def completion(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    logging_obj,
    vertex_project=None,
    vertex_location=None,
    vertex_credentials=None,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
    acompletion: bool = False,
):
    try:
        import vertexai
    except:
        raise VertexAIError(
            status_code=400,
            message="vertexai import failed please run `pip install google-cloud-aiplatform`",
        )

    if not (
        hasattr(vertexai, "preview") or hasattr(vertexai.preview, "language_models")
    ):
        raise VertexAIError(
            status_code=400,
            message="""Upgrade vertex ai. Run `pip install "google-cloud-aiplatform>=1.38"`""",
        )
    try:
        from vertexai.preview.language_models import (
            ChatModel,
            CodeChatModel,
            InputOutputTextPair,
        )
        from vertexai.language_models import TextGenerationModel, CodeGenerationModel
        from vertexai.preview.generative_models import (
            GenerativeModel,
            Part,
            GenerationConfig,
        )
        from google.cloud import aiplatform  # type: ignore
        from google.protobuf import json_format  # type: ignore
        from google.protobuf.struct_pb2 import Value  # type: ignore
        from google.cloud.aiplatform_v1beta1.types import content as gapic_content_types  # type: ignore
        import google.auth  # type: ignore
        import proto  # type: ignore

        ## Load credentials with the correct quota project ref: https://github.com/googleapis/python-aiplatform/issues/2557#issuecomment-1709284744
        print_verbose(
            f"VERTEX AI: vertex_project={vertex_project}; vertex_location={vertex_location}"
        )
        if vertex_credentials is not None and isinstance(vertex_credentials, str):
            import google.oauth2.service_account

            json_obj = json.loads(vertex_credentials)

            creds = google.oauth2.service_account.Credentials.from_service_account_info(
                json_obj,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            creds, _ = google.auth.default(quota_project_id=vertex_project)
        print_verbose(
            f"VERTEX AI: creds={creds}; google application credentials: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}"
        )
        vertexai.init(
            project=vertex_project, location=vertex_location, credentials=creds
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
                message["content"]
                for message in messages
                if isinstance(message["content"], str)
            ]
        )

        mode = ""

        request_str = ""
        response_obj = None
        async_client = None
        instances = None
        client_options = {
            "api_endpoint": f"{vertex_location}-aiplatform.googleapis.com"
        }
        if (
            model in litellm.vertex_language_models
            or model in litellm.vertex_vision_models
        ):
            llm_model = GenerativeModel(model)
            mode = "vision"
            request_str += f"llm_model = GenerativeModel({model})\n"
        elif model in litellm.vertex_chat_models:
            llm_model = ChatModel.from_pretrained(model)
            mode = "chat"
            request_str += f"llm_model = ChatModel.from_pretrained({model})\n"
        elif model in litellm.vertex_text_models:
            llm_model = TextGenerationModel.from_pretrained(model)
            mode = "text"
            request_str += f"llm_model = TextGenerationModel.from_pretrained({model})\n"
        elif model in litellm.vertex_code_text_models:
            llm_model = CodeGenerationModel.from_pretrained(model)
            mode = "text"
            request_str += f"llm_model = CodeGenerationModel.from_pretrained({model})\n"
        elif model in litellm.vertex_code_chat_models:  # vertex_code_llm_models
            llm_model = CodeChatModel.from_pretrained(model)
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
        if acompletion == True:
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

        if mode == "vision":
            print_verbose("\nMaking VertexAI Gemini Pro / Pro Vision Call")
            print_verbose(f"\nProcessing input messages = {messages}")
            tools = optional_params.pop("tools", None)
            prompt, images = _gemini_vision_convert_messages(messages=messages)
            content = [prompt] + images
            stream = optional_params.pop("stream", False)
            if stream == True:

                request_str += f"response = llm_model.generate_content({content}, generation_config=GenerationConfig(**{optional_params}), safety_settings={safety_settings}, stream={stream})\n"
                logging_obj.pre_call(
                    input=prompt,
                    api_key=None,
                    additional_args={
                        "complete_input_dict": optional_params,
                        "request_str": request_str,
                    },
                )

                model_response = llm_model.generate_content(
                    contents=content,
                    generation_config=optional_params,
                    safety_settings=safety_settings,
                    stream=True,
                    tools=tools,
                )

                return model_response

            request_str += f"response = llm_model.generate_content({content})\n"
            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )

            ## LLM Call
            response = llm_model.generate_content(
                contents=content,
                generation_config=optional_params,
                safety_settings=safety_settings,
                tools=tools,
            )

            if tools is not None and bool(
                getattr(response.candidates[0].content.parts[0], "function_call", None)
            ):
                function_call = response.candidates[0].content.parts[0].function_call
                args_dict = {}

                # Check if it's a RepeatedComposite instance
                for key, val in function_call.args.items():
                    if isinstance(
                        val, proto.marshal.collections.repeated.RepeatedComposite
                    ):
                        # If so, convert to list
                        args_dict[key] = [v for v in val]
                    else:
                        args_dict[key] = val

                try:
                    args_str = json.dumps(args_dict)
                except Exception as e:
                    raise VertexAIError(status_code=422, message=str(e))
                message = litellm.Message(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{str(uuid.uuid4())}",
                            "function": {
                                "arguments": args_str,
                                "name": function_call.name,
                            },
                            "type": "function",
                        }
                    ],
                )
                completion_response = message
            else:
                completion_response = response.text
            response_obj = response._raw_response
            optional_params["tools"] = tools
        elif mode == "chat":
            chat = llm_model.start_chat()
            request_str += f"chat = llm_model.start_chat()\n"

            if "stream" in optional_params and optional_params["stream"] == True:
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
            if "stream" in optional_params and optional_params["stream"] == True:
                optional_params.pop(
                    "stream", None
                )  # See note above on handling streaming for vertex ai
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
            if "stream" in optional_params and optional_params["stream"] == True:
                response = TextStreamer(completion_response)
                return response
        elif mode == "private":
            """
            Vertex AI Model Garden deployed on private endpoint
            """
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
            if "stream" in optional_params and optional_params["stream"] == True:
                response = TextStreamer(completion_response)
                return response

        ## LOGGING
        logging_obj.post_call(
            input=prompt, api_key=None, original_response=completion_response
        )

        ## RESPONSE OBJECT
        if isinstance(completion_response, litellm.Message):
            model_response["choices"][0]["message"] = completion_response
        elif len(str(completion_response)) > 0:
            model_response["choices"][0]["message"]["content"] = str(
                completion_response
            )
        model_response["created"] = int(time.time())
        model_response["model"] = model
        ## CALCULATING USAGE
        if model in litellm.vertex_language_models and response_obj is not None:
            model_response["choices"][0].finish_reason = map_finish_reason(
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
            prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
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
        return model_response
    except Exception as e:
        if isinstance(e, VertexAIError):
            raise e
        raise VertexAIError(status_code=500, message=str(e))


async def async_completion(
    llm_model,
    mode: str,
    prompt: str,
    model: str,
    model_response: ModelResponse,
    logging_obj=None,
    request_str=None,
    encoding=None,
    messages=None,
    print_verbose=None,
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
        import proto  # type: ignore

        if mode == "vision":
            print_verbose("\nMaking VertexAI Gemini Pro/Vision Call")
            print_verbose(f"\nProcessing input messages = {messages}")
            tools = optional_params.pop("tools", None)
            stream = optional_params.pop("stream", False)

            prompt, images = _gemini_vision_convert_messages(messages=messages)
            content = [prompt] + images

            request_str += f"response = llm_model.generate_content({content})\n"
            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )

            ## LLM Call
            # print(f"final content: {content}")
            response = await llm_model._generate_content_async(
                contents=content,
                generation_config=optional_params,
                safety_settings=safety_settings,
                tools=tools,
            )

            if tools is not None and bool(
                getattr(response.candidates[0].content.parts[0], "function_call", None)
            ):
                function_call = response.candidates[0].content.parts[0].function_call
                args_dict = {}

                # Check if it's a RepeatedComposite instance
                for key, val in function_call.args.items():
                    if isinstance(
                        val, proto.marshal.collections.repeated.RepeatedComposite
                    ):
                        # If so, convert to list
                        args_dict[key] = [v for v in val]
                    else:
                        args_dict[key] = val

                try:
                    args_str = json.dumps(args_dict)
                except Exception as e:
                    raise VertexAIError(status_code=422, message=str(e))
                message = litellm.Message(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{str(uuid.uuid4())}",
                            "function": {
                                "arguments": args_str,
                                "name": function_call.name,
                            },
                            "type": "function",
                        }
                    ],
                )
                completion_response = message
            else:
                completion_response = response.text
            response_obj = response._raw_response
            optional_params["tools"] = tools
        elif mode == "chat":
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
            model_response["choices"][0]["message"] = completion_response
        elif len(str(completion_response)) > 0:
            model_response["choices"][0]["message"]["content"] = str(
                completion_response
            )
        model_response["created"] = int(time.time())
        model_response["model"] = model
        ## CALCULATING USAGE
        if model in litellm.vertex_language_models and response_obj is not None:
            model_response["choices"][0].finish_reason = map_finish_reason(
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
            prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
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


async def async_streaming(
    llm_model,
    mode: str,
    prompt: str,
    model: str,
    model_response: ModelResponse,
    logging_obj=None,
    request_str=None,
    encoding=None,
    messages=None,
    print_verbose=None,
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
    if mode == "vision":
        stream = optional_params.pop("stream")
        tools = optional_params.pop("tools", None)
        print_verbose("\nMaking VertexAI Gemini Pro Vision Call")
        print_verbose(f"\nProcessing input messages = {messages}")

        prompt, images = _gemini_vision_convert_messages(messages=messages)
        content = [prompt] + images
        request_str += f"response = llm_model.generate_content({content}, generation_config=GenerationConfig(**{optional_params}), stream={stream})\n"
        logging_obj.pre_call(
            input=prompt,
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
                "request_str": request_str,
            },
        )

        response = await llm_model._generate_content_streaming_async(
            contents=content,
            generation_config=optional_params,
            safety_settings=safety_settings,
            tools=tools,
        )

    elif mode == "chat":
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

    logging_obj.post_call(input=prompt, api_key=None, original_response=response)

    streamwrapper = CustomStreamWrapper(
        completion_stream=response,
        model=model,
        custom_llm_provider="vertex_ai",
        logging_obj=logging_obj,
    )

    return streamwrapper


def embedding(
    model: str,
    input: Union[list, str],
    api_key: Optional[str] = None,
    logging_obj=None,
    model_response=None,
    optional_params=None,
    encoding=None,
    vertex_project=None,
    vertex_location=None,
    vertex_credentials=None,
    aembedding=False,
    print_verbose=None,
):
    # logic for parsing in - calling - parsing out model embedding calls
    try:
        import vertexai
    except:
        raise VertexAIError(
            status_code=400,
            message="vertexai import failed please run `pip install google-cloud-aiplatform`",
        )

    from vertexai.language_models import TextEmbeddingModel
    import google.auth  # type: ignore

    ## Load credentials with the correct quota project ref: https://github.com/googleapis/python-aiplatform/issues/2557#issuecomment-1709284744
    try:
        print_verbose(
            f"VERTEX AI: vertex_project={vertex_project}; vertex_location={vertex_location}"
        )
        if vertex_credentials is not None and isinstance(vertex_credentials, str):
            import google.oauth2.service_account

            json_obj = json.loads(vertex_credentials)

            creds = google.oauth2.service_account.Credentials.from_service_account_info(
                json_obj,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            creds, _ = google.auth.default(quota_project_id=vertex_project)
        print_verbose(
            f"VERTEX AI: creds={creds}; google application credentials: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}"
        )
        vertexai.init(
            project=vertex_project, location=vertex_location, credentials=creds
        )
    except Exception as e:
        raise VertexAIError(status_code=401, message=str(e))

    if isinstance(input, str):
        input = [input]

    try:
        llm_model = TextEmbeddingModel.from_pretrained(model)
    except Exception as e:
        raise VertexAIError(status_code=422, message=str(e))

    if aembedding == True:
        return async_embedding(
            model=model,
            client=llm_model,
            input=input,
            logging_obj=logging_obj,
            model_response=model_response,
            optional_params=optional_params,
            encoding=encoding,
        )

    request_str = f"""embeddings = llm_model.get_embeddings({input})"""
    ## LOGGING PRE-CALL
    logging_obj.pre_call(
        input=input,
        api_key=None,
        additional_args={
            "complete_input_dict": optional_params,
            "request_str": request_str,
        },
    )

    try:
        embeddings = llm_model.get_embeddings(input)
    except Exception as e:
        raise VertexAIError(status_code=500, message=str(e))

    ## LOGGING POST-CALL
    logging_obj.post_call(input=input, api_key=None, original_response=embeddings)
    ## Populate OpenAI compliant dictionary
    embedding_response = []
    for idx, embedding in enumerate(embeddings):
        embedding_response.append(
            {
                "object": "embedding",
                "index": idx,
                "embedding": embedding.values,
            }
        )
    model_response["object"] = "list"
    model_response["data"] = embedding_response
    model_response["model"] = model
    input_tokens = 0

    input_str = "".join(input)

    input_tokens += len(encoding.encode(input_str))

    usage = Usage(
        prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens
    )
    model_response.usage = usage

    return model_response


async def async_embedding(
    model: str,
    input: Union[list, str],
    logging_obj=None,
    model_response=None,
    optional_params=None,
    encoding=None,
    client=None,
):
    """
    Async embedding implementation
    """
    request_str = f"""embeddings = llm_model.get_embeddings({input})"""
    ## LOGGING PRE-CALL
    logging_obj.pre_call(
        input=input,
        api_key=None,
        additional_args={
            "complete_input_dict": optional_params,
            "request_str": request_str,
        },
    )

    try:
        embeddings = await client.get_embeddings_async(input)
    except Exception as e:
        raise VertexAIError(status_code=500, message=str(e))

    ## LOGGING POST-CALL
    logging_obj.post_call(input=input, api_key=None, original_response=embeddings)
    ## Populate OpenAI compliant dictionary
    embedding_response = []
    for idx, embedding in enumerate(embeddings):
        embedding_response.append(
            {
                "object": "embedding",
                "index": idx,
                "embedding": embedding.values,
            }
        )
    model_response["object"] = "list"
    model_response["data"] = embedding_response
    model_response["model"] = model
    input_tokens = 0

    input_str = "".join(input)

    input_tokens += len(encoding.encode(input_str))

    usage = Usage(
        prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens
    )
    model_response.usage = usage

    return model_response
