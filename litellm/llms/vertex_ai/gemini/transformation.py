"""
Transformation logic from OpenAI format to Gemini format. 

Why separate file? Make it easy to see how transformation works
"""

import os
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union, cast

import httpx
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_to_anthropic_image_obj,
    convert_to_gemini_tool_call_invoke,
    convert_to_gemini_tool_call_result,
    response_schema_prompt,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.files import (
    get_file_mime_type_for_file_type,
    get_file_type_from_extension,
    is_gemini_1_5_accepted_file_type,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAudioObject,
    ChatCompletionFileObject,
    ChatCompletionImageObject,
    ChatCompletionTextObject,
)
from litellm.types.llms.vertex_ai import *
from litellm.types.llms.vertex_ai import (
    GenerationConfig,
    PartType,
    RequestBody,
    SafetSettingsConfig,
    SystemInstructions,
    ToolConfig,
    Tools,
)

from ..common_utils import (
    _check_text_in_content,
    get_supports_response_schema,
    get_supports_system_message,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


def _process_gemini_image(image_url: str, format: Optional[str] = None) -> PartType:
    """
    Given an image URL, return the appropriate PartType for Gemini
    """

    try:
        # GCS URIs
        if "gs://" in image_url:
            # Figure out file type
            extension_with_dot = os.path.splitext(image_url)[-1]  # Ex: ".png"
            extension = extension_with_dot[1:]  # Ex: "png"

            if not format:
                file_type = get_file_type_from_extension(extension)

                # Validate the file type is supported by Gemini
                if not is_gemini_1_5_accepted_file_type(file_type):
                    raise Exception(f"File type not supported by gemini - {file_type}")

                mime_type = get_file_mime_type_for_file_type(file_type)
            else:
                mime_type = format
            file_data = FileDataType(mime_type=mime_type, file_uri=image_url)

            return PartType(file_data=file_data)
        elif (
            "https://" in image_url
            and (image_type := format or _get_image_mime_type_from_url(image_url))
            is not None
        ):
            file_data = FileDataType(file_uri=image_url, mime_type=image_type)
            return PartType(file_data=file_data)
        elif "http://" in image_url or "https://" in image_url or "base64" in image_url:
            # https links for unsupported mime types and base64 images
            image = convert_to_anthropic_image_obj(image_url, format=format)
            _blob = BlobType(data=image["data"], mime_type=image["media_type"])
            return PartType(inline_data=_blob)
        raise Exception("Invalid image received - {}".format(image_url))
    except Exception as e:
        raise e


def _get_image_mime_type_from_url(url: str) -> Optional[str]:
    """
    Get mime type for common image URLs
    See gemini mime types: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding#image-requirements

    Supported by Gemini:
     application/pdf
    audio/mpeg
    audio/mp3
    audio/wav
    image/png
    image/jpeg
    image/webp
    text/plain
    video/mov
    video/mpeg
    video/mp4
    video/mpg
    video/avi
    video/wmv
    video/mpegps
    video/flv
    """
    url = url.lower()

    # Map file extensions to mime types
    mime_types = {
        # Images
        (".jpg", ".jpeg"): "image/jpeg",
        (".png",): "image/png",
        (".webp",): "image/webp",
        # Videos
        (".mp4",): "video/mp4",
        (".mov",): "video/mov",
        (".mpeg", ".mpg"): "video/mpeg",
        (".avi",): "video/avi",
        (".wmv",): "video/wmv",
        (".mpegps",): "video/mpegps",
        (".flv",): "video/flv",
        # Audio
        (".mp3",): "audio/mp3",
        (".wav",): "audio/wav",
        (".mpeg",): "audio/mpeg",
        # Documents
        (".pdf",): "application/pdf",
        (".txt",): "text/plain",
    }

    # Check each extension group against the URL
    for extensions, mime_type in mime_types.items():
        if any(url.endswith(ext) for ext in extensions):
            return mime_type

    return None


def _gemini_convert_messages_with_history(  # noqa: PLR0915
    messages: List[AllMessageValues],
) -> List[ContentType]:
    """
    Converts given messages from OpenAI format to Gemini format

    - Parts must be iterable
    - Roles must alternate b/w 'user' and 'model' (same as anthropic -> merge consecutive roles)
    - Please ensure that function response turn comes immediately after a function call turn
    """
    user_message_types = {"user", "system"}
    contents: List[ContentType] = []

    last_message_with_tool_calls = None

    msg_i = 0
    tool_call_responses = []
    try:
        while msg_i < len(messages):
            user_content: List[PartType] = []
            init_msg_i = msg_i
            ## MERGE CONSECUTIVE USER CONTENT ##
            while (
                msg_i < len(messages) and messages[msg_i]["role"] in user_message_types
            ):
                _message_content = messages[msg_i].get("content")
                if _message_content is not None and isinstance(_message_content, list):
                    _parts: List[PartType] = []
                    for element_idx, element in enumerate(_message_content):
                        if (
                            element["type"] == "text"
                            and "text" in element
                            and len(element["text"]) > 0
                        ):
                            element = cast(ChatCompletionTextObject, element)
                            _part = PartType(text=element["text"])
                            _parts.append(_part)
                        elif element["type"] == "image_url":
                            element = cast(ChatCompletionImageObject, element)
                            img_element = element
                            format: Optional[str] = None
                            if isinstance(img_element["image_url"], dict):
                                image_url = img_element["image_url"]["url"]
                                format = img_element["image_url"].get("format")
                            else:
                                image_url = img_element["image_url"]
                            _part = _process_gemini_image(
                                image_url=image_url, format=format
                            )
                            _parts.append(_part)
                        elif element["type"] == "input_audio":
                            audio_element = cast(ChatCompletionAudioObject, element)
                            if audio_element["input_audio"].get("data") is not None:
                                _part = _process_gemini_image(
                                    image_url=audio_element["input_audio"]["data"],
                                    format=audio_element["input_audio"].get("format"),
                                )
                                _parts.append(_part)
                        elif element["type"] == "file":
                            file_element = cast(ChatCompletionFileObject, element)
                            file_id = file_element["file"].get("file_id")
                            format = file_element["file"].get("format")
                            file_data = file_element["file"].get("file_data")
                            passed_file = file_id or file_data
                            if passed_file is None:
                                raise Exception(
                                    "Unknown file type. Please pass in a file_id or file_data"
                                )
                            try:
                                _part = _process_gemini_image(
                                    image_url=passed_file, format=format
                                )
                                _parts.append(_part)
                            except Exception:
                                raise Exception(
                                    "Unable to determine mime type for file_id: {}, set this explicitly using message[{}].content[{}].file.format".format(
                                        file_id, msg_i, element_idx
                                    )
                                )
                    user_content.extend(_parts)
                elif (
                    _message_content is not None
                    and isinstance(_message_content, str)
                    and len(_message_content) > 0
                ):
                    _part = PartType(text=_message_content)
                    user_content.append(_part)

                msg_i += 1

            if user_content:
                """
                check that user_content has 'text' parameter.
                    - Known Vertex Error: Unable to submit request because it must have a text parameter.
                    - Relevant Issue: https://github.com/BerriAI/litellm/issues/5515
                """
                has_text_in_content = _check_text_in_content(user_content)
                if has_text_in_content is False:
                    verbose_logger.warning(
                        "No text in user content. Adding a blank text to user content, to ensure Gemini doesn't fail the request. Relevant Issue - https://github.com/BerriAI/litellm/issues/5515"
                    )
                    user_content.append(
                        PartType(text=" ")
                    )  # add a blank text, to ensure Gemini doesn't fail the request.
                contents.append(ContentType(role="user", parts=user_content))
            assistant_content = []
            ## MERGE CONSECUTIVE ASSISTANT CONTENT ##
            while msg_i < len(messages) and messages[msg_i]["role"] == "assistant":
                if isinstance(messages[msg_i], BaseModel):
                    msg_dict: Union[ChatCompletionAssistantMessage, dict] = messages[msg_i].model_dump()  # type: ignore
                else:
                    msg_dict = messages[msg_i]  # type: ignore
                assistant_msg = ChatCompletionAssistantMessage(**msg_dict)  # type: ignore
                _message_content = assistant_msg.get("content", None)
                if _message_content is not None and isinstance(_message_content, list):
                    _parts = []
                    for element in _message_content:
                        if isinstance(element, dict):
                            if element["type"] == "text":
                                _part = PartType(text=element["text"])
                                _parts.append(_part)
                    assistant_content.extend(_parts)
                elif (
                    _message_content is not None
                    and isinstance(_message_content, str)
                    and _message_content
                ):
                    assistant_text = _message_content  # either string or none
                    assistant_content.append(PartType(text=assistant_text))  # type: ignore

                ## HANDLE ASSISTANT FUNCTION CALL
                if (
                    assistant_msg.get("tool_calls", []) is not None
                    or assistant_msg.get("function_call") is not None
                ):  # support assistant tool invoke conversion
                    assistant_content.extend(
                        convert_to_gemini_tool_call_invoke(assistant_msg)
                    )
                    last_message_with_tool_calls = assistant_msg

                msg_i += 1

            if assistant_content:
                contents.append(ContentType(role="model", parts=assistant_content))

            ## APPEND TOOL CALL MESSAGES ##
            tool_call_message_roles = ["tool", "function"]
            if (
                msg_i < len(messages)
                and messages[msg_i]["role"] in tool_call_message_roles
            ):
                _part = convert_to_gemini_tool_call_result(
                    messages[msg_i], last_message_with_tool_calls  # type: ignore
                )
                msg_i += 1
                tool_call_responses.append(_part)
            if msg_i < len(messages) and (
                messages[msg_i]["role"] not in tool_call_message_roles
            ):
                if len(tool_call_responses) > 0:
                    contents.append(ContentType(parts=tool_call_responses))
                    tool_call_responses = []

            if msg_i == init_msg_i:  # prevent infinite loops
                raise Exception(
                    "Invalid Message passed in - {}. File an issue https://github.com/BerriAI/litellm/issues".format(
                        messages[msg_i]
                    )
                )
        if len(tool_call_responses) > 0:
            contents.append(ContentType(parts=tool_call_responses))
        return contents
    except Exception as e:
        raise e


def _transform_request_body(
    messages: List[AllMessageValues],
    model: str,
    optional_params: dict,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    litellm_params: dict,
    cached_content: Optional[str],
) -> RequestBody:
    """
    Common transformation logic across sync + async Gemini /generateContent calls.
    """
    # Separate system prompt from rest of message
    supports_system_message = get_supports_system_message(
        model=model, custom_llm_provider=custom_llm_provider
    )
    system_instructions, messages = _transform_system_message(
        supports_system_message=supports_system_message, messages=messages
    )
    # Checks for 'response_schema' support - if passed in
    if "response_schema" in optional_params:
        supports_response_schema = get_supports_response_schema(
            model=model, custom_llm_provider=custom_llm_provider
        )
        if supports_response_schema is False:
            user_response_schema_message = response_schema_prompt(
                model=model, response_schema=optional_params.get("response_schema")  # type: ignore
            )
            messages.append({"role": "user", "content": user_response_schema_message})
            optional_params.pop("response_schema")

    # Check for any 'litellm_param_*' set during optional param mapping

    remove_keys = []
    for k, v in optional_params.items():
        if k.startswith("litellm_param_"):
            litellm_params.update({k: v})
            remove_keys.append(k)

    optional_params = {k: v for k, v in optional_params.items() if k not in remove_keys}

    try:
        if custom_llm_provider == "gemini":
            content = litellm.GoogleAIStudioGeminiConfig()._transform_messages(
                messages=messages
            )
        else:
            content = litellm.VertexGeminiConfig()._transform_messages(
                messages=messages
            )
        tools: Optional[Tools] = optional_params.pop("tools", None)
        tool_choice: Optional[ToolConfig] = optional_params.pop("tool_choice", None)
        safety_settings: Optional[List[SafetSettingsConfig]] = optional_params.pop(
            "safety_settings", None
        )  # type: ignore
        config_fields = GenerationConfig.__annotations__.keys()

        filtered_params = {
            k: v for k, v in optional_params.items() if k in config_fields
        }

        generation_config: Optional[GenerationConfig] = GenerationConfig(
            **filtered_params
        )
        data = RequestBody(contents=content)
        if system_instructions is not None:
            data["system_instruction"] = system_instructions
        if tools is not None:
            data["tools"] = tools
        if tool_choice is not None:
            data["toolConfig"] = tool_choice
        if safety_settings is not None:
            data["safetySettings"] = safety_settings
        if generation_config is not None:
            data["generationConfig"] = generation_config
        if cached_content is not None:
            data["cachedContent"] = cached_content
    except Exception as e:
        raise e

    return data


def sync_transform_request_body(
    gemini_api_key: Optional[str],
    messages: List[AllMessageValues],
    api_base: Optional[str],
    model: str,
    client: Optional[HTTPHandler],
    timeout: Optional[Union[float, httpx.Timeout]],
    extra_headers: Optional[dict],
    optional_params: dict,
    logging_obj: LiteLLMLoggingObj,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    litellm_params: dict,
) -> RequestBody:
    from ..context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    context_caching_endpoints = ContextCachingEndpoints()

    if gemini_api_key is not None:
        messages, cached_content = context_caching_endpoints.check_and_create_cache(
            messages=messages,
            api_key=gemini_api_key,
            api_base=api_base,
            model=model,
            client=client,
            timeout=timeout,
            extra_headers=extra_headers,
            cached_content=optional_params.pop("cached_content", None),
            logging_obj=logging_obj,
        )
    else:  # [TODO] implement context caching for gemini as well
        cached_content = optional_params.pop("cached_content", None)

    return _transform_request_body(
        messages=messages,
        model=model,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        cached_content=cached_content,
        optional_params=optional_params,
    )


async def async_transform_request_body(
    gemini_api_key: Optional[str],
    messages: List[AllMessageValues],
    api_base: Optional[str],
    model: str,
    client: Optional[AsyncHTTPHandler],
    timeout: Optional[Union[float, httpx.Timeout]],
    extra_headers: Optional[dict],
    optional_params: dict,
    logging_obj: litellm.litellm_core_utils.litellm_logging.Logging,  # type: ignore
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    litellm_params: dict,
) -> RequestBody:
    from ..context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    context_caching_endpoints = ContextCachingEndpoints()

    if gemini_api_key is not None:
        (
            messages,
            cached_content,
        ) = await context_caching_endpoints.async_check_and_create_cache(
            messages=messages,
            api_key=gemini_api_key,
            api_base=api_base,
            model=model,
            client=client,
            timeout=timeout,
            extra_headers=extra_headers,
            cached_content=optional_params.pop("cached_content", None),
            logging_obj=logging_obj,
        )
    else:  # [TODO] implement context caching for gemini as well
        cached_content = optional_params.pop("cached_content", None)

    return _transform_request_body(
        messages=messages,
        model=model,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        cached_content=cached_content,
        optional_params=optional_params,
    )


def _transform_system_message(
    supports_system_message: bool, messages: List[AllMessageValues]
) -> Tuple[Optional[SystemInstructions], List[AllMessageValues]]:
    """
    Extracts the system message from the openai message list.

    Converts the system message to Gemini format

    Returns
    - system_content_blocks: Optional[SystemInstructions] - the system message list in Gemini format.
    - messages: List[AllMessageValues] - filtered list of messages in OpenAI format (transformed separately)
    """
    # Separate system prompt from rest of message
    system_prompt_indices = []
    system_content_blocks: List[PartType] = []
    if supports_system_message is True:
        for idx, message in enumerate(messages):
            if message["role"] == "system":
                _system_content_block: Optional[PartType] = None
                if isinstance(message["content"], str):
                    _system_content_block = PartType(text=message["content"])
                elif isinstance(message["content"], list):
                    system_text = ""
                    for content in message["content"]:
                        system_text += content.get("text") or ""
                    _system_content_block = PartType(text=system_text)
                if _system_content_block is not None:
                    system_content_blocks.append(_system_content_block)
                    system_prompt_indices.append(idx)
        if len(system_prompt_indices) > 0:
            for idx in reversed(system_prompt_indices):
                messages.pop(idx)

    if len(system_content_blocks) > 0:
        return SystemInstructions(parts=system_content_blocks), messages

    return None, messages
