"""
Transformation logic from OpenAI format to Gemini format.

Why separate file? Make it easy to see how transformation works
"""
import json
import os
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union, cast

import httpx
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _get_image_mime_type_from_url,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_generic_image_chunk_to_openai_image_obj,
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
    ChatCompletionUserMessage,
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
from litellm.types.utils import GenericImageParsingChunk, LlmProviders

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


def _convert_detail_to_media_resolution_enum(
    detail: Optional[str],
) -> Optional[Dict[str, str]]:
    if detail == "low":
        return {"level": "MEDIA_RESOLUTION_LOW"}
    elif detail == "medium":
        return {"level": "MEDIA_RESOLUTION_MEDIUM"}
    elif detail == "high":
        return {"level": "MEDIA_RESOLUTION_HIGH"}
    elif detail == "ultra_high":
        return {"level": "MEDIA_RESOLUTION_ULTRA_HIGH"}
    return None


def _apply_gemini_3_metadata(
    part: PartType,
    model: Optional[str],
    media_resolution_enum: Optional[Dict[str, str]],
    video_metadata: Optional[Dict[str, Any]],
) -> PartType:
    """
    Apply the unique media_resolution and video_metadata parameters of Gemini 3+    
    """
    if model is None:
        return part

    from .vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    if not VertexGeminiConfig._is_gemini_3_or_newer(model):
        return part

    part_dict = dict(part)

    if media_resolution_enum is not None:
        part_dict["media_resolution"] = media_resolution_enum

    if video_metadata is not None:
        gemini_video_metadata = {}
        if "fps" in video_metadata:
            gemini_video_metadata["fps"] = video_metadata["fps"]
        if "start_offset" in video_metadata:
            gemini_video_metadata["startOffset"] = video_metadata["start_offset"]
        if "end_offset" in video_metadata:
            gemini_video_metadata["endOffset"] = video_metadata["end_offset"]
        if gemini_video_metadata:
            part_dict["video_metadata"] = gemini_video_metadata

    return cast(PartType, part_dict)


def _process_gemini_media(
    image_url: str,
    format: Optional[str] = None,
    media_resolution_enum: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
    video_metadata: Optional[Dict[str, Any]] = None,
) -> PartType:
    """
    Given a media URL (image, audio, or video), return the appropriate PartType for Gemini
    By the way, actually video_metadata can only be used with videos; it cannot be used with images, audio, or files. However, I haven't made any special handling because vertex returns a parameter error.

    Args:
        image_url: The URL or base64 string of the media (image, audio, or video)
        format: The MIME type of the media
        media_resolution_enum: Media resolution level (for Gemini 3+)
        model: The model name (to check version compatibility)
        video_metadata: Video-specific metadata (fps, start_offset, end_offset)
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
            part: PartType = {"file_data": file_data}
            return _apply_gemini_3_metadata(
                part, model, media_resolution_enum, video_metadata
            )
        elif (
            "https://" in image_url
            and (image_type := format or _get_image_mime_type_from_url(image_url))
            is not None
        ):
            file_data = FileDataType(mime_type=image_type, file_uri=image_url)
            part = {"file_data": file_data}
            return _apply_gemini_3_metadata(
                part, model, media_resolution_enum, video_metadata
            )
        elif "http://" in image_url or "https://" in image_url or "base64" in image_url:
            image = convert_to_anthropic_image_obj(image_url, format=format)
            _blob: BlobType = {"data": image["data"], "mime_type": image["media_type"]}
            part = {"inline_data": cast(BlobType, _blob)}
            return _apply_gemini_3_metadata(
                part, model, media_resolution_enum, video_metadata
            )
        raise Exception("Invalid image received - {}".format(image_url))
    except Exception as e:
        raise e


def _snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase"""
    components = snake_str.split("_")
    return components[0] + "".join(x.capitalize() for x in components[1:])


def _camel_to_snake(camel_str: str) -> str:
    """Convert camelCase to snake_case"""
    import re

    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str).lower()


def _get_equivalent_key(key: str, available_keys: set) -> Optional[str]:
    """
    Get the equivalent key from available keys, checking both camelCase and snake_case variants
    """
    if key in available_keys:
        return key

    # Try camelCase version
    camel_key = _snake_to_camel(key)
    if camel_key in available_keys:
        return camel_key

    # Try snake_case version
    snake_key = _camel_to_snake(key)
    if snake_key in available_keys:
        return snake_key

    return None


def check_if_part_exists_in_parts(
    parts: List[PartType], part: PartType, excluded_keys: List[str] = []
) -> bool:
    """
    Check if a part exists in a list of parts
    Handles both camelCase and snake_case key variations (e.g., function_call vs functionCall)
    """
    keys_to_compare = set(part.keys()) - set(excluded_keys)
    for p in parts:
        p_keys = set(p.keys())
        # Check if all keys in part have equivalent values in p
        match_found = True
        for key in keys_to_compare:
            equivalent_key = _get_equivalent_key(key, p_keys)
            if equivalent_key is None or p.get(equivalent_key, None) != part.get(
                key, None
            ):
                match_found = False
                break

        if match_found:
            return True
    return False


def _gemini_convert_messages_with_history(  # noqa: PLR0915
    messages: List[AllMessageValues],
    model: Optional[str] = None,
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
                            media_resolution_enum: Optional[Dict[str, str]] = None
                            if isinstance(img_element["image_url"], dict):
                                image_url = img_element["image_url"]["url"]
                                format = img_element["image_url"].get("format")
                                detail = img_element["image_url"].get("detail")
                                media_resolution_enum = _convert_detail_to_media_resolution_enum(detail)
                            else:
                                image_url = img_element["image_url"]
                            _part = _process_gemini_media(
                                image_url=image_url,
                                format=format,
                                media_resolution_enum=media_resolution_enum,
                                model=model,
                            )
                            _parts.append(_part)
                        elif element["type"] == "input_audio":
                            audio_element = cast(ChatCompletionAudioObject, element)
                            audio_data = audio_element["input_audio"].get("data")
                            audio_format = audio_element["input_audio"].get("format")
                            if audio_data is not None and audio_format is not None:
                                audio_format_modified = (
                                    "audio/" + audio_format
                                    if audio_format.startswith("audio/") is False
                                    else audio_format
                                )  # Gemini expects audio/wav, audio/mp3, etc.
                                openai_image_str = (
                                    convert_generic_image_chunk_to_openai_image_obj(
                                        image_chunk=GenericImageParsingChunk(
                                            type="base64",
                                            media_type=audio_format_modified,
                                            data=audio_data,
                                        )
                                    )
                                )
                                _part = _process_gemini_media(
                                    image_url=openai_image_str,
                                    format=audio_format_modified,
                                    model=model,
                                )
                                _parts.append(_part)
                        elif element["type"] == "file":
                            file_element = cast(ChatCompletionFileObject, element)
                            file_id = file_element["file"].get("file_id")
                            format = file_element["file"].get("format")
                            file_data = file_element["file"].get("file_data")
                            detail = file_element["file"].get("detail")
                            video_metadata = file_element["file"].get("video_metadata")
                            passed_file = file_id or file_data
                            if passed_file is None:
                                raise Exception(
                                    "Unknown file type. Please pass in a file_id or file_data"
                                )

                            # Convert detail to media_resolution_enum
                            media_resolution_enum = _convert_detail_to_media_resolution_enum(detail)

                            try:
                                _part = _process_gemini_media(
                                    image_url=passed_file,
                                    format=format,
                                    model=model,
                                    media_resolution_enum=media_resolution_enum,
                                    video_metadata=video_metadata,
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
                reasoning_content = assistant_msg.get("reasoning_content", None)
                thinking_blocks = assistant_msg.get("thinking_blocks")
                if reasoning_content is not None:
                    assistant_content.append(
                        PartType(thought=True, text=reasoning_content)
                    )
                if thinking_blocks is not None:
                    for block in thinking_blocks:
                        if block["type"] == "thinking":
                            block_thinking_str = block.get("thinking")
                            block_signature = block.get("signature")
                            if (
                                block_thinking_str is not None
                                and block_signature is not None
                            ):
                                try:
                                    assistant_content.append(
                                        PartType(
                                            thoughtSignature=block_signature,
                                            **json.loads(block_thinking_str),
                                        )
                                    )
                                except Exception:
                                    assistant_content.append(
                                        PartType(
                                            thoughtSignature=block_signature,
                                            text=block_thinking_str,
                                        )
                                    )
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
                ):
                    assistant_text = _message_content
                    # Check if message has thought_signatures in provider_specific_fields
                    provider_specific_fields = assistant_msg.get("provider_specific_fields")
                    thought_signatures = None
                    if provider_specific_fields and isinstance(provider_specific_fields, dict):
                        thought_signatures = provider_specific_fields.get("thought_signatures")
                    
                    # If we have thought signatures, add them to the part
                    if thought_signatures and isinstance(thought_signatures, list) and len(thought_signatures) > 0:
                        # Use the first signature for the text part (Gemini expects one signature per part)
                        assistant_content.append(PartType(text=assistant_text, thoughtSignature=thought_signatures[0]))  # type: ignore
                    else:
                        assistant_content.append(PartType(text=assistant_text))  # type: ignore

                ## HANDLE ASSISTANT IMAGES FIELD
                # Process images field if present (for generated images from assistant)
                assistant_images = assistant_msg.get("images")
                if assistant_images is not None and isinstance(assistant_images, list):
                    for image_item in assistant_images:
                        if isinstance(image_item, dict):
                            image_url_obj = image_item.get("image_url")
                            if isinstance(image_url_obj, dict):
                                assistant_image_url = image_url_obj.get("url")
                                format = image_url_obj.get("format")
                                detail = image_url_obj.get("detail")
                                media_resolution_enum = _convert_detail_to_media_resolution_enum(detail)
                                if assistant_image_url:
                                    _part = _process_gemini_media(
                                        image_url=assistant_image_url,
                                        format=format,
                                        media_resolution_enum=media_resolution_enum,
                                        model=model,
                                    )
                                    assistant_content.append(_part)

                ## HANDLE ASSISTANT FUNCTION CALL
                if (
                    assistant_msg.get("tool_calls", []) is not None
                    or assistant_msg.get("function_call") is not None
                ):  # support assistant tool invoke conversion
                    gemini_tool_call_parts = convert_to_gemini_tool_call_invoke(
                        assistant_msg, model=model
                    )
                    ## check if gemini_tool_call already exists in assistant_content
                    for gemini_tool_call_part in gemini_tool_call_parts:
                        if not check_if_part_exists_in_parts(
                            assistant_content,
                            gemini_tool_call_part,
                            excluded_keys=["thoughtSignature"],
                        ):
                            assistant_content.append(gemini_tool_call_part)
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
                # Handle both single part and list of parts (for Computer Use with images)
                if isinstance(_part, list):
                    tool_call_responses.extend(_part)
                else:
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

        if len(contents) == 0:
            verbose_logger.warning(
                """
                No contents in messages. Contents are required. See
                https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.publishers.models/generateContent#request-body.
                If the original request did not comply to OpenAI API requirements it should have failed by now,
                but LiteLLM does not check for missing messages.
                Setting an empty content to prevent an 400 error.
                Relevant Issue - https://github.com/BerriAI/litellm/issues/9733
                """
            )
            contents.append(ContentType(role="user", parts=[PartType(text=" ")]))
        return contents
    except Exception as e:
        raise e


def _pop_and_merge_extra_body(data: RequestBody, optional_params: dict) -> None:
    """Pop extra_body from optional_params and shallow-merge into data, deep-merging dict values."""
    extra_body: Optional[dict] = optional_params.pop("extra_body", None)
    if extra_body is not None:
        for k, v in extra_body.items():
            if k in data and isinstance(data[k], dict) and isinstance(v, dict):
                data[k].update(v)
            else:
                data[k] = v


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
                messages=messages, model=model
            )
        else:
            content = litellm.VertexGeminiConfig()._transform_messages(
                messages=messages, model=model
            )
        tools: Optional[Tools] = optional_params.pop("tools", None)
        tool_choice: Optional[ToolConfig] = optional_params.pop("tool_choice", None)
        safety_settings: Optional[List[SafetSettingsConfig]] = optional_params.pop(
            "safety_settings", None
        )  # type: ignore
        config_fields = GenerationConfig.__annotations__.keys()

        # If the LiteLLM client sends Gemini-supported parameter "labels", add it
        # as "labels" field to the request sent to the Gemini backend.
        labels: Optional[dict[str, str]] = optional_params.pop("labels", None)
        # If the LiteLLM client sends OpenAI-supported parameter "metadata", add it
        # as "labels" field to the request sent to the Gemini backend.
        if labels is None and "metadata" in litellm_params:
            metadata = litellm_params["metadata"]
            if metadata is not None and "requester_metadata" in metadata:
                rm = metadata["requester_metadata"]
                labels = {k: v for k, v in rm.items() if isinstance(v, str)}

        filtered_params = {
            k: v for k, v in optional_params.items() if _get_equivalent_key(k, set(config_fields))
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
        if generation_config is not None and len(generation_config) > 0:
            data["generationConfig"] = generation_config
        if cached_content is not None:
            data["cachedContent"] = cached_content
        # Only add labels for Vertex AI endpoints (not Google GenAI/AI Studio) and only if non-empty
        if labels and custom_llm_provider != LlmProviders.GEMINI:
            data["labels"] = labels
        _pop_and_merge_extra_body(data, optional_params)
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
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_auth_header: Optional[str],
) -> RequestBody:
    from ..context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    context_caching_endpoints = ContextCachingEndpoints()

    (
    messages,
    optional_params,
    cached_content,
    ) = context_caching_endpoints.check_and_create_cache(
        messages=messages,
        optional_params=optional_params,
        api_key=gemini_api_key or "dummy",
        api_base=api_base,
        model=model,
        client=client,
        timeout=timeout,
        extra_headers=extra_headers,
        cached_content=optional_params.pop("cached_content", None),
        logging_obj=logging_obj,
        custom_llm_provider=custom_llm_provider,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        vertex_auth_header=vertex_auth_header,
    )


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
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_auth_header: Optional[str],
) -> RequestBody:
    from ..context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    context_caching_endpoints = ContextCachingEndpoints()

    (
    messages,
    optional_params,
    cached_content,
    ) = await context_caching_endpoints.async_check_and_create_cache(
        messages=messages,
        optional_params=optional_params,
        api_key=gemini_api_key or "dummy",
        api_base=api_base,
        model=model,
        client=client,
        timeout=timeout,
        extra_headers=extra_headers,
        cached_content=optional_params.pop("cached_content", None),
        logging_obj=logging_obj,
        custom_llm_provider=custom_llm_provider,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        vertex_auth_header=vertex_auth_header,
    )

    return _transform_request_body(
        messages=messages,
        model=model,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        cached_content=cached_content,
        optional_params=optional_params,
    )


def _default_user_message_when_system_message_passed() -> ChatCompletionUserMessage:
    """
    Returns a default user message when a "system" message is passed in gemini fails.

    This adds a blank user message to the messages list, to ensure that gemini doesn't fail the request.
    """
    return ChatCompletionUserMessage(content=".", role="user")


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
        #########################################################
        # If no messages are passed in, add a blank user message
        # Relevant Issue - https://github.com/BerriAI/litellm/issues/13769
        #########################################################
        if len(messages) == 0:
            messages.append(_default_user_message_when_system_message_passed())
        #########################################################
        return SystemInstructions(parts=system_content_blocks), messages

    return None, messages
