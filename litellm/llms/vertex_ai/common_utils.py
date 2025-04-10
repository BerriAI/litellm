from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union, get_type_hints
import re

import httpx

import litellm
from litellm import supports_response_schema, supports_system_messages, verbose_logger
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.litellm_core_utils.prompt_templates.common_utils import unpack_defs
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.vertex_ai import PartType, Schema


class VertexAIError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[Dict, httpx.Headers]] = None,
    ):
        super().__init__(message=message, status_code=status_code, headers=headers)


def get_supports_system_message(
    model: str, custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"]
) -> bool:
    try:
        _custom_llm_provider = custom_llm_provider
        if custom_llm_provider == "vertex_ai_beta":
            _custom_llm_provider = "vertex_ai"
        supports_system_message = supports_system_messages(
            model=model, custom_llm_provider=_custom_llm_provider
        )

        # Vertex Models called in the `/gemini` request/response format also support system messages
        if litellm.VertexGeminiConfig._is_model_gemini_spec_model(model):
            supports_system_message = True
    except Exception as e:
        verbose_logger.warning(
            "Unable to identify if system message supported. Defaulting to 'False'. Received error message - {}\nAdd it here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json".format(
                str(e)
            )
        )
        supports_system_message = False

    return supports_system_message


def get_supports_response_schema(
    model: str, custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"]
) -> bool:
    _custom_llm_provider = custom_llm_provider
    if custom_llm_provider == "vertex_ai_beta":
        _custom_llm_provider = "vertex_ai"

    _supports_response_schema = supports_response_schema(
        model=model, custom_llm_provider=_custom_llm_provider
    )

    return _supports_response_schema


from typing import Literal, Optional

all_gemini_url_modes = Literal[
    "chat", "embedding", "batch_embedding", "image_generation"
]


def _get_vertex_url(
    mode: all_gemini_url_modes,
    model: str,
    stream: Optional[bool],
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_api_version: Literal["v1", "v1beta1"],
) -> Tuple[str, str]:
    url: Optional[str] = None
    endpoint: Optional[str] = None

    model = litellm.VertexGeminiConfig.get_model_for_vertex_ai_url(model=model)
    if mode == "chat":
        ### SET RUNTIME ENDPOINT ###
        endpoint = "generateContent"
        if stream is True:
            endpoint = "streamGenerateContent"
            url = f"https://{vertex_location}-aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:{endpoint}?alt=sse"
        else:
            url = f"https://{vertex_location}-aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:{endpoint}"

        # if model is only numeric chars then it's a fine tuned gemini model
        # model = 4965075652664360960
        # send to this url: url = f"https://{vertex_location}-aiplatform.googleapis.com/{version}/projects/{vertex_project}/locations/{vertex_location}/endpoints/{model}:{endpoint}"
        if model.isdigit():
            # It's a fine-tuned Gemini model
            url = f"https://{vertex_location}-aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/endpoints/{model}:{endpoint}"
            if stream is True:
                url += "?alt=sse"
    elif mode == "embedding":
        endpoint = "predict"
        url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:{endpoint}"
        if model.isdigit():
            # https://us-central1-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/us-central1/endpoints/$ENDPOINT_ID:predict
            url = f"https://{vertex_location}-aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/endpoints/{model}:{endpoint}"
    elif mode == "image_generation":
        endpoint = "predict"
        url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:{endpoint}"
        if model.isdigit():
            url = f"https://{vertex_location}-aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/endpoints/{model}:{endpoint}"
    if not url or not endpoint:
        raise ValueError(f"Unable to get vertex url/endpoint for mode: {mode}")
    return url, endpoint


def _get_gemini_url(
    mode: all_gemini_url_modes,
    model: str,
    stream: Optional[bool],
    gemini_api_key: Optional[str],
) -> Tuple[str, str]:
    _gemini_model_name = "models/{}".format(model)
    if mode == "chat":
        endpoint = "generateContent"
        if stream is True:
            endpoint = "streamGenerateContent"
            url = "https://generativelanguage.googleapis.com/v1beta/{}:{}?key={}&alt=sse".format(
                _gemini_model_name, endpoint, gemini_api_key
            )
        else:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/{}:{}?key={}".format(
                    _gemini_model_name, endpoint, gemini_api_key
                )
            )
    elif mode == "embedding":
        endpoint = "embedContent"
        url = "https://generativelanguage.googleapis.com/v1beta/{}:{}?key={}".format(
            _gemini_model_name, endpoint, gemini_api_key
        )
    elif mode == "batch_embedding":
        endpoint = "batchEmbedContents"
        url = "https://generativelanguage.googleapis.com/v1beta/{}:{}?key={}".format(
            _gemini_model_name, endpoint, gemini_api_key
        )
    elif mode == "image_generation":
        raise ValueError(
            "LiteLLM's `gemini/` route does not support image generation yet. Let us know if you need this feature by opening an issue at https://github.com/BerriAI/litellm/issues"
        )

    return url, endpoint


def _check_text_in_content(parts: List[PartType]) -> bool:
    """
    check that user_content has 'text' parameter.
        - Known Vertex Error: Unable to submit request because it must have a text parameter.
        - 'text' param needs to be len > 0
        - Relevant Issue: https://github.com/BerriAI/litellm/issues/5515
    """
    has_text_param = False
    for part in parts:
        if "text" in part and part.get("text"):
            has_text_param = True

    return has_text_param


def _build_vertex_schema(parameters: dict):
    """
    This is a modified version of https://github.com/google-gemini/generative-ai-python/blob/8f77cc6ac99937cd3a81299ecf79608b91b06bbb/google/generativeai/types/content_types.py#L419
    """
    # Get valid fields from Schema TypedDict
    valid_schema_fields = set(get_type_hints(Schema).keys())

    defs = parameters.pop("$defs", {})
    # flatten the defs
    for name, value in defs.items():
        unpack_defs(value, defs)
    unpack_defs(parameters, defs)

    # 5. Nullable fields:
    #     * https://github.com/pydantic/pydantic/issues/1270
    #     * https://stackoverflow.com/a/58841311
    #     * https://github.com/pydantic/pydantic/discussions/4872
    convert_anyof_null_to_nullable(parameters)
    add_object_type(parameters)
    # Postprocessing
    # Filter out fields that don't exist in Schema
    filtered_parameters = filter_schema_fields(parameters, valid_schema_fields)
    return filtered_parameters


def filter_schema_fields(
    schema_dict: Dict[str, Any], valid_fields: Set[str], processed=None
) -> Dict[str, Any]:
    """
    Recursively filter a schema dictionary to keep only valid fields.
    """
    if processed is None:
        processed = set()

    # Handle circular references
    schema_id = id(schema_dict)
    if schema_id in processed:
        return schema_dict
    processed.add(schema_id)

    if not isinstance(schema_dict, dict):
        return schema_dict

    result = {}
    for key, value in schema_dict.items():
        if key not in valid_fields:
            continue

        if key == "properties" and isinstance(value, dict):
            result[key] = {
                k: filter_schema_fields(v, valid_fields, processed)
                for k, v in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            result[key] = filter_schema_fields(value, valid_fields, processed)
        elif key == "anyOf" and isinstance(value, list):
            result[key] = [
                filter_schema_fields(item, valid_fields, processed) for item in value  # type: ignore
            ]
        else:
            result[key] = value

    return result


def convert_anyof_null_to_nullable(schema, depth=0):
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        raise ValueError(
            f"Max depth of {DEFAULT_MAX_RECURSE_DEPTH} exceeded while processing schema. Please check the schema for excessive nesting."
        )
    """ Converts null objects within anyOf by removing them and adding nullable to all remaining objects """
    anyof = schema.get("anyOf", None)
    if anyof is not None:
        contains_null = False
        for atype in anyof:
            if atype == {"type": "null"}:
                # remove null type
                anyof.remove(atype)
                contains_null = True

        if len(anyof) == 0:
            # Edge case: response schema with only null type present is invalid in Vertex AI
            raise ValueError(
                "Invalid input: AnyOf schema with only null type is not supported. "
                "Please provide a non-null type."
            )

        if contains_null:
            # set all types to nullable following guidance found here: https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-controlled-generation-response-schema-3#generativeaionvertexai_gemini_controlled_generation_response_schema_3-python
            for atype in anyof:
                atype["nullable"] = True

    properties = schema.get("properties", None)
    if properties is not None:
        for name, value in properties.items():
            convert_anyof_null_to_nullable(value, depth=depth + 1)

    items = schema.get("items", None)
    if items is not None:
        convert_anyof_null_to_nullable(items, depth=depth + 1)


def add_object_type(schema):
    properties = schema.get("properties", None)
    if properties is not None:
        if "required" in schema and schema["required"] is None:
            schema.pop("required", None)
        schema["type"] = "object"
        for name, value in properties.items():
            add_object_type(value)

    items = schema.get("items", None)
    if items is not None:
        add_object_type(items)


def strip_field(schema, field_name: str):
    schema.pop(field_name, None)

    properties = schema.get("properties", None)
    if properties is not None:
        for name, value in properties.items():
            strip_field(value, field_name)

    items = schema.get("items", None)
    if items is not None:
        strip_field(items, field_name)


def _convert_vertex_datetime_to_openai_datetime(vertex_datetime: str) -> int:
    """
    Converts a Vertex AI datetime string to an OpenAI datetime integer

    vertex_datetime: str = "2024-12-04T21:53:12.120184Z"
    returns: int = 1722729192
    """
    from datetime import datetime

    # Parse the ISO format string to datetime object
    dt = datetime.strptime(vertex_datetime, "%Y-%m-%dT%H:%M:%S.%fZ")
    # Convert to Unix timestamp (seconds since epoch)
    return int(dt.timestamp())


def get_vertex_project_id_from_url(url: str) -> Optional[str]:
    """
    Get the vertex project id from the url

    `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:streamGenerateContent`
    """
    match = re.search(r"/projects/([^/]+)", url)
    return match.group(1) if match else None


def get_vertex_location_from_url(url: str) -> Optional[str]:
    """
    Get the vertex location from the url

    `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:streamGenerateContent`
    """
    match = re.search(r"/locations/([^/]+)", url)
    return match.group(1) if match else None


def replace_project_and_location_in_route(
    requested_route: str, vertex_project: str, vertex_location: str
) -> str:
    """
    Replace project and location values in the route with the provided values
    """
    # Replace project and location values while keeping route structure
    modified_route = re.sub(
        r"/projects/[^/]+/locations/[^/]+/",
        f"/projects/{vertex_project}/locations/{vertex_location}/",
        requested_route,
    )
    return modified_route


def construct_target_url(
    base_url: str,
    requested_route: str,
    vertex_location: Optional[str],
    vertex_project: Optional[str],
) -> httpx.URL:
    """
    Allow user to specify their own project id / location.

    If missing, use defaults

    Handle cachedContent scenario - https://github.com/BerriAI/litellm/issues/5460

    Constructed Url:
    POST https://LOCATION-aiplatform.googleapis.com/{version}/projects/PROJECT_ID/locations/LOCATION/cachedContents
    """
    new_base_url = httpx.URL(base_url)
    if "locations" in requested_route:  # contains the target project id + location
        if vertex_project and vertex_location:
            requested_route = replace_project_and_location_in_route(
                requested_route, vertex_project, vertex_location
            )
        return new_base_url.copy_with(path=requested_route)

    """
    - Add endpoint version (e.g. v1beta for cachedContent, v1 for rest)
    - Add default project id
    - Add default location
    """
    vertex_version: Literal["v1", "v1beta1"] = "v1"
    if "cachedContent" in requested_route:
        vertex_version = "v1beta1"

    base_requested_route = "{}/projects/{}/locations/{}".format(
        vertex_version, vertex_project, vertex_location
    )

    updated_requested_route = "/" + base_requested_route + requested_route

    updated_url = new_base_url.copy_with(path=updated_requested_route)
    return updated_url
