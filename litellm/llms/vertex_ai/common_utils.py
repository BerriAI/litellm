import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union, get_type_hints

import httpx

import litellm
from litellm import supports_response_schema, supports_system_messages, verbose_logger
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.litellm_core_utils.prompt_templates.common_utils import unpack_defs
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo, BaseTokenCounter
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.vertex_ai import PartType, Schema
from litellm.types.utils import TokenCountResponse


class VertexAIError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[Dict, httpx.Headers]] = None,
    ):
        super().__init__(message=message, status_code=status_code, headers=headers)


class VertexAIModelRoute(str, Enum):
    """Enum for Vertex AI model routing"""
    PARTNER_MODELS = "partner_models"
    GEMINI = "gemini"
    GEMMA = "gemma"
    MODEL_GARDEN = "model_garden"
    NON_GEMINI = "non_gemini"


def get_vertex_ai_model_route(model: str, litellm_params: Optional[dict] = None) -> VertexAIModelRoute:
    """
    Determine which handler to use for a Vertex AI model based on the model name.
    
    Args:
        model: The model name (e.g., "llama3-405b", "gemini-pro", "gemma/gemma-3-12b-it", "openai/gpt-oss-120b")
        litellm_params: Optional litellm parameters dict that may contain base_model for routing
        
    Returns:
        VertexAIModelRoute: The route enum indicating which handler should be used
        
    Examples:
        >>> get_vertex_ai_model_route("llama3-405b")
        VertexAIModelRoute.PARTNER_MODELS
        
        >>> get_vertex_ai_model_route("gemini-pro")
        VertexAIModelRoute.GEMINI
        
        >>> get_vertex_ai_model_route("gemma/gemma-3-12b-it")
        VertexAIModelRoute.GEMMA
        
        >>> get_vertex_ai_model_route("openai/gpt-oss-120b")
        VertexAIModelRoute.MODEL_GARDEN
    """
    from litellm.llms.vertex_ai.vertex_ai_partner_models.main import (
        VertexAIPartnerModels,
    )

    # Check base_model in litellm_params for gemini override
    if litellm_params and litellm_params.get("base_model") is not None:
        if "gemini" in litellm_params["base_model"]:
            return VertexAIModelRoute.GEMINI
    
    # Check for partner models (llama, mistral, claude, etc.)
    if VertexAIPartnerModels.is_vertex_partner_model(model=model):
        return VertexAIModelRoute.PARTNER_MODELS
    
    # Check for gemma models
    if "gemma/" in model:
        return VertexAIModelRoute.GEMMA
    
    # Check for model garden openai models
    if "openai" in model:
        return VertexAIModelRoute.MODEL_GARDEN
    
    # Check for gemini models
    if "gemini" in model:
        return VertexAIModelRoute.GEMINI
    
    # Default to non-gemini (legacy vertex models like chat-bison, text-bison, etc.)
    return VertexAIModelRoute.NON_GEMINI


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
    "chat", "embedding", "batch_embedding", "image_generation", "count_tokens"
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
            if vertex_location == "global":
                url = f"https://aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/global/publishers/google/models/{model}:{endpoint}?alt=sse"
            else:
                url = f"https://{vertex_location}-aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:{endpoint}?alt=sse"
        else:
            if vertex_location == "global":
                url = f"https://aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/global/publishers/google/models/{model}:{endpoint}"
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
    elif mode == "count_tokens":
        endpoint = "countTokens"
        if vertex_location == "global":
            url = f"https://aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/global/publishers/google/models/{model}:{endpoint}"
        else:
            url = f"https://{vertex_location}-aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:{endpoint}"
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
    elif mode == "count_tokens":
        endpoint = "countTokens"
        url = "https://generativelanguage.googleapis.com/v1beta/{}:{}?key={}".format(
            _gemini_model_name, endpoint, gemini_api_key
        )
    elif mode == "image_generation":
        raise ValueError(
            "LiteLLM's `gemini/` route does not support image generation yet. Let us know if you need this feature by opening an issue at https://github.com/BerriAI/litellm/issues"
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

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


def _fix_enum_empty_strings(schema, depth=0):
    """Fix empty strings in enum values by replacing them with None. Gemini doesn't accept empty strings in enums."""
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        raise ValueError(f"Max depth of {DEFAULT_MAX_RECURSE_DEPTH} exceeded while processing schema.")
    
    if "enum" in schema and isinstance(schema["enum"], list):
        schema["enum"] = [None if value == "" else value for value in schema["enum"]]

    # Reuse existing recursion pattern from convert_anyof_null_to_nullable
    properties = schema.get("properties", None)
    if properties is not None:
        for _, value in properties.items():
            _fix_enum_empty_strings(value, depth=depth + 1)

    items = schema.get("items", None)
    if items is not None:
        _fix_enum_empty_strings(items, depth=depth + 1)


def _build_vertex_schema(parameters: dict, add_property_ordering: bool = False):
    """
    This is a modified version of https://github.com/google-gemini/generative-ai-python/blob/8f77cc6ac99937cd3a81299ecf79608b91b06bbb/google/generativeai/types/content_types.py#L419

    Updates the input parameters, removing extraneous fields, adjusting types, unwinding $defs, and adding propertyOrdering if specified, returning the updated parameters.

    Parameters:
        parameters: dict - the json schema to build from
        add_property_ordering: bool - whether to add propertyOrdering to the schema. This is only applicable to schemas for structured outputs. See
          set_schema_property_ordering for more details.
    Returns:
        parameters: dict - the input parameters, modified in place
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

    _convert_schema_types(parameters)

    # Handle empty strings in enum values - Gemini doesn't accept empty strings in enums
    _fix_enum_empty_strings(parameters)

    # Handle empty items objects
    process_items(parameters)
    add_object_type(parameters)
    # Postprocessing
    # Filter out fields that don't exist in Schema

    parameters = filter_schema_fields(parameters, valid_schema_fields)

    if add_property_ordering:
        set_schema_property_ordering(parameters)

    return parameters


def _filter_anyof_fields(schema_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    When anyof is present, only keep the anyof field and its contents - otherwise VertexAI will throw an error - https://github.com/BerriAI/litellm/issues/11164
    Filter out other fields in the same dict.

    E.g. {"anyOf": [{"type": "string"}, {"type": "null"}], "default": "test"} -> {"anyOf": [{"type": "string"}, {"type": "null"}]}

    Case 2: If additional metadata is present, try to keep it
    E.g. {"anyOf": [{"type": "string"}, {"type": "null"}], "default": "test", "title": "test"} -> {"anyOf": [{"type": "string", "title": "test"}, {"type": "null", "title": "test"}]}
    """
    title = schema_dict.get("title", None)
    description = schema_dict.get("description", None)

    if isinstance(schema_dict, dict) and schema_dict.get("anyOf"):
        any_of = schema_dict["anyOf"]
        if (
            (title or description)
            and isinstance(any_of, list)
            and all(isinstance(item, dict) for item in any_of)
        ):
            for item in any_of:
                if title:
                    item["title"] = title
                if description:
                    item["description"] = description
        return {"anyOf": any_of}
    return schema_dict


def process_items(schema, depth=0):
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        raise ValueError(
            f"Max depth of {DEFAULT_MAX_RECURSE_DEPTH} exceeded while processing schema. Please check the schema for excessive nesting."
        )
    if isinstance(schema, dict):
        if "items" in schema and schema["items"] == {}:
            schema["items"] = {"type": "object"}
        for key, value in schema.items():
            if isinstance(value, dict):
                process_items(value, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        process_items(item, depth + 1)


def set_schema_property_ordering(
    schema: Dict[str, Any], depth: int = 0
) -> Dict[str, Any]:
    """
    vertex ai and generativeai apis order output of fields alphabetically, unless you specify the order.
    python dicts retain order, so we just use that. Note that this field only applies to structured outputs, and not tools.
    Function tools are not afflicted by the same alphabetical ordering issue, (the order of keys returned seems to be arbitrary, up to the model)
    https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.cachedContents#Schema.FIELDS.property_ordering

    Args:
        schema: The schema dictionary to process
        depth: Current recursion depth to prevent infinite loops
    """
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        raise ValueError(
            f"Max depth of {DEFAULT_MAX_RECURSE_DEPTH} exceeded while processing schema. Please check the schema for excessive nesting."
        )

    if "properties" in schema and isinstance(schema["properties"], dict):
        # retain propertyOrdering as an escape hatch if user already specifies it
        if "propertyOrdering" not in schema:
            schema["propertyOrdering"] = [k for k, v in schema["properties"].items()]
        for k, v in schema["properties"].items():
            set_schema_property_ordering(v, depth + 1)
    if "items" in schema:
        set_schema_property_ordering(schema["items"], depth + 1)
    return schema


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
    schema_dict = _filter_anyof_fields(schema_dict)
    for key, value in schema_dict.items():
        if key not in valid_fields:
            continue

        if key == "properties" and isinstance(value, dict):
            result[key] = {
                k: filter_schema_fields(v, valid_fields, processed)
                for k, v in value.items()
            }
        elif key == "format":
            if value in {"enum", "date-time"}:
                result[key] = value
            else:
                continue
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
            elif "type" not in atype and len(atype) == 0:
                # Handle empty object case
                atype["type"] = "object"

        if len(anyof) == 0:
            # Edge case: response schema with only null type present is invalid in Vertex AI
            raise ValueError(
                "Invalid input: AnyOf schema with only null type is not supported. "
                "Please provide a non-null type."
            )

        if contains_null:
            # set all types to nullable following guidance found here: https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-controlled-generation-response-schema-3#generativeaionvertexai_gemini_controlled_generation_response_schema_3-python
            for atype in anyof:
                # Remove items field if type is array and items is empty
                if (
                    atype.get("type") == "array"
                    and "items" in atype
                    and not atype["items"]
                ):
                    atype.pop("items")
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


def _convert_schema_types(schema, depth=0):
    """
    Convert type arrays and lowercase types for Vertex AI compatibility.
    
    Transforms OpenAI-style schemas to Vertex AI format by converting type arrays 
    like ["string", "number"] to anyOf format and converting all types to uppercase.
    """
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        raise ValueError(
            f"Max depth of {DEFAULT_MAX_RECURSE_DEPTH} exceeded while processing schema. Please check the schema for excessive nesting."
        )
    
    if not isinstance(schema, dict):
        return

    
    # Handle type field
    if "type" in schema:
        type_val = schema["type"]
        if isinstance(type_val, list) and len(type_val) > 1:
            # Convert ["string", "number"] -> {"anyOf": [{"type": "STRING"}, {"type": "NUMBER"}]}
            schema["anyOf"] = [{"type": t} for t in type_val if isinstance(t, str)]
            schema.pop("type")
        elif isinstance(type_val, list) and len(type_val) == 1:
            schema["type"] = type_val[0]
        elif isinstance(type_val, str):
            schema["type"] = type_val
    
    # Recursively process nested properties, items, and anyOf
    for key in ["properties", "items", "anyOf"]:
        if key in schema:
            value = schema[key]
            if key == "properties" and isinstance(value, dict):
                for prop_schema in value.values():
                    _convert_schema_types(prop_schema, depth + 1)
            elif key == "items":
                _convert_schema_types(value, depth + 1)
            elif key == "anyOf" and isinstance(value, list):
                for anyof_schema in value:
                    _convert_schema_types(anyof_schema, depth + 1)

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


def is_global_only_vertex_model(model: str) -> bool:
    """
    Check if a model is only available in the global region.

    Args:
        model: The model name to check

    Returns:
        True if the model is only available in global region, False otherwise
    """
    from litellm.utils import get_supported_regions

    supported_regions = get_supported_regions(
        model=model, custom_llm_provider="vertex_ai"
    )
    if supported_regions is None:
        return False
    return "global" in supported_regions

class VertexAIModelInfo(BaseLLMModelInfo):    
    def get_token_counter(self) -> Optional[BaseTokenCounter]:
        """
        Factory method to create a token counter for this provider.
        
        Returns:
            Optional TokenCounterInterface implementation for this provider,
            or None if token counting is not supported.
        """
        return VertexAITokenCounter()
    
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        raise NotImplementedError("Vertex AI models are not supported yet")
    
    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        """
        Returns a list of models supported by this provider.
        """
        raise NotImplementedError("Vertex AI models are not supported yet")

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        raise NotImplementedError("Vertex AI models are not supported yet")

    @staticmethod
    def get_api_base(
        api_base: Optional[str] = None,
    ) -> Optional[str]:
        raise NotImplementedError("Vertex AI models are not supported yet")



    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        """
        Returns the base model name from the given model name.

        Some providers like bedrock - can receive model=`invoke/anthropic.claude-3-opus-20240229-v1:0` or `converse/anthropic.claude-3-opus-20240229-v1:0`
            This function will return `anthropic.claude-3-opus-20240229-v1:0`
        """
        raise NotImplementedError("Vertex AI models are not supported yet")


class VertexAITokenCounter(BaseTokenCounter):
    """Token counter implementation for Google AI Studio provider."""
    def should_use_token_counting_api(
        self, 
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        from litellm.types.utils import LlmProviders
        return custom_llm_provider == LlmProviders.VERTEX_AI.value
    
    async def count_tokens(
        self,
        model_to_use: str,
        messages: Optional[List[Dict[str, Any]]],
        contents: Optional[List[Dict[str, Any]]],
        deployment: Optional[Dict[str, Any]] = None,
        request_model: str = "",
    ) -> Optional[TokenCountResponse]:
        import copy

        from litellm.llms.vertex_ai.count_tokens.handler import VertexAITokenCounter
        deployment = deployment or {}
        count_tokens_params_request = copy.deepcopy(deployment.get("litellm_params", {}))
        count_tokens_params = {
            "model": model_to_use,
            "contents": contents,
        }
        count_tokens_params_request.update(count_tokens_params)
        result = await VertexAITokenCounter().acount_tokens(
            **count_tokens_params_request,
        )
        
        if result is not None:
            return TokenCountResponse(
                total_tokens=result.get("totalTokens", 0),
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type=result.get("tokenizer_used", ""),
                original_response=result,
            )
        
        return None