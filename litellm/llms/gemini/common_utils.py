import base64
import datetime
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Union

import httpx

import litellm
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo, BaseTokenCounter
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import TokenCountResponse

GEMINI_IMAGE_ASPECT_RATIOS: Dict[str, float] = {
    "1:1": 1 / 1,
    "1:4": 1 / 4,
    "1:8": 1 / 8,
    "2:3": 2 / 3,
    "3:2": 3 / 2,
    "3:4": 3 / 4,
    "4:1": 4 / 1,
    "4:3": 4 / 3,
    "4:5": 4 / 5,
    "5:4": 5 / 4,
    "8:1": 8 / 1,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
}

# Supported aspect ratio dimensions from Google Gemini image generation docs:
# https://ai.google.dev/gemini-api/docs/image-generation#aspect_ratios_and_image_size
GEMINI_IMAGE_SIZE_TO_ASPECT_RATIO: Dict[tuple[int, int], str] = {
    (512, 512): "1:1",
    (1024, 1024): "1:1",
    (2048, 2048): "1:1",
    (4096, 4096): "1:1",
    (256, 1024): "1:4",
    (512, 2048): "1:4",
    (1024, 4096): "1:4",
    (2048, 8192): "1:4",
    (192, 1536): "1:8",
    (384, 3072): "1:8",
    (768, 6144): "1:8",
    (1536, 12288): "1:8",
    (424, 632): "2:3",
    (848, 1264): "2:3",
    (1696, 2528): "2:3",
    (3392, 5056): "2:3",
    (632, 424): "3:2",
    (1264, 848): "3:2",
    (2528, 1696): "3:2",
    (5056, 3392): "3:2",
    (448, 600): "3:4",
    (896, 1200): "3:4",
    (1792, 2400): "3:4",
    (3584, 4800): "3:4",
    (1024, 256): "4:1",
    (2048, 512): "4:1",
    (4096, 1024): "4:1",
    (8192, 2048): "4:1",
    (600, 448): "4:3",
    (1200, 896): "4:3",
    (2400, 1792): "4:3",
    (4800, 3584): "4:3",
    (464, 576): "4:5",
    (928, 1152): "4:5",
    (1856, 2304): "4:5",
    (3712, 4608): "4:5",
    (576, 464): "5:4",
    (1152, 928): "5:4",
    (2304, 1856): "5:4",
    (4608, 3712): "5:4",
    (1536, 192): "8:1",
    (3072, 384): "8:1",
    (6144, 768): "8:1",
    (12288, 1536): "8:1",
    (384, 688): "9:16",
    (768, 1376): "9:16",
    (1536, 2752): "9:16",
    (3072, 5504): "9:16",
    (688, 384): "16:9",
    (1376, 768): "16:9",
    (2752, 1536): "16:9",
    (5504, 3072): "16:9",
    (792, 336): "21:9",
    (1584, 672): "21:9",
    (3168, 1344): "21:9",
    (6336, 2688): "21:9",
    (1280, 896): "4:3",
    (896, 1280): "3:4",
}


def map_openai_size_to_gemini_image_config(
    size: str, model: str
) -> Optional[Dict[str, str]]:
    dimensions = _parse_openai_image_size(size)
    if dimensions is None:
        return None

    width, height = dimensions
    image_config = {
        "aspectRatio": _map_dimensions_to_gemini_aspect_ratio(width, height)
    }
    image_size = _map_dimensions_to_gemini_image_size(width, height)
    if is_gemini_image_model(model):
        if supports_gemini_image_size(model):
            image_config["imageSize"] = image_size
    else:
        image_config["imageSize"] = image_size
    return image_config


def supports_gemini_image_size(model: str) -> bool:
    try:
        model_info = litellm.get_model_info(model=model)
        value = model_info.get("supports_image_size")
        if value is not None:
            return bool(value)
    except Exception:
        pass
    return "2.5-flash" not in model


def is_gemini_image_model(model: str) -> bool:
    base_model = model.split("/", 1)[-1]
    return "gemini" in base_model


def map_openai_image_params_to_gemini(
    params: Dict[str, Any],
    model: str,
    supported_params: Sequence[str],
    optional_params: Optional[Dict[str, Any]] = None,
    parse_image_config_string: bool = False,
) -> Dict[str, Any]:
    optional_params = optional_params or {}
    filtered_params = {
        key: value for key, value in params.items() if key in supported_params
    }

    mapped_params: Dict[str, Any] = {}

    if "n" in filtered_params and "n" not in optional_params:
        mapped_params["sampleCount"] = filtered_params["n"]

    if "size" in filtered_params and "size" not in optional_params:
        image_config = map_openai_size_to_gemini_image_config(
            filtered_params["size"],
            model,
        )
        if image_config is not None:
            if is_gemini_image_model(model):
                mapped_params["imageConfig"] = image_config
            else:
                mapped_params["aspectRatio"] = image_config["aspectRatio"]
                if "imageSize" in image_config:
                    mapped_params["imageSize"] = image_config["imageSize"]

    image_config_param = filtered_params.get("imageConfig")
    if isinstance(image_config_param, str) and parse_image_config_string:
        try:
            image_config_param = json.loads(image_config_param)
        except json.JSONDecodeError as exc:
            raise litellm.UnsupportedParamsError(
                model=model,
                message="`imageConfig` must be valid JSON when provided as a string.",
            ) from exc
    if isinstance(image_config_param, dict):
        mapped_params["imageConfig"] = image_config_param

    for key, value in filtered_params.items():
        if (
            key not in ("n", "size", "imageConfig", "tools", "web_search_options")
            and key not in optional_params
        ):
            mapped_params[key] = value

    return mapped_params


def _dedupe_gemini_search_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    search_tool_keys = VertexGeminiConfig._search_tool_keys()
    seen_search_keys: set[str] = set()
    deduped_tools: List[Dict[str, Any]] = []

    for tool in tools:
        if not isinstance(tool, dict):
            deduped_tools.append(tool)
            continue

        search_key = next((key for key in search_tool_keys if key in tool), None)
        if search_key is None:
            deduped_tools.append(tool)
            continue

        if search_key in seen_search_keys:
            continue

        seen_search_keys.add(search_key)
        deduped_tools.append(tool)

    return deduped_tools


def _has_gemini_search_tool(tools: List[Any]) -> bool:
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    search_tool_keys = VertexGeminiConfig._search_tool_keys()
    return any(
        isinstance(tool, dict) and any(key in tool for key in search_tool_keys)
        for tool in tools
    )


def map_gemini_image_tools_params(
    non_default_params: Dict[str, Any],
    mapped_params: Dict[str, Any],
) -> Dict[str, Any]:
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    gemini_config = VertexGeminiConfig()
    result = dict(mapped_params)
    result.pop("web_search_options", None)

    tools_value = non_default_params.get("tools")
    if isinstance(tools_value, list) and tools_value:
        mapped_tools = gemini_config._map_function(
            value=tools_value, optional_params=result
        )
        result = gemini_config._add_tools_to_optional_params(result, mapped_tools)

    web_search_options = non_default_params.get("web_search_options")
    existing_tools = result.get("tools")
    if isinstance(web_search_options, dict) and not (
        isinstance(existing_tools, list) and _has_gemini_search_tool(existing_tools)
    ):
        search_tool = gemini_config._map_web_search_options(web_search_options)
        result = gemini_config._add_tools_to_optional_params(result, [search_tool])

    gemini_config._drop_search_tools_mixed_with_functions(result)

    if isinstance(result.get("tools"), list):
        result["tools"] = _dedupe_gemini_search_tools(result["tools"])

    return result


def get_gemini_image_web_search_requests(
    response_data: Dict[str, Any],
) -> Optional[int]:
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    grounding_metadata: List[Dict[str, Any]] = []
    for candidate in response_data.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        candidate_grounding = candidate.get("groundingMetadata")
        if isinstance(candidate_grounding, list):
            grounding_metadata.extend(candidate_grounding)
        elif isinstance(candidate_grounding, dict):
            grounding_metadata.append(candidate_grounding)

    return VertexGeminiConfig._calculate_web_search_requests(grounding_metadata)


def get_gemini_image_generation_config(
    model: str,
    optional_params: Dict[str, Any],
) -> Dict[str, Any]:
    generation_config: Dict[str, Any] = {"response_modalities": ["IMAGE", "TEXT"]}

    image_config: Dict[str, Any] = {}
    if isinstance(optional_params.get("imageConfig"), dict):
        image_config.update(optional_params["imageConfig"])

    if not supports_gemini_image_size(model):
        image_config.pop("imageSize", None)

    if image_config:
        generation_config["imageConfig"] = image_config

    candidate_count = next(
        (
            optional_params[key]
            for key in ("candidateCount", "candidate_count", "sampleCount", "n")
            if optional_params.get(key) is not None
        ),
        None,
    )
    if candidate_count is not None:
        generation_config["candidateCount"] = candidate_count

    return generation_config


def _parse_openai_image_size(size: str) -> Optional[tuple[int, int]]:
    if size == "auto":
        return None

    width_str, separator, height_str = size.lower().partition("x")
    if not separator:
        return None

    try:
        width = int(width_str)
        height = int(height_str)
    except ValueError:
        return None

    if width <= 0 or height <= 0:
        return None

    return width, height


def _map_dimensions_to_gemini_aspect_ratio(width: int, height: int) -> str:
    if (width, height) in GEMINI_IMAGE_SIZE_TO_ASPECT_RATIO:
        return GEMINI_IMAGE_SIZE_TO_ASPECT_RATIO[(width, height)]

    requested_ratio = width / height
    return min(
        GEMINI_IMAGE_ASPECT_RATIOS,
        key=lambda aspect_ratio: abs(
            math.log(GEMINI_IMAGE_ASPECT_RATIOS[aspect_ratio] / requested_ratio)
        ),
    )


def _map_dimensions_to_gemini_image_size(width: int, height: int) -> str:
    effective_square_side = math.sqrt(width * height)
    if effective_square_side < 768:
        return "512"
    if effective_square_side < 1536:
        return "1K"
    if effective_square_side < 3072:
        return "2K"
    return "4K"


class GeminiError(BaseLLMException):
    pass


class GeminiModelInfo(BaseLLMModelInfo):
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
        """Google AI Studio sends api key via x-goog-api-key header"""
        return headers

    @property
    def api_version(self) -> str:
        return "v1beta"

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return (
            api_base
            or get_secret_str("GEMINI_API_BASE")
            or "https://generativelanguage.googleapis.com"
        )

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return (
            api_key
            or (get_secret_str("GOOGLE_API_KEY"))
            or (get_secret_str("GEMINI_API_KEY"))
        )

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        return model.replace("gemini/", "")

    def process_model_name(self, models: List[Dict[str, str]]) -> List[str]:
        litellm_model_names = []
        for model in models:
            stripped_model_name = model["name"].replace("models/", "")
            litellm_model_name = "gemini/" + stripped_model_name
            litellm_model_names.append(litellm_model_name)
        return litellm_model_names

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        api_base = GeminiModelInfo.get_api_base(api_base)
        api_key = GeminiModelInfo.get_api_key(api_key)
        endpoint = f"/{self.api_version}/models"
        if api_base is None or api_key is None:
            raise ValueError(
                "GEMINI_API_BASE or GEMINI_API_KEY/GOOGLE_API_KEY is not set. Please set the environment variable, to query Gemini's `/models` endpoint."
            )

        response = litellm.module_level_client.get(
            url=f"{api_base}{endpoint}",
            headers={"x-goog-api-key": api_key},
        )

        if response.status_code != 200:
            raise ValueError(
                f"Failed to fetch models from Gemini. Status code: {response.status_code}, Response: {response.json()}"
            )

        models = response.json()["models"]

        litellm_model_names = self.process_model_name(models)
        return litellm_model_names

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return GeminiError(
            status_code=status_code, message=error_message, headers=headers
        )

    def get_token_counter(self) -> Optional[BaseTokenCounter]:
        """
        Factory method to create a token counter for this provider.

        Returns:
            Optional TokenCounterInterface implementation for this provider,
            or None if token counting is not supported.
        """
        return GoogleAIStudioTokenCounter()


def encode_unserializable_types(
    data: Dict[str, object], depth: int = 0
) -> Dict[str, object]:
    """Converts unserializable types in dict to json.dumps() compatible types.

    This function is called in models.py after calling convert_to_dict(). The
    convert_to_dict() can convert pydantic object to dict. However, the input to
    convert_to_dict() is dict mixed of pydantic object and nested dict(the output
    of converters). So they may be bytes in the dict and they are out of
    `ser_json_bytes` control in model_dump(mode='json') called in
    `convert_to_dict`, as well as datetime deserialization in Pydantic json mode.

    Returns:
      A dictionary with json.dumps() incompatible type (e.g. bytes datetime)
      to compatible type (e.g. base64 encoded string, isoformat date string).
    """
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return data
    processed_data: dict[str, object] = {}
    if not isinstance(data, dict):
        return data
    for key, value in data.items():
        if isinstance(value, bytes):
            processed_data[key] = base64.urlsafe_b64encode(value).decode("ascii")
        elif isinstance(value, datetime.datetime):
            processed_data[key] = value.isoformat()
        elif isinstance(value, dict):
            processed_data[key] = encode_unserializable_types(value, depth + 1)
        elif isinstance(value, list):
            if all(isinstance(v, bytes) for v in value):
                processed_data[key] = [
                    base64.urlsafe_b64encode(v).decode("ascii") for v in value
                ]
            if all(isinstance(v, datetime.datetime) for v in value):
                processed_data[key] = [v.isoformat() for v in value]
            else:
                processed_data[key] = [
                    encode_unserializable_types(v, depth + 1) for v in value
                ]
        else:
            processed_data[key] = value
    return processed_data


def get_api_key_from_env() -> Optional[str]:
    return get_secret_str("GOOGLE_API_KEY") or get_secret_str("GEMINI_API_KEY")


class GoogleAIStudioTokenCounter(BaseTokenCounter):
    """Token counter implementation for Google AI Studio provider."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        from litellm.types.utils import LlmProviders

        return custom_llm_provider == LlmProviders.GEMINI.value

    async def count_tokens(
        self,
        model_to_use: str,
        messages: Optional[List[Dict[str, Any]]],
        contents: Optional[List[Dict[str, Any]]],
        deployment: Optional[Dict[str, Any]] = None,
        request_model: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[Any] = None,
    ) -> Optional[TokenCountResponse]:
        import copy

        from litellm.llms.gemini.count_tokens.handler import GoogleAIStudioTokenCounter

        deployment = deployment or {}
        count_tokens_params_request = copy.deepcopy(
            deployment.get("litellm_params", {})
        )
        count_tokens_params = {
            "model": model_to_use,
            "contents": contents,
        }
        count_tokens_params_request.update(count_tokens_params)
        result = await GoogleAIStudioTokenCounter().acount_tokens(
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
