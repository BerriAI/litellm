import json
from typing import Any, Dict, List, Optional

import httpx
from openai.types.image import Image

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.llms.vertex_ai.common_utils import all_gemini_url_modes
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.utils import ImageResponse


class VertexImageGeneration(VertexLLM):
    @staticmethod
    def _is_gemini_image_preview_model(model: Optional[str]) -> bool:
        if model is None:
            return False

        normalized_model = model.split("/")[-1]
        return "2.5-flash-image-preview" in normalized_model

    def process_image_generation_response(
        self,
        json_response: Dict[str, Any],
        model_response: ImageResponse,
        model: Optional[str] = None,
    ) -> ImageResponse:
        # Gemini generateContent returns `candidates`, Imagen returns `predictions`
        if "candidates" in json_response:
            response_data: List[Image] = []
            candidates = json_response.get("candidates", [])
            for candidate in candidates:
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    inline_data = part.get("inlineData")
                    if inline_data and inline_data.get("data"):
                        response_data.append(
                            Image(
                                b64_json=inline_data["data"],
                            )
                        )

            if not response_data:
                raise litellm.InternalServerError(
                    message=(
                        "Unable to parse Gemini image response. "
                        f"Received payload: {json_response}"
                    ),
                    llm_provider="vertex_ai",
                    model=model,
                )

            model_response.data = response_data
            return model_response

        if "predictions" in json_response:
            predictions = json_response["predictions"]
            response_data: List[Image] = []
            for prediction in predictions:
                bytes_base64_encoded = prediction.get("bytesBase64Encoded")
                if bytes_base64_encoded is not None:
                    response_data.append(Image(b64_json=bytes_base64_encoded))

            if not response_data:
                raise litellm.InternalServerError(
                    message=(
                        "Unable to parse Imagen response: missing bytesBase64Encoded. "
                        f"Received payload: {json_response}"
                    ),
                    llm_provider="vertex_ai",
                    model=model,
                )

            model_response.data = response_data
            return model_response

        raise litellm.InternalServerError(
            message=(
                "Unexpected image generation response. "
                f"Received payload: {json_response}"
            ),
            llm_provider="vertex_ai",
            model=model,
        )

    def transform_optional_params(self, optional_params: Optional[dict]) -> dict:
        """
        Transform the optional params to the format expected by the Vertex AI API.
        For example, "aspect_ratio" is transformed to "aspectRatio".
        """
        if optional_params is None:
            return {
                "sampleCount": 1,
            }

        def snake_to_camel(snake_str: str) -> str:
            """Convert snake_case to camelCase"""
            components = snake_str.split("_")
            return components[0] + "".join(word.capitalize() for word in components[1:])

        transformed_params = {}
        for key, value in optional_params.items():
            if "_" in key:
                camel_case_key = snake_to_camel(key)
                transformed_params[camel_case_key] = value
            else:
                transformed_params[key] = value

        return transformed_params

    def image_generation(
        self,
        prompt: str,
        api_base: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        model_response: ImageResponse,
        logging_obj: Any,
        model: str = "imagegeneration",  # vertex ai uses imagegeneration as the default model
        client: Optional[Any] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[int] = None,
        aimg_generation=False,
        extra_headers: Optional[dict] = None,
    ) -> ImageResponse:
        if aimg_generation is True:
            return self.aimage_generation(  # type: ignore
                prompt=prompt,
                api_base=api_base,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                vertex_credentials=vertex_credentials,
                model=model,
                client=client,
                optional_params=optional_params,
                timeout=timeout,
                logging_obj=logging_obj,
                model_response=model_response,
            )

        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            sync_handler: HTTPHandler = HTTPHandler(**_params)  # type: ignore
        else:
            sync_handler = client  # type: ignore

        auth_header: Optional[str] = None
        auth_header, _ = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        is_gemini_image_preview = self._is_gemini_image_preview_model(model)
        vertex_mode: all_gemini_url_modes = (
            "chat" if is_gemini_image_preview else "image_generation"
        )

        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=None,
            auth_header=auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=False,
            custom_llm_provider="vertex_ai",
            api_base=api_base,
            should_use_v1beta1_features=False,
            mode=vertex_mode,
        )
        optional_params = (optional_params or {}).copy()

        if is_gemini_image_preview:
            request_data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
            }
        else:
            optional_params = optional_params or {"sampleCount": 1}
            optional_params = self.transform_optional_params(optional_params)

            request_data = {
                "instances": [{"prompt": prompt}],
                "parameters": optional_params,
            }

        headers = self.set_headers(auth_header=auth_header, extra_headers=extra_headers)

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": request_data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        if is_gemini_image_preview:
            response = sync_handler.post(
                url=api_base,
                headers=headers,
                json=request_data,
            )
        else:
            response = sync_handler.post(
                url=api_base,
                headers=headers,
                data=json.dumps(request_data),
            )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        json_response = response.json()
        return self.process_image_generation_response(
            json_response, model_response, model
        )

    async def aimage_generation(
        self,
        prompt: str,
        api_base: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        model_response: litellm.ImageResponse,
        logging_obj: Any,
        model: str = "imagegeneration",  # vertex ai uses imagegeneration as the default model
        client: Optional[AsyncHTTPHandler] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[int] = None,
        extra_headers: Optional[dict] = None,
    ):
        response = None
        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            self.async_handler = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.VERTEX_AI,
                params={"timeout": timeout},
            )
        else:
            self.async_handler = client  # type: ignore

        auth_header: Optional[str] = None
        auth_header, _ = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        is_gemini_image_preview = self._is_gemini_image_preview_model(model)
        vertex_mode: all_gemini_url_modes = (
            "chat" if is_gemini_image_preview else "image_generation"
        )

        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=None,
            auth_header=auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=False,
            custom_llm_provider="vertex_ai",
            api_base=api_base,
            should_use_v1beta1_features=False,
            mode=vertex_mode,
        )

        optional_params = (optional_params or {}).copy()

        if is_gemini_image_preview:
            request_data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
            }
        else:
            optional_params = optional_params or {"sampleCount": 1}
            optional_params = self.transform_optional_params(optional_params)

            request_data = {
                "instances": [{"prompt": prompt}],
                "parameters": optional_params,
            }

        headers = self.set_headers(auth_header=auth_header, extra_headers=extra_headers)

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": request_data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        if is_gemini_image_preview:
            response = await self.async_handler.post(
                url=api_base,
                headers=headers,
                json=request_data,
            )
        else:
            response = await self.async_handler.post(
                url=api_base,
                headers=headers,
                data=json.dumps(request_data),
            )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        json_response = response.json()
        return self.process_image_generation_response(
            json_response, model_response, model
        )

    def is_image_generation_response(self, json_response: Dict[str, Any]) -> bool:
        if "predictions" in json_response:
            if "bytesBase64Encoded" in json_response["predictions"][0]:
                return True
        return False
