import json
from typing import Any, Final

import httpx
from openai.types.image import Image

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.utils import ImageResponse


class VertexImageGeneration(VertexLLM):
    def process_image_generation_response(
        self,
        json_response: dict[str, Any],
        model_response: ImageResponse,
        model: str | None = None,
    ) -> ImageResponse:
        if "predictions" not in json_response:
            raise litellm.InternalServerError(
                message=f"image generation response does not contain 'predictions', got {json_response}",
                llm_provider="vertex_ai",
                model=model,
            )

        predictions: Final = json_response["predictions"]
        response_data: Final[list[Image]] = []

        for prediction in predictions:
            bytes_base64_encoded = prediction["bytesBase64Encoded"]
            image_object = Image(b64_json=bytes_base64_encoded)
            response_data.append(image_object)

        model_response.data = response_data
        return model_response

    def transform_optional_params(self, optional_params: dict | None) -> dict:
        """
        Transform the optional params to the format expected by the Vertex AI API.
        For example, "aspect_ratio" is transformed to "aspectRatio".
        """
        default_params: Final = {
            "sampleCount": 1,
        }
        if optional_params is None:
            return default_params

        def snake_to_camel(snake_str: str) -> str:
            """Convert snake_case to camelCase"""
            components: Final = snake_str.split("_")
            return components[0] + "".join(word.capitalize() for word in components[1:])

        transformed_params: Final = default_params.copy()
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
        api_base: str | None,
        vertex_project: str | None,
        vertex_location: str | None,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
        model_response: ImageResponse,
        logging_obj: Any,
        model: str = "imagegeneration",  # vertex ai uses imagegeneration as the default model
        client: Any | None = None,
        optional_params: dict | None = None,
        timeout: int | None = None,
        aimg_generation=False,
        extra_headers: dict | None = None,
    ) -> ImageResponse:
        if aimg_generation is True:
            return self.aimage_generation(
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
            _params: Final = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout: Final = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            sync_handler: HTTPHandler = HTTPHandler(**_params)
        else:
            sync_handler = client

        # url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:predict"

        auth_header: str | None = None
        auth_header, _ = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
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
            mode="image_generation",
        )
        optional_params = optional_params or {"sampleCount": 1}  # default optional params

        # Transform optional params to camelCase format
        optional_params = self.transform_optional_params(optional_params)

        request_data: Final = {
            "instances": [{"prompt": prompt}],
            "parameters": optional_params,
        }

        headers: Final = self.set_headers(auth_header=auth_header, extra_headers=extra_headers)

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": optional_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        response: Final = sync_handler.post(
            url=api_base,
            headers=headers,
            data=json.dumps(request_data),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        json_response: Final = response.json()
        return self.process_image_generation_response(json_response, model_response, model)

    async def aimage_generation(
        self,
        prompt: str,
        api_base: str | None,
        vertex_project: str | None,
        vertex_location: str | None,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
        model_response: ImageResponse,
        logging_obj: Any,
        model: str = "imagegeneration",  # vertex ai uses imagegeneration as the default model
        client: AsyncHTTPHandler | None = None,
        optional_params: dict | None = None,
        timeout: int | None = None,
        extra_headers: dict | None = None,
    ):
        response = None
        if client is None:
            _params: Final = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout: Final = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            self.async_handler = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.VERTEX_AI,
                params={"timeout": timeout},
            )
        else:
            self.async_handler = client

        # make POST request to
        # https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/us-central1/publishers/google/models/imagegeneration:predict

        """
        Docs link: https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/imagegeneration?project=adroit-crow-413218
        curl -X POST \
        -H "Authorization: Bearer $(gcloud auth print-access-token)" \
        -H "Content-Type: application/json; charset=utf-8" \
        -d {
            "instances": [
                {
                    "prompt": "a cat"
                }
            ],
            "parameters": {
                "sampleCount": 1
            }
        } \
        "https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/us-central1/publishers/google/models/imagegeneration:predict"
        """
        auth_header: str | None = None
        auth_header, _ = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
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
            mode="image_generation",
        )

        # Transform optional params to camelCase format
        optional_params = self.transform_optional_params(optional_params)

        request_data: Final = {
            "instances": [{"prompt": prompt}],
            "parameters": optional_params,
        }

        headers: Final = self.set_headers(auth_header=auth_header, extra_headers=extra_headers)

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": optional_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        response = await self.async_handler.post(
            url=api_base,
            headers=headers,
            data=json.dumps(request_data),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        json_response: Final = response.json()
        return self.process_image_generation_response(json_response, model_response, model)

    def is_image_generation_response(self, json_response: dict[str, Any]) -> bool:
        if "predictions" in json_response:
            if "bytesBase64Encoded" in json_response["predictions"][0]:
                return True
        return False
