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
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.utils import ImageResponse


class VertexImageGeneration(VertexLLM):
    def process_image_generation_response(
        self,
        json_response: Dict[str, Any],
        model_response: ImageResponse,
        model: Optional[str] = None,
    ) -> ImageResponse:
        if "predictions" not in json_response:
            raise litellm.InternalServerError(
                message=f"image generation response does not contain 'predictions', got {json_response}",
                llm_provider="vertex_ai",
                model=model,
            )

        predictions = json_response["predictions"]
        response_data: List[Image] = []

        for prediction in predictions:
            bytes_base64_encoded = prediction["bytesBase64Encoded"]
            image_object = Image(b64_json=bytes_base64_encoded)
            response_data.append(image_object)

        model_response.data = response_data
        return model_response

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

        # url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:predict"

        auth_header: Optional[str] = None
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
        optional_params = optional_params or {
            "sampleCount": 1
        }  # default optional params

        request_data = {
            "instances": [{"prompt": prompt}],
            "parameters": optional_params,
        }

        headers = self.set_headers(auth_header=auth_header, extra_headers=extra_headers)

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": optional_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

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
        auth_header: Optional[str] = None
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
        optional_params = optional_params or {
            "sampleCount": 1
        }  # default optional params

        request_data = {
            "instances": [{"prompt": prompt}],
            "parameters": optional_params,
        }

        headers = self.set_headers(auth_header=auth_header, extra_headers=extra_headers)

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

        json_response = response.json()
        return self.process_image_generation_response(
            json_response, model_response, model
        )

    def is_image_generation_response(self, json_response: Dict[str, Any]) -> bool:
        if "predictions" in json_response:
            if "bytesBase64Encoded" in json_response["predictions"][0]:
                return True
        return False
