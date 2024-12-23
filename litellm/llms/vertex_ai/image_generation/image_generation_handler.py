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
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[str],
        model_response: ImageResponse,
        logging_obj: Any,
        model: Optional[
            str
        ] = "imagegeneration",  # vertex ai uses imagegeneration as the default model
        client: Optional[Any] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[int] = None,
        aimg_generation=False,
    ) -> ImageResponse:
        if aimg_generation is True:
            return self.aimage_generation(  # type: ignore
                prompt=prompt,
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

        url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:predict"

        auth_header, _ = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        optional_params = optional_params or {
            "sampleCount": 1
        }  # default optional params

        request_data = {
            "instances": [{"prompt": prompt}],
            "parameters": optional_params,
        }

        request_str = f"\n curl -X POST \\\n -H \"Authorization: Bearer {auth_header[:10] + 'XXXXXXXXXX'}\" \\\n -H \"Content-Type: application/json; charset=utf-8\" \\\n -d {request_data} \\\n \"{url}\""
        logging_obj.pre_call(
            input=prompt,
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
                "request_str": request_str,
            },
        )

        logging_obj.pre_call(
            input=prompt,
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
                "request_str": request_str,
            },
        )

        response = sync_handler.post(
            url=url,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {auth_header}",
            },
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
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[str],
        model_response: litellm.ImageResponse,
        logging_obj: Any,
        model: Optional[
            str
        ] = "imagegeneration",  # vertex ai uses imagegeneration as the default model
        client: Optional[AsyncHTTPHandler] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[int] = None,
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
        url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:predict"

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
        auth_header, _ = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        optional_params = optional_params or {
            "sampleCount": 1
        }  # default optional params

        request_data = {
            "instances": [{"prompt": prompt}],
            "parameters": optional_params,
        }

        request_str = f"\n curl -X POST \\\n -H \"Authorization: Bearer {auth_header[:10] + 'XXXXXXXXXX'}\" \\\n -H \"Content-Type: application/json; charset=utf-8\" \\\n -d {request_data} \\\n \"{url}\""
        logging_obj.pre_call(
            input=prompt,
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
                "request_str": request_str,
            },
        )

        response = await self.async_handler.post(
            url=url,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {auth_header}",
            },
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
