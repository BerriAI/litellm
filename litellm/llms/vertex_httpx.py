import os, types
import json
from enum import Enum
import requests  # type: ignore
import time
from typing import Callable, Optional, Union, List, Any, Tuple
from litellm.utils import ModelResponse, Usage, CustomStreamWrapper, map_finish_reason
import litellm, uuid
import httpx, inspect  # type: ignore
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from .base import BaseLLM


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


class VertexLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__()
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self._credentials: Optional[Any] = None
        self.project_id: Optional[str] = None
        self.async_handler: Optional[AsyncHTTPHandler] = None

    def load_auth(self) -> Tuple[Any, str]:
        from google.auth.transport.requests import Request  # type: ignore[import-untyped]
        from google.auth.credentials import Credentials  # type: ignore[import-untyped]
        import google.auth as google_auth

        credentials, project_id = google_auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        credentials.refresh(Request())

        if not project_id:
            raise ValueError("Could not resolve project_id")

        if not isinstance(project_id, str):
            raise TypeError(
                f"Expected project_id to be a str but got {type(project_id)}"
            )

        return credentials, project_id

    def refresh_auth(self, credentials: Any) -> None:
        from google.auth.transport.requests import Request  # type: ignore[import-untyped]

        credentials.refresh(Request())

    def _prepare_request(self, request: httpx.Request) -> None:
        access_token = self._ensure_access_token()

        if request.headers.get("Authorization"):
            # already authenticated, nothing for us to do
            return

        request.headers["Authorization"] = f"Bearer {access_token}"

    def _ensure_access_token(self) -> str:
        if self.access_token is not None:
            return self.access_token

        if not self._credentials:
            self._credentials, project_id = self.load_auth()
            if not self.project_id:
                self.project_id = project_id
        else:
            self.refresh_auth(self._credentials)

        if not self._credentials.token:
            raise RuntimeError("Could not resolve API token from the environment")

        assert isinstance(self._credentials.token, str)
        return self._credentials.token

    def image_generation(
        self,
        prompt: str,
        vertex_project: str,
        vertex_location: str,
        model: Optional[
            str
        ] = "imagegeneration",  # vertex ai uses imagegeneration as the default model
        client: Optional[AsyncHTTPHandler] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[int] = None,
        logging_obj=None,
        model_response=None,
        aimg_generation=False,
    ):
        if aimg_generation == True:
            response = self.aimage_generation(
                prompt=prompt,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                model=model,
                client=client,
                optional_params=optional_params,
                timeout=timeout,
                logging_obj=logging_obj,
                model_response=model_response,
            )
            return response

    async def aimage_generation(
        self,
        prompt: str,
        vertex_project: str,
        vertex_location: str,
        model_response: litellm.ImageResponse,
        model: Optional[
            str
        ] = "imagegeneration",  # vertex ai uses imagegeneration as the default model
        client: Optional[AsyncHTTPHandler] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[int] = None,
        logging_obj=None,
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

            self.async_handler = AsyncHTTPHandler(**_params)  # type: ignore
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
        auth_header = self._ensure_access_token()
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
        """
        Vertex AI Image generation response example:
        {
            "predictions": [
                {
                "bytesBase64Encoded": "BASE64_IMG_BYTES",
                "mimeType": "image/png"
                },
                {
                "mimeType": "image/png",
                "bytesBase64Encoded": "BASE64_IMG_BYTES"
                }
            ]
        }
        """

        _json_response = response.json()
        _predictions = _json_response["predictions"]

        _response_data: List[litellm.ImageObject] = []
        for _prediction in _predictions:
            _bytes_base64_encoded = _prediction["bytesBase64Encoded"]
            image_object = litellm.ImageObject(b64_json=_bytes_base64_encoded)
            _response_data.append(image_object)

        model_response.data = _response_data

        return model_response
