import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

HUNYUAN_SUBMIT_ENDPOINT = "v1/aiart/openai/image/submit"
HUNYUAN_QUERY_ENDPOINT = "v1/aiart/openai/image/query"
HUNYUAN_POLLING_TIMEOUT = 600


class HunyuanImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for Tencent Hunyuan image generation via OpenAI-compatible API.

    Submit: POST https://api.cloudai.tencent.com/v1/aiart/openai/image/submit
    Query:  POST https://api.cloudai.tencent.com/v1/aiart/openai/image/query
    """

    DEFAULT_BASE_URL: str = "https://api.cloudai.tencent.com"

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base = (
            api_base
            or get_secret_str("HUNYUAN_API_BASE")
            or self.DEFAULT_BASE_URL
        )
        base = base.rstrip("/")
        return f"{base}/{HUNYUAN_SUBMIT_ENDPOINT}"

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
        final_api_key: Optional[str] = (
            api_key
            or get_secret_str("HUNYUAN_API_KEY")
        )
        if not final_api_key:
            raise ValueError("HUNYUAN_API_KEY is not set")

        headers["Authorization"] = final_api_key
        headers["Content-Type"] = "application/json"
        return headers

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return [
            "n",
            "quality",
            "response_format",
            "size",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        for k in non_default_params.keys():
            if k not in optional_params:
                if k in supported_params:
                    optional_params[k] = non_default_params[k]
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. "
                        f"Supported parameters are {supported_params}. "
                        "Set drop_params=True to drop unsupported parameters."
                    )
        return optional_params

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        request_body: Dict[str, Any] = {
            "prompt": prompt,
            "model": model or "gpt-image-2",
        }
        for k, v in optional_params.items():
            if k not in ("response_format",):
                request_body[k] = v
        return request_body

    @staticmethod
    def _build_query_url(api_base: Optional[str]) -> str:
        base = (
            api_base
            or get_secret_str("HUNYUAN_API_BASE")
            or "https://api.cloudai.tencent.com"
        )
        base = base.rstrip("/")
        return f"{base}/{HUNYUAN_QUERY_ENDPOINT}"

    @staticmethod
    def _check_timeout(start_time: float, timeout_secs: float) -> None:
        if time.time() - start_time > timeout_secs:
            raise TimeoutError(
                f"Hunyuan task polling timed out after {timeout_secs} seconds"
            )

    @staticmethod
    def _check_task_status(response_data: Dict[str, Any]) -> str:
        status = response_data.get("status", "").upper()
        verbose_logger.debug(f"Hunyuan task status: {status}")
        if status == "DONE":
            return "done"
        elif status == "FAIL":
            raise ValueError(
                f"Hunyuan image generation failed: {response_data.get('message', 'Unknown error')}"
            )
        elif status in ("WAIT", "RUN"):
            return "running"
        else:
            raise ValueError(f"Unknown Hunyuan task status: {status}")

    @staticmethod
    def _transform_response_to_openai(
        response_data: Dict[str, Any],
        model_response: ImageResponse,
    ) -> ImageResponse:
        if not model_response.data:
            model_response.data = []

        for image_item in response_data.get("data", []):
            if isinstance(image_item, dict):
                model_response.data.append(
                    ImageObject(
                        url=image_item.get("url"),
                        b64_json=image_item.get("b64_json"),
                    )
                )
        return model_response

    def _poll_task_sync(
        self,
        job_id: str,
        query_url: str,
        headers: Dict[str, str],
        timeout_secs: float = HUNYUAN_POLLING_TIMEOUT,
    ) -> Dict[str, Any]:
        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        client = _get_httpx_client()
        start_time = time.time()

        while True:
            self._check_timeout(start_time=start_time, timeout_secs=timeout_secs)

            response = client.post(
                url=query_url,
                json={"job_id": job_id},
                headers=headers,
            )
            response.raise_for_status()
            response_data = response.json()

            status = self._check_task_status(response_data)
            if status == "done":
                return response_data
            time.sleep(3)

    async def _poll_task_async(
        self,
        job_id: str,
        query_url: str,
        headers: Dict[str, str],
        timeout_secs: float = HUNYUAN_POLLING_TIMEOUT,
    ) -> Dict[str, Any]:
        import litellm
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        client = get_async_httpx_client(llm_provider=litellm.LlmProviders.HUNYUAN)
        start_time = time.time()

        while True:
            self._check_timeout(start_time=start_time, timeout_secs=timeout_secs)

            response = await client.post(
                url=query_url,
                json={"job_id": job_id},
                headers=headers,
            )
            response.raise_for_status()
            response_data = response.json()

            status = self._check_task_status(response_data)
            if status == "done":
                return response_data
            await asyncio.sleep(3)

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        try:
            submit_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error parsing Hunyuan submit response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        job_id = submit_data.get("job_id")
        if not job_id:
            raise ValueError(
                f"Hunyuan submit response missing job_id: {submit_data}"
            )

        verbose_logger.debug(f"Hunyuan polling job_id={job_id}...")

        auth_header = raw_response.request.headers.get("Authorization", "")
        poll_headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
        }

        api_base = litellm_params.get("api_base") or self.DEFAULT_BASE_URL
        query_url = self._build_query_url(api_base)

        result = self._poll_task_sync(
            job_id=job_id,
            query_url=query_url,
            headers=poll_headers,
        )

        verbose_logger.debug("Hunyuan polling complete, transforming response")
        return self._transform_response_to_openai(
            response_data=result,
            model_response=model_response,
        )

    async def async_transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        try:
            submit_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error parsing Hunyuan submit response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        job_id = submit_data.get("job_id")
        if not job_id:
            raise ValueError(
                f"Hunyuan submit response missing job_id: {submit_data}"
            )

        verbose_logger.debug(f"Hunyuan polling job_id={job_id} (async)...")

        auth_header = raw_response.request.headers.get("Authorization", "")
        poll_headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
        }

        api_base = litellm_params.get("api_base") or self.DEFAULT_BASE_URL
        query_url = self._build_query_url(api_base)

        result = await self._poll_task_async(
            job_id=job_id,
            query_url=query_url,
            headers=poll_headers,
        )

        verbose_logger.debug("Hunyuan polling complete (async), transforming response")
        return self._transform_response_to_openai(
            response_data=result,
            model_response=model_response,
        )
