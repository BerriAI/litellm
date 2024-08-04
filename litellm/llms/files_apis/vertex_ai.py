import traceback
import uuid
from datetime import datetime
from typing import Any, Coroutine, Literal, Optional, Union

import httpx
from openai.types.fine_tuning.fine_tuning_job import FineTuningJob, Hyperparameters

from litellm._logging import verbose_logger
from litellm.llms.base import BaseLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_httpx import VertexLLM
from litellm.types.llms.openai import FineTuningJobCreate
from litellm.types.llms.vertex_ai import (
    FineTuneJobCreate,
    FineTunesupervisedTuningSpec,
    ResponseTuningJob,
)


class VertexFilesAPI(VertexLLM):
    """
    Vertex methods to support for batches
    """

    def __init__(self) -> None:
        super().__init__()
        self.async_handler = AsyncHTTPHandler(
            timeout=httpx.Timeout(timeout=600.0, connect=5.0)
        )
        self.BUCKET_NAME = "litellm"

    async def upload_to_gcs(
        self,
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[str],
    ):
        object_name = f"litellm_finetune_{uuid.uuid4().hex}.jsonl"
        auth_header, _ = self._get_token_and_url(
            model="",
            gemini_api_key=None,
            vertex_credentials=vertex_credentials,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            stream=False,
            custom_llm_provider="vertex_ai_beta",
            api_base=None,
        )

        headers = {
            "Authorization": f"Bearer {auth_header}",
            "Content-Type": "application/json",
        }
        response = await self.async_handler.post(  # type: ignore
            headers=headers,
            url=f"https://storage.googleapis.com/upload/storage/v1/b/{self.BUCKET_NAME}/o?uploadType=media&name={object_name}",
            json={},
        )

        if response.status_code != 200:
            verbose_logger.error("GCS Bucket logging error: %s", str(response.text))

        verbose_logger.debug("GCS Bucket response %s", response)
        verbose_logger.debug("GCS Bucket status code %s", response.status_code)
        verbose_logger.debug("GCS Bucket response.text %s", response.text)

    async def acreate_file(
        self,
        fine_tuning_url: str,
        headers: dict,
        request_data: FineTuneJobCreate,
    ):
        from litellm.fine_tuning.main import FineTuningJob

        try:
            verbose_logger.debug(
                "about to create fine tuning job: %s, request_data: %s",
                fine_tuning_url,
                request_data,
            )
            if self.async_handler is None:
                raise ValueError(
                    "VertexAI Fine Tuning - async_handler is not initialized"
                )
            response = await self.async_handler.post(
                headers=headers,
                url=fine_tuning_url,
                json=request_data,  # type: ignore
            )

            if response.status_code != 200:
                raise Exception(
                    f"Error creating fine tuning job. Status code: {response.status_code}. Response: {response.text}"
                )

            verbose_logger.debug(
                "got response from creating fine tuning job: %s", response.json()
            )

            vertex_response = ResponseTuningJob(  # type: ignore
                **response.json(),
            )

            verbose_logger.debug("vertex_response %s", vertex_response)
            # open_ai_response = self.convert_vertex_response_to_open_ai_response(
            #     vertex_response
            # )

        except Exception as e:
            verbose_logger.error("asyncerror creating fine tuning job %s", e)
            trace_back_str = traceback.format_exc()
            verbose_logger.error(trace_back_str)
            raise e

    def create_file(
        self,
        _is_async: bool,
        create_file_data: FineTuningJobCreate,
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        **kwargs,
    ):

        verbose_logger.debug("creating fine tuning job, args= %s", create_file_data)

        auth_header, _ = self._get_token_and_url(
            model="",
            gemini_api_key=None,
            vertex_credentials=vertex_credentials,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            stream=False,
            custom_llm_provider="vertex_ai_beta",
            api_base=api_base,
        )

        # if _is_async is True:
        #     return self.acreate_file(  # type: ignore
        #         fine_tuning_url=fine_tuning_url,
        #         headers=headers,
        #         request_data=fine_tune_job,
        #     )
        # sync_handler = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))

        # verbose_logger.debug(
        #     "about to create fine tuning job: %s, request_data: %s",
        #     fine_tuning_url,
        #     fine_tune_job,
        # )
        # response = sync_handler.post(
        #     headers=headers,
        #     url=fine_tuning_url,
        #     json=fine_tune_job,  # type: ignore
        # )

        # if response.status_code != 200:
        #     raise Exception(
        #         f"Error creating fine tuning job. Status code: {response.status_code}. Response: {response.text}"
        #     )

        # verbose_logger.debug(
        #     "got response from creating fine tuning job: %s", response.json()
        # )
        # vertex_response = ResponseTuningJob(  # type: ignore
        #     **response.json(),
        # )

        # verbose_logger.debug("vertex_response %s", vertex_response)
        # open_ai_response = self.convert_vertex_response_to_open_ai_response(
        #     vertex_response
        # )
        # return open_ai_response
