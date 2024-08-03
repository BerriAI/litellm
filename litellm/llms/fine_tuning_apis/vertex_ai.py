import traceback
from datetime import datetime
from typing import Any, Coroutine, Literal, Optional, Union

import httpx
from openai.types.fine_tuning.fine_tuning_job import FineTuningJob, Hyperparameters

from litellm._logging import verbose_logger
from litellm.llms.base import BaseLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.vertex_httpx import VertexLLM
from litellm.types.llms.openai import FineTuningJobCreate
from litellm.types.llms.vertex_ai import (
    FineTuneJobCreate,
    FineTunesupervisedTuningSpec,
    ResponseTuningJob,
)


class VertexFineTuningAPI(VertexLLM):
    """
    Vertex methods to support for batches
    """

    def __init__(self) -> None:
        super().__init__()
        self.async_handler = AsyncHTTPHandler(
            timeout=httpx.Timeout(timeout=600.0, connect=5.0)
        )

    def convert_response_created_at(self, response: ResponseTuningJob):
        try:
            create_time = datetime.fromisoformat(
                response["createTime"].replace("Z", "+00:00")
            )
            # Convert to Unix timestamp (seconds since epoch)
            created_at = int(create_time.timestamp())

            return created_at
        except Exception as e:
            return 0

    def convert_vertex_response_to_open_ai_response(
        self, response: ResponseTuningJob
    ) -> FineTuningJob:
        status: Literal[
            "validating_files", "queued", "running", "succeeded", "failed", "cancelled"
        ] = "queued"
        if response["state"] == "JOB_STATE_PENDING":
            status = "validating_files"
        if response["state"] == "JOB_STATE_SUCCEEDED":
            status = "succeeded"
        if response["state"] == "JOB_STATE_FAILED":
            status = "failed"
        if response["state"] == "JOB_STATE_CANCELLED":
            status = "cancelled"
        if response["state"] == "JOB_STATE_RUNNING":
            status = "running"

        created_at = self.convert_response_created_at(response)

        return FineTuningJob(
            id=response["name"],
            created_at=created_at,
            fine_tuned_model=response["tunedModelDisplayName"],
            finished_at=None,
            hyperparameters=Hyperparameters(
                n_epochs=0, batch_size="", learning_rate_multiplier=""
            ),
            model=response["baseModel"],
            object="fine_tuning.job",
            organization_id="",
            result_files=[],
            seed=0,
            status=status,
            trained_tokens=None,
            training_file=response["supervisedTuningSpec"]["trainingDatasetUri"],
            validation_file=None,
            estimated_finish=None,
            integrations=[],
            user_provided_suffix=None,
        )

    async def acreate_fine_tuning_job(
        self,
        fine_tuning_url: str,
        headers: dict,
        request_data: dict,
    ):
        from litellm.fine_tuning.main import FineTuningJob

        try:
            verbose_logger.debug(
                "about to create fine tuning job: %s, request_data: %s",
                fine_tuning_url,
                request_data,
            )
            response = await self.async_handler.post(
                headers=headers,
                url=fine_tuning_url,
                json=request_data,
            )

            if response.status_code != 200:
                raise Exception(
                    f"Error creating fine tuning job. Status code: {response.status_code}. Response: {response.text}"
                )

            verbose_logger.debug(
                "got response from creating fine tuning job: %s", response.json()
            )

            vertex_response = ResponseTuningJob(**response.json())

            verbose_logger.debug("vertex_response %s", vertex_response)
            open_ai_response = self.convert_vertex_response_to_open_ai_response(
                vertex_response
            )
            return open_ai_response

        except Exception as e:
            verbose_logger.error("asyncerror creating fine tuning job %s", e)
            trace_back_str = traceback.format_exc()
            verbose_logger.error(trace_back_str)
            raise e

    def create_fine_tuning_job(
        self,
        _is_async: bool,
        create_fine_tuning_job_data: FineTuningJobCreate,
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
    ):

        verbose_logger.debug(
            "creating fine tuning job, args= %s", create_fine_tuning_job_data
        )

        auth_header, _ = self._get_token_and_url(
            model=None,
            gemini_api_key=None,
            vertex_credentials=vertex_credentials,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            stream=False,
            custom_llm_provider="vertex_ai_beta",
            api_base=api_base,
        )

        headers = {
            "Authorization": f"Bearer {auth_header}",
            "Content-Type": "application/json",
        }

        supervised_tuning_spec = FineTunesupervisedTuningSpec(
            training_dataset_uri=create_fine_tuning_job_data.training_file
        )
        fine_tune_job = FineTuneJobCreate(
            baseModel=create_fine_tuning_job_data.model,
            supervisedTuningSpec=supervised_tuning_spec,
        )

        fine_tuning_url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/tuningJobs"
        if _is_async is True:
            return self.acreate_fine_tuning_job(  # type: ignore
                fine_tuning_url=fine_tuning_url,
                headers=headers,
                request_data=fine_tune_job,
            )
        # response = self.async_handler.post(
        #     url=fine_tuning_url,
        #     headers=headers,
        #     json=fine_tune_job,
        # )
