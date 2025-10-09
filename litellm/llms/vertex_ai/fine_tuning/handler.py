import json
import traceback
from datetime import datetime
from typing import Any, Coroutine, Literal, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import HTTPHandler, get_async_httpx_client
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.fine_tuning import OpenAIFineTuningHyperparameters
from litellm.types.llms.openai import FineTuningJobCreate
from litellm.types.llms.vertex_ai import (
    VERTEX_CREDENTIALS_TYPES,
    FineTuneHyperparameters,
    FineTuneJobCreate,
    FineTunesupervisedTuningSpec,
    ResponseSupervisedTuningSpec,
    ResponseTuningJob,
)
from litellm.types.utils import LiteLLMFineTuningJob


class VertexFineTuningAPI(VertexLLM):
    """
    Vertex methods to support for batches
    """

    def __init__(self) -> None:
        super().__init__()
        self.async_handler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
            params={"timeout": 600.0},
        )

    def convert_response_created_at(self, response: ResponseTuningJob):
        try:
            create_time_str = response.get("createTime", "") or ""
            create_time_datetime = datetime.fromisoformat(
                create_time_str.replace("Z", "+00:00")
            )
            # Convert to Unix timestamp (seconds since epoch)
            created_at = int(create_time_datetime.timestamp())

            return created_at
        except Exception:
            return 0

    def convert_openai_request_to_vertex(
        self,
        create_fine_tuning_job_data: FineTuningJobCreate,
        original_hyperparameters: dict = {},
        kwargs: Optional[dict] = None,
    ) -> FineTuneJobCreate:
        """
        convert request from OpenAI format to Vertex format
        https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/tuning
        supervised_tuning_spec = FineTunesupervisedTuningSpec(
        """

        supervised_tuning_spec = FineTunesupervisedTuningSpec(
            training_dataset_uri=create_fine_tuning_job_data.training_file,
        )

        if create_fine_tuning_job_data.validation_file:
            supervised_tuning_spec["validation_dataset"] = (
                create_fine_tuning_job_data.validation_file
            )

        _vertex_hyperparameters = (
            self._transform_openai_hyperparameters_to_vertex_hyperparameters(
                create_fine_tuning_job_data=create_fine_tuning_job_data,
                kwargs=kwargs,
                original_hyperparameters=original_hyperparameters,
            )
        )

        if _vertex_hyperparameters and len(_vertex_hyperparameters) > 0:
            supervised_tuning_spec["hyperParameters"] = _vertex_hyperparameters

        fine_tune_job = FineTuneJobCreate(
            baseModel=create_fine_tuning_job_data.model,
            supervisedTuningSpec=supervised_tuning_spec,
            tunedModelDisplayName=create_fine_tuning_job_data.suffix,
        )

        return fine_tune_job

    def _transform_openai_hyperparameters_to_vertex_hyperparameters(
        self,
        create_fine_tuning_job_data: FineTuningJobCreate,
        original_hyperparameters: dict = {},
        kwargs: Optional[dict] = None,
    ) -> FineTuneHyperparameters:
        _oai_hyperparameters = create_fine_tuning_job_data.hyperparameters
        _vertex_hyperparameters = FineTuneHyperparameters()
        if _oai_hyperparameters:
            if _oai_hyperparameters.n_epochs:
                _vertex_hyperparameters["epoch_count"] = int(
                    _oai_hyperparameters.n_epochs
                )
            if _oai_hyperparameters.learning_rate_multiplier:
                _vertex_hyperparameters["learning_rate_multiplier"] = float(
                    _oai_hyperparameters.learning_rate_multiplier
                )

        _adapter_size = original_hyperparameters.get("adapter_size", None)
        if _adapter_size:
            _vertex_hyperparameters["adapter_size"] = _adapter_size

        return _vertex_hyperparameters

    def convert_vertex_response_to_open_ai_response(
        self, response: ResponseTuningJob
    ) -> LiteLLMFineTuningJob:
        status: Literal[
            "validating_files", "queued", "running", "succeeded", "failed", "cancelled"
        ] = "queued"
        if response["state"] == "JOB_STATE_PENDING":
            status = "queued"
        if response["state"] == "JOB_STATE_SUCCEEDED":
            status = "succeeded"
        if response["state"] == "JOB_STATE_FAILED":
            status = "failed"
        if response["state"] == "JOB_STATE_CANCELLED":
            status = "cancelled"
        if response["state"] == "JOB_STATE_RUNNING":
            status = "running"

        created_at = self.convert_response_created_at(response)

        _supervisedTuningSpec: ResponseSupervisedTuningSpec = (
            response.get("supervisedTuningSpec", None) or {}
        )
        training_uri: str = _supervisedTuningSpec.get("trainingDatasetUri", "") or ""
        return LiteLLMFineTuningJob(
            id=response.get("name", "") or "",
            created_at=created_at,
            fine_tuned_model=response.get("tunedModelDisplayName", ""),
            finished_at=None,
            hyperparameters=self._translate_vertex_response_hyperparameters(
                vertex_hyper_parameters=_supervisedTuningSpec.get(
                    "hyperParameters", FineTuneHyperparameters()
                )
                or {}
            ),
            model=response.get("baseModel", "") or "",
            object="fine_tuning.job",
            organization_id="",
            result_files=[],
            seed=0,
            status=status,
            trained_tokens=None,
            training_file=training_uri,
            validation_file=None,
            estimated_finish=None,
            integrations=[],
        )

    def _translate_vertex_response_hyperparameters(
        self, vertex_hyper_parameters: FineTuneHyperparameters
    ) -> OpenAIFineTuningHyperparameters:
        """
        translate vertex responsehyperparameters to openai hyperparameters
        """
        _dict_remaining_hyperparameters: dict = dict(vertex_hyper_parameters)
        return OpenAIFineTuningHyperparameters(
            n_epochs=_dict_remaining_hyperparameters.pop("epoch_count", 0),
            **_dict_remaining_hyperparameters,
        )

    async def acreate_fine_tuning_job(
        self,
        fine_tuning_url: str,
        headers: dict,
        request_data: FineTuneJobCreate,
    ):
        try:
            verbose_logger.debug(
                "about to create fine tuning job: %s, request_data: %s",
                fine_tuning_url,
                json.dumps(request_data, indent=4),
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
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        kwargs: Optional[dict] = None,
        original_hyperparameters: Optional[dict] = {},
    ) -> Union[LiteLLMFineTuningJob, Coroutine[Any, Any, LiteLLMFineTuningJob]]:
        verbose_logger.debug(
            "creating fine tuning job, args= %s", create_fine_tuning_job_data
        )
        _auth_header, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai_beta",
        )

        auth_header, _ = self._get_token_and_url(
            model="",
            auth_header=_auth_header,
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

        fine_tune_job = self.convert_openai_request_to_vertex(
            create_fine_tuning_job_data=create_fine_tuning_job_data,
            kwargs=kwargs,
            original_hyperparameters=original_hyperparameters or {},
        )

        fine_tuning_url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/tuningJobs"
        if _is_async is True:
            return self.acreate_fine_tuning_job(  # type: ignore
                fine_tuning_url=fine_tuning_url,
                headers=headers,
                request_data=fine_tune_job,
            )
        sync_handler = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))

        verbose_logger.debug(
            "about to create fine tuning job: %s, request_data: %s",
            fine_tuning_url,
            fine_tune_job,
        )
        response = sync_handler.post(
            headers=headers,
            url=fine_tuning_url,
            json=fine_tune_job,  # type: ignore
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
        open_ai_response = self.convert_vertex_response_to_open_ai_response(
            vertex_response
        )
        return open_ai_response

    async def pass_through_vertex_ai_POST_request(
        self,
        request_data: dict,
        vertex_project: str,
        vertex_location: str,
        vertex_credentials: str,
        request_route: str,
    ):
        _auth_header, vertex_project = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai_beta",
        )
        auth_header, _ = self._get_token_and_url(
            model="",
            auth_header=_auth_header,
            gemini_api_key=None,
            vertex_credentials=vertex_credentials,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            stream=False,
            custom_llm_provider="vertex_ai_beta",
            api_base="",
        )

        headers = {
            "Authorization": f"Bearer {auth_header}",
            "Content-Type": "application/json",
        }

        url = None
        if request_route == "/tuningJobs":
            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/tuningJobs"
        elif "/tuningJobs/" in request_route and "cancel" in request_route:
            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/tuningJobs{request_route}"
        elif "generateContent" in request_route:
            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}{request_route}"
        elif "predict" in request_route:
            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}{request_route}"
        elif "/batchPredictionJobs" in request_route:
            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}{request_route}"
        elif "countTokens" in request_route:
            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}{request_route}"
        elif "cachedContents" in request_route:
            _model = request_data.get("model")
            if _model is not None and "/publishers/google/models/" not in _model:
                request_data["model"] = (
                    f"projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{_model}"
                )

            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1beta1/projects/{vertex_project}/locations/{vertex_location}{request_route}"
        else:
            raise ValueError(f"Unsupported Vertex AI request route: {request_route}")
        if self.async_handler is None:
            raise ValueError("VertexAI Fine Tuning - async_handler is not initialized")

        response = await self.async_handler.post(
            headers=headers,
            url=url,
            json=request_data,  # type: ignore
        )

        if response.status_code != 200:
            raise Exception(
                f"Error creating fine tuning job. Status code: {response.status_code}. Response: {response.text}"
            )

        response_json = response.json()
        return response_json
