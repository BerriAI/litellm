"""
Handler for Vertex AI Supervised Fine-Tuning
"""

import json
import traceback
from typing import Any, Coroutine, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import HTTPHandler, get_async_httpx_client
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES

from .transformation import VertexAIFineTuningTransformation
from .types import (
    FineTuningJobCreate,
    FineTuningJobStatus,
    FineTuningJobList,
    DatasetValidationResult,
    FineTuningCostEstimate,
)


class VertexAIFineTuningHandler(VertexLLM):
    """
    Handler for Vertex AI Supervised Fine-Tuning
    """

    def __init__(self) -> None:
        super().__init__()
        self.async_handler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
            params={"timeout": 300.0},  # Longer timeout for fine-tuning operations
        )

    def create_fine_tuning_job(
        self,
        job_create: FineTuningJobCreate,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
        acompletion: bool = False,
    ) -> Union[FineTuningJobStatus, Coroutine[Any, Any, FineTuningJobStatus]]:
        """
        Create a supervised fine-tuning job
        """
        try:
            # Validate model supports fine-tuning
            if not VertexAIFineTuningTransformation.validate_model_supports_fine_tuning(job_create.model):
                raise ValueError(f"Model {job_create.model} does not support fine-tuning")

            # Validate dataset format
            training_validation = VertexAIFineTuningTransformation.validate_dataset_format(
                job_create.training_file
            )
            if not training_validation.is_valid:
                raise ValueError(f"Training dataset validation failed: {training_validation.errors}")

            if job_create.validation_file:
                validation_validation = VertexAIFineTuningTransformation.validate_dataset_format(
                    job_create.validation_file
                )
                if not validation_validation.is_valid:
                    raise ValueError(f"Validation dataset validation failed: {validation_validation.errors}")

            # Get authentication
            access_token, project_id = self._ensure_access_token(
                credentials=vertex_credentials,
                project_id=job_create.vertex_project,
                custom_llm_provider="vertex_ai",
            )

            # Create fine-tuning request
            request_data = VertexAIFineTuningTransformation.create_fine_tuning_request(job_create)

            # Create fine-tuning URL
            fine_tuning_url = VertexAIFineTuningTransformation.create_fine_tuning_url(
                project_id=project_id,
                location=job_create.vertex_location
            )

            # Set up headers
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {access_token}",
            }

            verbose_logger.debug(
                "Creating fine-tuning job: %s, request_data: %s",
                fine_tuning_url,
                json.dumps(request_data, indent=2),
            )

            if acompletion:
                return self._async_create_fine_tuning_job(
                    fine_tuning_url=fine_tuning_url,
                    headers=headers,
                    request_data=request_data,
                )

            # Make synchronous request
            sync_handler = HTTPHandler()
            response = sync_handler.post(
                url=fine_tuning_url,
                headers=headers,
                json=request_data,
            )

            if response.status_code != 200:
                error_message = self._extract_error_from_response(response.text)
                raise Exception(
                    f"Failed to create fine-tuning job. Status: {response.status_code}. Error: {error_message}"
                )

            # Parse response
            response_data = response.json()
            job_id = VertexAIFineTuningTransformation.extract_job_id_from_response(response_data)
            
            # Transform to job status
            job_status = VertexAIFineTuningTransformation.transform_vertex_response_to_job_status(
                response_data, job_id
            )

            verbose_logger.debug(
                "Created fine-tuning job: %s", job_status.dict()
            )

            return job_status

        except Exception as e:
            verbose_logger.error(
                "Error creating fine-tuning job: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
            raise e

    async def _async_create_fine_tuning_job(
        self,
        fine_tuning_url: str,
        headers: Dict[str, str],
        request_data: Dict[str, Any],
    ) -> FineTuningJobStatus:
        """
        Async version of create fine-tuning job
        """
        try:
            if self.async_handler is None:
                raise ValueError("Async handler is not initialized")

            response = await self.async_handler.post(
                url=fine_tuning_url,
                headers=headers,
                json=request_data,
            )

            if response.status_code != 200:
                error_message = self._extract_error_from_response(response.text)
                raise Exception(
                    f"Failed to create fine-tuning job. Status: {response.status_code}. Error: {error_message}"
                )

            # Parse response
            response_data = response.json()
            job_id = VertexAIFineTuningTransformation.extract_job_id_from_response(response_data)
            
            # Transform to job status
            job_status = VertexAIFineTuningTransformation.transform_vertex_response_to_job_status(
                response_data, job_id
            )

            return job_status

        except Exception as e:
            verbose_logger.error(
                "Async error creating fine-tuning job: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
            raise e

    def get_fine_tuning_job(
        self,
        job_id: str,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
        acompletion: bool = False,
    ) -> Union[FineTuningJobStatus, Coroutine[Any, Any, FineTuningJobStatus]]:
        """
        Get the status of a fine-tuning job
        """
        try:
            # Get authentication
            access_token, project_id = self._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_project,
                custom_llm_provider="vertex_ai",
            )

            # Create job status URL
            job_status_url = VertexAIFineTuningTransformation.create_job_status_url(
                project_id=project_id,
                location=vertex_location or "us-central1",
                job_id=job_id
            )

            # Set up headers
            headers = {
                "Authorization": f"Bearer {access_token}",
            }

            if acompletion:
                return self._async_get_fine_tuning_job(
                    job_status_url=job_status_url,
                    headers=headers,
                    job_id=job_id,
                )

            # Make synchronous request
            sync_handler = HTTPHandler()
            response = sync_handler.get(
                url=job_status_url,
                headers=headers,
            )

            if response.status_code != 200:
                error_message = self._extract_error_from_response(response.text)
                raise Exception(
                    f"Failed to get fine-tuning job. Status: {response.status_code}. Error: {error_message}"
                )

            # Parse response
            response_data = response.json()
            
            # Transform to job status
            job_status = VertexAIFineTuningTransformation.transform_vertex_response_to_job_status(
                response_data, job_id
            )

            return job_status

        except Exception as e:
            verbose_logger.error(
                "Error getting fine-tuning job: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
            raise e

    async def _async_get_fine_tuning_job(
        self,
        job_status_url: str,
        headers: Dict[str, str],
        job_id: str,
    ) -> FineTuningJobStatus:
        """
        Async version of get fine-tuning job
        """
        try:
            if self.async_handler is None:
                raise ValueError("Async handler is not initialized")

            response = await self.async_handler.get(
                url=job_status_url,
                headers=headers,
            )

            if response.status_code != 200:
                error_message = self._extract_error_from_response(response.text)
                raise Exception(
                    f"Failed to get fine-tuning job. Status: {response.status_code}. Error: {error_message}"
                )

            # Parse response
            response_data = response.json()
            
            # Transform to job status
            job_status = VertexAIFineTuningTransformation.transform_vertex_response_to_job_status(
                response_data, job_id
            )

            return job_status

        except Exception as e:
            verbose_logger.error(
                "Async error getting fine-tuning job: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
            raise e

    def list_fine_tuning_jobs(
        self,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
        page_size: int = 20,
        page_token: Optional[str] = None,
        acompletion: bool = False,
    ) -> Union[FineTuningJobList, Coroutine[Any, Any, FineTuningJobList]]:
        """
        List fine-tuning jobs
        """
        try:
            # Get authentication
            access_token, project_id = self._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_project,
                custom_llm_provider="vertex_ai",
            )

            # Create list URL
            list_url = VertexAIFineTuningTransformation.create_fine_tuning_url(
                project_id=project_id,
                location=vertex_location or "us-central1"
            )

            # Add query parameters
            params = {"pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token

            # Set up headers
            headers = {
                "Authorization": f"Bearer {access_token}",
            }

            if acompletion:
                return self._async_list_fine_tuning_jobs(
                    list_url=list_url,
                    headers=headers,
                    params=params,
                )

            # Make synchronous request
            sync_handler = HTTPHandler()
            response = sync_handler.get(
                url=list_url,
                headers=headers,
                params=params,
            )

            if response.status_code != 200:
                error_message = self._extract_error_from_response(response.text)
                raise Exception(
                    f"Failed to list fine-tuning jobs. Status: {response.status_code}. Error: {error_message}"
                )

            # Parse response
            response_data = response.json()
            
            # Transform to job list
            jobs = VertexAIFineTuningTransformation.transform_job_list_response(response_data)
            
            return FineTuningJobList(
                jobs=jobs,
                next_page_token=response_data.get("nextPageToken")
            )

        except Exception as e:
            verbose_logger.error(
                "Error listing fine-tuning jobs: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
            raise e

    async def _async_list_fine_tuning_jobs(
        self,
        list_url: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
    ) -> FineTuningJobList:
        """
        Async version of list fine-tuning jobs
        """
        try:
            if self.async_handler is None:
                raise ValueError("Async handler is not initialized")

            response = await self.async_handler.get(
                url=list_url,
                headers=headers,
                params=params,
            )

            if response.status_code != 200:
                error_message = self._extract_error_from_response(response.text)
                raise Exception(
                    f"Failed to list fine-tuning jobs. Status: {response.status_code}. Error: {error_message}"
                )

            # Parse response
            response_data = response.json()
            
            # Transform to job list
            jobs = VertexAIFineTuningTransformation.transform_job_list_response(response_data)
            
            return FineTuningJobList(
                jobs=jobs,
                next_page_token=response_data.get("nextPageToken")
            )

        except Exception as e:
            verbose_logger.error(
                "Async error listing fine-tuning jobs: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
            raise e

    def cancel_fine_tuning_job(
        self,
        job_id: str,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
        acompletion: bool = False,
    ) -> Union[FineTuningJobStatus, Coroutine[Any, Any, FineTuningJobStatus]]:
        """
        Cancel a fine-tuning job
        """
        try:
            # Get authentication
            access_token, project_id = self._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_project,
                custom_llm_provider="vertex_ai",
            )

            # Create cancel URL
            cancel_url = f"{VertexAIFineTuningTransformation.create_job_status_url(project_id, vertex_location or 'us-central1', job_id)}:cancel"

            # Set up headers
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {access_token}",
            }

            if acompletion:
                return self._async_cancel_fine_tuning_job(
                    cancel_url=cancel_url,
                    headers=headers,
                    job_id=job_id,
                )

            # Make synchronous request
            sync_handler = HTTPHandler()
            response = sync_handler.post(
                url=cancel_url,
                headers=headers,
                json={},  # Empty body for cancel request
            )

            if response.status_code != 200:
                error_message = self._extract_error_from_response(response.text)
                raise Exception(
                    f"Failed to cancel fine-tuning job. Status: {response.status_code}. Error: {error_message}"
                )

            # Return the updated job status
            return self.get_fine_tuning_job(
                job_id=job_id,
                vertex_project=project_id,
                vertex_location=vertex_location or "us-central1",
                vertex_credentials=vertex_credentials,
                acompletion=acompletion,
            )

        except Exception as e:
            verbose_logger.error(
                "Error canceling fine-tuning job: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
            raise e

    async def _async_cancel_fine_tuning_job(
        self,
        cancel_url: str,
        headers: Dict[str, str],
        job_id: str,
    ) -> FineTuningJobStatus:
        """
        Async version of cancel fine-tuning job
        """
        try:
            if self.async_handler is None:
                raise ValueError("Async handler is not initialized")

            response = await self.async_handler.post(
                url=cancel_url,
                headers=headers,
                json={},  # Empty body for cancel request
            )

            if response.status_code != 200:
                error_message = self._extract_error_from_response(response.text)
                raise Exception(
                    f"Failed to cancel fine-tuning job. Status: {response.status_code}. Error: {error_message}"
                )

            # Return the updated job status
            return await self.get_fine_tuning_job(
                job_id=job_id,
                acompletion=True,
            )

        except Exception as e:
            verbose_logger.error(
                "Async error canceling fine-tuning job: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
            raise e

    def estimate_fine_tuning_cost(
        self,
        model: str,
        training_file_size_mb: float,
        hyperparameters: Dict[str, Any],
    ) -> FineTuningCostEstimate:
        """
        Estimate the cost of fine-tuning
        """
        try:
            # Validate hyperparameters
            validated_hyperparameters = VertexAIFineTuningTransformation.validate_hyperparameters(
                hyperparameters
            )
            
            # Estimate cost
            cost_estimate = VertexAIFineTuningTransformation.estimate_fine_tuning_cost(
                model=model,
                training_file_size_mb=training_file_size_mb,
                hyperparameters=validated_hyperparameters,
            )
            
            return cost_estimate

        except Exception as e:
            verbose_logger.error(
                "Error estimating fine-tuning cost: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
            raise e

    def _extract_error_from_response(self, response_text: str) -> str:
        """
        Extract error message from Vertex AI response
        """
        try:
            error_data = json.loads(response_text)
            if "error" in error_data:
                return error_data["error"].get("message", "Unknown error")
            return response_text
        except json.JSONDecodeError:
            return response_text 