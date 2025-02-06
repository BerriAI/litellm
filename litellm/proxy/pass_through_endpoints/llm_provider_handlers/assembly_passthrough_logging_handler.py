import asyncio
import json
import os
import time
from datetime import datetime
from typing import Optional, TypedDict

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.proxy.pass_through_endpoints.types import PassthroughStandardLoggingPayload


class AssemblyAITranscriptResponse(TypedDict, total=False):
    id: str
    language_model: str
    acoustic_model: str
    language_code: str
    status: str
    audio_duration: float


class AssemblyAIPassthroughLoggingHandler:
    def __init__(self):
        self.assembly_ai_base_url = "https://api.assemblyai.com/v2"
        """
        The base URL for the AssemblyAI API
        """

        self.polling_interval: float = 10
        """
        The polling interval for the AssemblyAI API. 
        litellm needs to poll the GET /transcript/{transcript_id} endpoint to get the status of the transcript.
        """

        self.max_polling_attempts = 180
        """
        The maximum number of polling attempts for the AssemblyAI API.
        """

        self.assemblyai_api_key = os.environ.get("ASSEMBLYAI_API_KEY")

    def assemblyai_passthrough_logging_handler(
        self,
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs,
    ):
        """
        Since cost tracking requires polling the AssemblyAI API, we need to handle this in a separate thread. Hence the executor.submit.
        """
        executor.submit(
            self._handle_assemblyai_passthrough_logging,
            httpx_response,
            response_body,
            logging_obj,
            url_route,
            result,
            start_time,
            end_time,
            cache_hit,
            **kwargs,
        )

    def _handle_assemblyai_passthrough_logging(
        self,
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs,
    ):
        """
        Handles logging for AssemblyAI successful passthrough requests
        """
        from ..pass_through_endpoints import pass_through_endpoint_logging

        model = response_body.get("model", "")
        verbose_proxy_logger.debug("response body", json.dumps(response_body, indent=4))
        kwargs["model"] = model
        kwargs["custom_llm_provider"] = "assemblyai"

        transcript_id = response_body.get("id")
        if transcript_id is None:
            raise ValueError(
                "Transcript ID is required to log the cost of the transcription"
            )
        transcript_response = self._poll_assembly_for_transcript_response(transcript_id)
        verbose_proxy_logger.debug(
            "finished polling assembly for transcript response- got transcript response",
            json.dumps(transcript_response, indent=4),
        )
        if transcript_response:
            cost = self.get_cost_for_assembly_transcript(transcript_response)
            kwargs["response_cost"] = cost

        logging_obj.model_call_details["model"] = logging_obj.model

        # Make standard logging object for Vertex AI
        standard_logging_object = get_standard_logging_object_payload(
            kwargs=kwargs,
            init_response_obj=transcript_response,
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
            status="success",
        )

        passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload] = (  # type: ignore
            kwargs.get("passthrough_logging_payload")
        )

        verbose_proxy_logger.debug(
            "standard_passthrough_logging_object %s",
            json.dumps(passthrough_logging_payload, indent=4),
        )

        # pretty print standard logging object
        verbose_proxy_logger.debug(
            "standard_logging_object= %s", json.dumps(standard_logging_object, indent=4)
        )
        asyncio.run(
            pass_through_endpoint_logging._handle_logging(
                logging_obj=logging_obj,
                standard_logging_response_object=self._get_response_to_log(
                    transcript_response
                ),
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                **kwargs,
            )
        )

        pass

    def _get_response_to_log(
        self, transcript_response: Optional[AssemblyAITranscriptResponse]
    ) -> dict:
        if transcript_response is None:
            return {}
        return dict(transcript_response)

    def _get_assembly_transcript(self, transcript_id: str) -> Optional[dict]:
        """
        Get the transcript details from AssemblyAI API

        Args:
            response_body (dict): Response containing the transcript ID

        Returns:
            Optional[dict]: Transcript details if successful, None otherwise
        """
        try:
            url = f"{self.assembly_ai_base_url}/transcript/{transcript_id}"
            headers = {
                "Authorization": f"Bearer {self.assemblyai_api_key}",
                "Content-Type": "application/json",
            }

            response = httpx.get(url, headers=headers)
            response.raise_for_status()

            return response.json()
        except Exception as e:
            verbose_proxy_logger.debug(f"Error getting AssemblyAI transcript: {str(e)}")
            return None

    def _poll_assembly_for_transcript_response(
        self, transcript_id: str
    ) -> Optional[AssemblyAITranscriptResponse]:
        """
        Poll the status of the transcript until it is completed or timeout (30 minutes)
        """
        for _ in range(
            self.max_polling_attempts
        ):  # 180 attempts * 10s = 30 minutes max
            transcript = self._get_assembly_transcript(transcript_id)
            if transcript is None:
                return None
            if (
                transcript.get("status") == "completed"
                or transcript.get("status") == "error"
            ):
                return AssemblyAITranscriptResponse(**transcript)
            time.sleep(self.polling_interval)
        return None

    @staticmethod
    def get_cost_for_assembly_transcript(
        transcript_response: AssemblyAITranscriptResponse,
    ) -> Optional[float]:
        """
        Get the cost for the assembly transcript
        """
        _audio_duration = transcript_response.get("audio_duration")
        if _audio_duration is None:
            return None
        return _audio_duration * 0.0001

    @staticmethod
    def _should_log_request(request_method: str) -> bool:
        """
        only POST transcription jobs are logged. litellm will POLL assembly to wait for the transcription to complete to log the complete response / cost
        """
        return request_method == "POST"
