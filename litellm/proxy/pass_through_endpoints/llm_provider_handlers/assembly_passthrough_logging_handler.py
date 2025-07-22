import asyncio
import json
import time
from datetime import datetime
from typing import Literal, Optional, TypedDict
from urllib.parse import urlparse

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.types.passthrough_endpoints.assembly_ai import (
    ASSEMBLY_AI_MAX_POLLING_ATTEMPTS,
    ASSEMBLY_AI_POLLING_INTERVAL,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)


class AssemblyAITranscriptResponse(TypedDict, total=False):
    id: str
    speech_model: str
    acoustic_model: str
    language_code: str
    status: str
    audio_duration: float


class AssemblyAIPassthroughLoggingHandler:
    def __init__(self):
        self.assembly_ai_base_url = "https://api.assemblyai.com"
        self.assembly_ai_eu_base_url = "https://eu.assemblyai.com"
        """
        The base URL for the AssemblyAI API
        """

        self.polling_interval: float = ASSEMBLY_AI_POLLING_INTERVAL
        """
        The polling interval for the AssemblyAI API. 
        litellm needs to poll the GET /transcript/{transcript_id} endpoint to get the status of the transcript.
        """

        self.max_polling_attempts = ASSEMBLY_AI_MAX_POLLING_ATTEMPTS
        """
        The maximum number of polling attempts for the AssemblyAI API.
        """

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

        model = response_body.get("speech_model", "")
        verbose_proxy_logger.debug(
            "response body %s", json.dumps(response_body, indent=4)
        )
        kwargs["model"] = model
        kwargs["custom_llm_provider"] = "assemblyai"
        response_cost: Optional[float] = None

        transcript_id = response_body.get("id")
        if transcript_id is None:
            raise ValueError(
                "Transcript ID is required to log the cost of the transcription"
            )
        transcript_response = self._poll_assembly_for_transcript_response(
            transcript_id=transcript_id, url_route=url_route
        )
        verbose_proxy_logger.debug(
            "finished polling assembly for transcript response- got transcript response %s",
            json.dumps(transcript_response, indent=4),
        )
        if transcript_response:
            cost = self.get_cost_for_assembly_transcript(
                speech_model=model,
                transcript_response=transcript_response,
            )
            response_cost = cost

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
        logging_obj.model_call_details["model"] = model
        logging_obj.model_call_details["custom_llm_provider"] = "assemblyai"
        logging_obj.model_call_details["response_cost"] = response_cost

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

    def _get_assembly_transcript(
        self,
        transcript_id: str,
        request_region: Optional[Literal["eu"]] = None,
    ) -> Optional[dict]:
        """
        Get the transcript details from AssemblyAI API

        Args:
            response_body (dict): Response containing the transcript ID

        Returns:
            Optional[dict]: Transcript details if successful, None otherwise
        """
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            passthrough_endpoint_router,
        )

        _base_url = (
            self.assembly_ai_eu_base_url
            if request_region == "eu"
            else self.assembly_ai_base_url
        )
        _api_key = passthrough_endpoint_router.get_credentials(
            custom_llm_provider="assemblyai",
            region_name=request_region,
        )
        if _api_key is None:
            raise ValueError("AssemblyAI API key not found")
        try:
            url = f"{_base_url}/v2/transcript/{transcript_id}"
            headers = {
                "Authorization": f"Bearer {_api_key}",
                "Content-Type": "application/json",
            }

            response = httpx.get(url, headers=headers)
            response.raise_for_status()

            return response.json()
        except Exception as e:
            verbose_proxy_logger.exception(
                f"[Non blocking logging error] Error getting AssemblyAI transcript: {str(e)}"
            )
            return None

    def _poll_assembly_for_transcript_response(
        self,
        transcript_id: str,
        url_route: Optional[str] = None,
    ) -> Optional[AssemblyAITranscriptResponse]:
        """
        Poll the status of the transcript until it is completed or timeout (30 minutes)
        """
        for _ in range(
            self.max_polling_attempts
        ):  # 180 attempts * 10s = 30 minutes max
            transcript = self._get_assembly_transcript(
                request_region=AssemblyAIPassthroughLoggingHandler._get_assembly_region_from_url(
                    url=url_route
                ),
                transcript_id=transcript_id,
            )
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
        speech_model: str,
    ) -> Optional[float]:
        """
        Get the cost for the assembly transcript
        """
        _audio_duration = transcript_response.get("audio_duration")
        if _audio_duration is None:
            return None
        _cost_per_second = (
            AssemblyAIPassthroughLoggingHandler.get_cost_per_second_for_assembly_model(
                speech_model=speech_model
            )
        )
        if _cost_per_second is None:
            return None
        return _audio_duration * _cost_per_second

    @staticmethod
    def get_cost_per_second_for_assembly_model(speech_model: str) -> Optional[float]:
        """
        Get the cost per second for the assembly model.
        Falls back to assemblyai/nano if the specific speech model info cannot be found.
        """
        try:
            # First try with the provided speech model
            try:
                model_info = litellm.get_model_info(
                    model=speech_model,
                    custom_llm_provider="assemblyai",
                )
                if model_info and model_info.get("input_cost_per_second") is not None:
                    return model_info.get("input_cost_per_second")
            except Exception:
                pass  # Continue to fallback if model not found

            # Fallback to assemblyai/nano if speech model info not found
            try:
                model_info = litellm.get_model_info(
                    model="assemblyai/nano",
                    custom_llm_provider="assemblyai",
                )
                if model_info and model_info.get("input_cost_per_second") is not None:
                    return model_info.get("input_cost_per_second")
            except Exception:
                pass

            return None
        except Exception as e:
            verbose_proxy_logger.exception(
                f"[Non blocking logging error] Error getting AssemblyAI model info: {str(e)}"
            )
            return None

    @staticmethod
    def _should_log_request(request_method: str) -> bool:
        """
        only POST transcription jobs are logged. litellm will POLL assembly to wait for the transcription to complete to log the complete response / cost
        """
        return request_method == "POST"

    @staticmethod
    def _get_assembly_region_from_url(url: Optional[str]) -> Optional[Literal["eu"]]:
        """
        Get the region from the URL
        """
        if url is None:
            return None
        if urlparse(url).hostname == "eu.assemblyai.com":
            return "eu"
        return None

    @staticmethod
    def _get_assembly_base_url_from_region(region: Optional[Literal["eu"]]) -> str:
        """
        Get the base URL for the AssemblyAI API
        if region == "eu", return "https://api.eu.assemblyai.com"
        else return "https://api.assemblyai.com"
        """
        if region == "eu":
            return "https://api.eu.assemblyai.com"
        return "https://api.assemblyai.com"
