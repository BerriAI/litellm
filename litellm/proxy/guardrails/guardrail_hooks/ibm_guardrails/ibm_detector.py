# +-------------------------------------------------------------+
#
#           Use IBM Guardrails Detector for your LLM calls
#           Based on IBM's FMS Guardrails
#
# +-------------------------------------------------------------+

import os
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.ibm import (
    IBMDetectorDetection,
    IBMDetectorResponseOrchestrator,
)
from litellm.types.utils import CallTypesLiteral, GuardrailStatus, ModelResponseStream

GUARDRAIL_NAME = "ibm_guardrails"


class IBMGuardrailDetector(CustomGuardrail):
    def __init__(
        self,
        guardrail_name: str = "ibm_detector",
        auth_token: Optional[str] = None,
        base_url: Optional[str] = None,
        detector_id: Optional[str] = None,
        is_detector_server: bool = True,
        detector_params: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        score_threshold: Optional[float] = None,
        block_on_detection: bool = True,
        verify_ssl: bool = True,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
            params={"ssl_verify": verify_ssl},
        )

        # Set API configuration
        self.auth_token = auth_token or os.getenv("IBM_GUARDRAILS_AUTH_TOKEN")
        if not self.auth_token:
            raise ValueError(
                "IBM Guardrails auth token is required. Set IBM_GUARDRAILS_AUTH_TOKEN environment variable or pass auth_token parameter."
            )

        self.base_url = base_url
        if not self.base_url:
            raise ValueError(
                "IBM Guardrails base_url is required. Pass base_url parameter."
            )

        self.detector_id = detector_id
        if not self.detector_id:
            raise ValueError(
                "IBM Guardrails detector_id is required. Pass detector_id parameter."
            )

        self.is_detector_server = is_detector_server
        self.detector_params = detector_params or {}
        self.extra_headers = extra_headers or {}
        self.score_threshold = score_threshold
        self.block_on_detection = block_on_detection
        self.verify_ssl = verify_ssl

        # Construct API URL based on server type
        if self.is_detector_server:
            self.api_url = f"{self.base_url}/api/v1/text/contents"
        else:
            self.api_url = f"{self.base_url}/api/v2/text/detection/content"

        self.guardrail_name = guardrail_name
        self.guardrail_provider = "ibm_guardrails"

        # store kwargs as optional_params
        self.optional_params = kwargs

        # Set supported event hooks
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
            ]

        super().__init__(guardrail_name=guardrail_name, **kwargs)

        verbose_proxy_logger.debug(
            "IBM Guardrail Detector initialized with guardrail_name: %s, detector_id: %s, is_detector_server: %s",
            self.guardrail_name,
            self.detector_id,
            self.is_detector_server,
        )

    async def _call_detector_server(
        self,
        contents: List[str],
        event_type: GuardrailEventHooks,
        request_data: Optional[dict] = None,
    ) -> List[List[IBMDetectorDetection]]:
        """
        Call IBM Detector Server directly.

        Args:
            contents: List of text strings to analyze
            request_data: Optional request data for logging purposes

        Returns:
            List of lists where top-level list is per message in contents,
            sublists are individual detections on that message
        """
        start_time = datetime.now()

        payload = {"contents": contents, "detector_params": self.detector_params}

        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "content-type": "application/json",
            "detector-id": self.detector_id,
        }

        # Add any extra headers to the request
        for header, value in self.extra_headers.items():
            headers[header] = value

        verbose_proxy_logger.debug(
            "IBM Detector Server request to %s with payload: %s",
            self.api_url,
            payload,
        )

        try:
            response = await self.async_handler.post(
                url=self.api_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            response_json: List[List[IBMDetectorDetection]] = response.json()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Add guardrail information to request trace
            if request_data:
                guardrail_status = self._determine_guardrail_status_detector_server(
                    response_json
                )
                self.add_standard_logging_guardrail_information_to_request_data(
                    guardrail_provider=self.guardrail_provider,
                    guardrail_json_response={
                        "detections": [
                            [detection for detection in message_detections]
                            for message_detections in response_json
                        ]
                    },
                    request_data=request_data,
                    guardrail_status=guardrail_status,
                    start_time=start_time.timestamp(),
                    end_time=end_time.timestamp(),
                    duration=duration,
                    event_type=event_type,
                )

            return response_json

        except httpx.HTTPError as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            verbose_proxy_logger.error("IBM Detector Server request failed: %s", str(e))

            # Add guardrail information with failure status
            if request_data:
                self.add_standard_logging_guardrail_information_to_request_data(
                    guardrail_provider=self.guardrail_provider,
                    guardrail_json_response={"error": str(e)},
                    request_data=request_data,
                    guardrail_status="guardrail_failed_to_respond",
                    start_time=start_time.timestamp(),
                    end_time=end_time.timestamp(),
                    duration=duration,
                    event_type=event_type,
                )

            raise

    async def _call_orchestrator(
        self,
        content: str,
        event_type: GuardrailEventHooks,
        request_data: Optional[dict] = None,
    ) -> List[IBMDetectorDetection]:
        """
        Call IBM FMS Guardrails Orchestrator.

        Args:
            content: Text string to analyze
            request_data: Optional request data for logging purposes

        Returns:
            List of detections
        """
        start_time = datetime.now()

        payload = {
            "content": content,
            "detectors": {self.detector_id: self.detector_params},
        }

        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "content-type": "application/json",
        }

        # Add any extra headers to the request
        for header, value in self.extra_headers.items():
            headers[header] = value

        verbose_proxy_logger.debug(
            "IBM Orchestrator request to %s with payload: %s",
            self.api_url,
            payload,
        )

        try:
            response = await self.async_handler.post(
                url=self.api_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            response_json: IBMDetectorResponseOrchestrator = response.json()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Add guardrail information to request trace
            if request_data:
                guardrail_status = self._determine_guardrail_status_orchestrator(
                    response_json
                )
                self.add_standard_logging_guardrail_information_to_request_data(
                    guardrail_provider=self.guardrail_provider,
                    guardrail_json_response=dict(response_json),
                    request_data=request_data,
                    guardrail_status=guardrail_status,
                    start_time=start_time.timestamp(),
                    end_time=end_time.timestamp(),
                    duration=duration,
                    event_type=event_type,
                )

            return response_json.get("detections", [])

        except httpx.HTTPError as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            verbose_proxy_logger.error("IBM Orchestrator request failed: %s", str(e))

            # Add guardrail information with failure status
            if request_data:
                self.add_standard_logging_guardrail_information_to_request_data(
                    guardrail_provider=self.guardrail_provider,
                    guardrail_json_response={"error": str(e)},
                    request_data=request_data,
                    guardrail_status="guardrail_failed_to_respond",
                    start_time=start_time.timestamp(),
                    end_time=end_time.timestamp(),
                    duration=duration,
                    event_type=event_type,
                )

            raise

    def _filter_detections_by_threshold(
        self, detections: List[IBMDetectorDetection]
    ) -> List[IBMDetectorDetection]:
        """
        Filter detections based on score threshold.

        Args:
            detections: List of detections

        Returns:
            Filtered list of detections that meet the threshold
        """
        if self.score_threshold is None:
            return detections

        return [
            detection
            for detection in detections
            if detection.get("score", 0.0) >= self.score_threshold
        ]

    def _determine_guardrail_status_detector_server(
        self, response_json: List[List[IBMDetectorDetection]]
    ) -> GuardrailStatus:
        """
        Determine the guardrail status based on IBM Detector Server response.

        Returns:
            "success": Content allowed through with no violations
            "guardrail_intervened": Content blocked due to detections
            "guardrail_failed_to_respond": Technical error or API failure
        """
        try:
            if not isinstance(response_json, list):
                return "guardrail_failed_to_respond"

            # Check if any detections were found
            has_detections = False
            for message_detections in response_json:
                if message_detections:
                    # Apply threshold filtering
                    filtered = self._filter_detections_by_threshold(message_detections)
                    if filtered:
                        has_detections = True
                        break

            if has_detections:
                return "guardrail_intervened"

            return "success"

        except Exception as e:
            verbose_proxy_logger.error(
                "Error determining IBM Detector Server guardrail status: %s", str(e)
            )
            return "guardrail_failed_to_respond"

    def _determine_guardrail_status_orchestrator(
        self, response_json: IBMDetectorResponseOrchestrator
    ) -> GuardrailStatus:
        """
        Determine the guardrail status based on IBM Orchestrator response.

        Returns:
            "success": Content allowed through with no violations
            "guardrail_intervened": Content blocked due to detections
            "guardrail_failed_to_respond": Technical error or API failure
        """
        try:
            if not isinstance(response_json, dict):
                return "guardrail_failed_to_respond"

            detections = response_json.get("detections", [])
            # Apply threshold filtering
            filtered = self._filter_detections_by_threshold(detections)

            if filtered:
                return "guardrail_intervened"

            return "success"

        except Exception as e:
            verbose_proxy_logger.error(
                "Error determining IBM Orchestrator guardrail status: %s", str(e)
            )
            return "guardrail_failed_to_respond"

    def _create_error_message_detector_server(
        self, detections_list: List[List[IBMDetectorDetection]]
    ) -> str:
        """
        Create a detailed error message from detector server response.

        Args:
            detections_list: List of lists of detections

        Returns:
            Formatted error message string
        """
        total_detections = 0
        error_message = "IBM Guardrail Detector failed:\n\n"

        for idx, message_detections in enumerate(detections_list):
            filtered_detections = self._filter_detections_by_threshold(
                message_detections
            )
            if filtered_detections:
                error_message += f"Message {idx + 1}:\n"
                total_detections += len(filtered_detections)

                for detection in filtered_detections:
                    detection_type = detection.get("detection_type", "unknown")
                    score = detection.get("score", 0.0)
                    text = detection.get("text", "")
                    error_message += (
                        f"  - {detection_type.upper()} (score: {score:.3f})\n"
                    )
                    error_message += f"    Text: '{text}'\n"

                error_message += "\n"

        error_message = (
            f"IBM Guardrail Detector failed: {total_detections} violation(s) detected\n\n"
            + error_message
        )
        return error_message.strip()

    def _create_error_message_orchestrator(
        self, detections: List[IBMDetectorDetection]
    ) -> str:
        """
        Create a detailed error message from orchestrator response.

        Args:
            detections: List of detections

        Returns:
            Formatted error message string
        """
        filtered_detections = self._filter_detections_by_threshold(detections)

        error_message = f"IBM Guardrail Detector failed: {len(filtered_detections)} violation(s) detected\n\n"

        for detection in filtered_detections:
            detection_type = detection.get("detection_type", "unknown")
            detector_id = detection.get("detector_id", self.detector_id)
            score = detection.get("score", 0.0)
            text = detection.get("text", "")

            error_message += f"- {detection_type.upper()} (detector: {detector_id}, score: {score:.3f})\n"
            error_message += f"  Text: '{text}'\n\n"

        return error_message.strip()

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        """
        Runs before the LLM API call
        Runs on only Input
        Use this if you want to MODIFY the input
        """
        verbose_proxy_logger.debug("Running IBM Guardrail Detector pre-call hook")

        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        _messages = data.get("messages")
        if _messages:
            contents_to_check: List[str] = []
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    contents_to_check.append(_content)

            if contents_to_check:
                if self.is_detector_server:
                    # Call detector server with all contents at once
                    result = await self._call_detector_server(
                        contents=contents_to_check,
                        request_data=data,
                        event_type=GuardrailEventHooks.pre_call,
                    )

                    verbose_proxy_logger.debug(
                        "IBM Detector Server async_pre_call_hook result: %s", result
                    )

                    # Check if any detections were found
                    has_violations = False
                    for message_detections in result:
                        filtered = self._filter_detections_by_threshold(
                            message_detections
                        )
                        if filtered:
                            has_violations = True
                            break

                    if has_violations and self.block_on_detection:
                        error_message = self._create_error_message_detector_server(
                            result
                        )
                        raise ValueError(error_message)

                else:
                    # Call orchestrator for each content separately
                    for content in contents_to_check:
                        orchestrator_result = await self._call_orchestrator(
                            content=content,
                            request_data=data,
                            event_type=GuardrailEventHooks.pre_call,
                        )

                        verbose_proxy_logger.debug(
                            "IBM Orchestrator async_pre_call_hook result: %s",
                            orchestrator_result,
                        )

                        filtered = self._filter_detections_by_threshold(
                            orchestrator_result
                        )
                        if filtered and self.block_on_detection:
                            error_message = self._create_error_message_orchestrator(
                                orchestrator_result
                            )
                            raise ValueError(error_message)

        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ):
        """
        Runs in parallel to LLM API call
        Runs on only Input

        This can NOT modify the input, only used to reject or accept a call before going to LLM API
        """
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return

        _messages = data.get("messages")
        if _messages:
            contents_to_check: List[str] = []
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    contents_to_check.append(_content)

            if contents_to_check:
                if self.is_detector_server:
                    # Call detector server with all contents at once
                    result = await self._call_detector_server(
                        contents=contents_to_check,
                        request_data=data,
                        event_type=GuardrailEventHooks.during_call,
                    )

                    verbose_proxy_logger.debug(
                        "IBM Detector Server async_moderation_hook result: %s", result
                    )

                    # Check if any detections were found
                    has_violations = False
                    for message_detections in result:
                        filtered = self._filter_detections_by_threshold(
                            message_detections
                        )
                        if filtered:
                            has_violations = True
                            break

                    if has_violations and self.block_on_detection:
                        error_message = self._create_error_message_detector_server(
                            result
                        )
                        raise ValueError(error_message)

                else:
                    # Call orchestrator for each content separately
                    for content in contents_to_check:
                        orchestrator_result = await self._call_orchestrator(
                            content=content,
                            request_data=data,
                            event_type=GuardrailEventHooks.during_call,
                        )

                        verbose_proxy_logger.debug(
                            "IBM Orchestrator async_moderation_hook result: %s",
                            orchestrator_result,
                        )

                        filtered = self._filter_detections_by_threshold(
                            orchestrator_result
                        )
                        if filtered and self.block_on_detection:
                            error_message = self._create_error_message_orchestrator(
                                orchestrator_result
                            )
                            raise ValueError(error_message)

        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Runs on response from LLM API call

        It can be used to reject a response

        Uses IBM Guardrails Detector to check the response for violations
        """
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )
        from litellm.types.guardrails import GuardrailEventHooks

        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            return

        verbose_proxy_logger.debug(
            "async_post_call_success_hook response: %s", response
        )

        # Check if the ModelResponse has text content in its choices
        # to avoid sending empty content to IBM Detector (e.g., during tool calls)
        if isinstance(response, litellm.ModelResponse):
            has_text_content = False
            for choice in response.choices:
                if isinstance(choice, litellm.Choices):
                    if choice.message.content and isinstance(
                        choice.message.content, str
                    ):
                        has_text_content = True
                        break

            if not has_text_content:
                verbose_proxy_logger.warning(
                    "IBM Guardrail Detector: not running guardrail. No output text in response"
                )
                return

            contents_to_check: List[str] = []
            for choice in response.choices:
                if isinstance(choice, litellm.Choices):
                    verbose_proxy_logger.debug(
                        "async_post_call_success_hook choice: %s", choice
                    )
                    if choice.message.content and isinstance(
                        choice.message.content, str
                    ):
                        contents_to_check.append(choice.message.content)

            if contents_to_check:
                if self.is_detector_server:
                    # Call detector server with all contents at once
                    result = await self._call_detector_server(
                        contents=contents_to_check,
                        request_data=data,
                        event_type=GuardrailEventHooks.post_call,
                    )

                    verbose_proxy_logger.debug(
                        "IBM Detector Server async_post_call_success_hook result: %s",
                        result,
                    )

                    # Check if any detections were found
                    has_violations = False
                    for message_detections in result:
                        filtered = self._filter_detections_by_threshold(
                            message_detections
                        )
                        if filtered:
                            has_violations = True
                            break

                    if has_violations and self.block_on_detection:
                        error_message = self._create_error_message_detector_server(
                            result
                        )
                        raise ValueError(error_message)

                else:
                    # Call orchestrator for each content separately
                    for content in contents_to_check:
                        orchestrator_result = await self._call_orchestrator(
                            content=content,
                            request_data=data,
                            event_type=GuardrailEventHooks.post_call,
                        )

                        verbose_proxy_logger.debug(
                            "IBM Orchestrator async_post_call_success_hook result: %s",
                            orchestrator_result,
                        )

                        filtered = self._filter_detections_by_threshold(
                            orchestrator_result
                        )
                        if filtered and self.block_on_detection:
                            error_message = self._create_error_message_orchestrator(
                                orchestrator_result
                            )
                            raise ValueError(error_message)

        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Passes the entire stream to the guardrail

        This is useful for guardrails that need to see the entire response, such as PII masking.

        Triggered by mode: 'post_call'
        """
        async for item in response:
            yield item

    @staticmethod
    def get_config_model():
        from litellm.types.proxy.guardrails.guardrail_hooks.ibm import (
            IBMDetectorGuardrailConfigModel,
        )

        return IBMDetectorGuardrailConfigModel
