from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.highflame import (
    HighflameGuardInput,
    HighflameGuardRequest,
    HighflameGuardResponse,
)
from litellm.types.utils import CallTypesLiteral, GuardrailStatus

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class HighflameGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        default_on: bool = True,
        guardrail_name: str = "trustsafety",
        highflame_guard_name: Optional[str] = None,
        api_version: str = "v1",
        metadata: Optional[Dict] = None,
        config: Optional[Dict] = None,
        application: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the HighflameGuardrail class.

        For single guardrails: {api_base}/{api_version}/guardrail/{guard_name}/apply
        For multi-guard (highflame_guard): {api_base}/{api_version}/guardrails/apply

        See https://docs.highflame.ai/documentation/agent-control-fabric/guardrails-policies/guardrail-apis

        Args:
            api_key: API key for Highflame service.
            api_base: Base URL for Highflame API.
            default_on: Whether the guardrail is enabled by default.
            guardrail_name: Name used within litellm for this guardrail instance.
            highflame_guard_name: Name of the Highflame guard to call.
            api_version: API version for Highflame service.
            metadata: Additional metadata to send with requests.
            config: Configuration parameters for the guardrail.
            application: Application name for policy-specific guardrails.
        """

        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.highflame_api_key = api_key or get_secret_str("HIGHFLAME_API_KEY")
        self.api_base = (
            api_base
            or get_secret_str("HIGHFLAME_API_BASE")
            or "https://api.highflame.app"
        )
        self.api_version = api_version
        self.guardrail_name = guardrail_name
        self.highflame_guard_name = highflame_guard_name or "highflame_guard"
        self.default_on = default_on
        self.metadata = metadata
        self.config = config
        self.application = application
        verbose_proxy_logger.debug(
            "Highflame Guardrail: Initialized with guardrail_name=%s, highflame_guard_name=%s, api_base=%s, api_version=%s",
            self.guardrail_name,
            self.highflame_guard_name,
            self.api_base,
            self.api_version,
        )

        super().__init__(guardrail_name=guardrail_name, default_on=default_on, **kwargs)

    async def call_highflame_guard(
        self,
        request: HighflameGuardRequest,
        event_type: GuardrailEventHooks,
    ) -> HighflameGuardResponse:
        """
        Call the Highflame guard API.

        For single guardrails: POST {api_base}/{api_version}/guardrail/{guard_name}/apply
        For multi-guard: POST {api_base}/{api_version}/guardrails/apply
        """
        start_time = datetime.now()
        if request.get("metadata") is None and self.metadata is not None:
            request = {**request, "metadata": self.metadata}
        headers = {
            "x-highflame-apikey": self.highflame_api_key,
        }
        if self.application:
            headers["x-highflame-application"] = self.application

        status: GuardrailStatus = "guardrail_failed_to_respond"
        highflame_response: Optional[HighflameGuardResponse] = None
        exception_str = ""

        try:
            url = f"{self.api_base}/{self.api_version}/guardrail/{self.highflame_guard_name}/apply"
            if self.highflame_guard_name == "highflame_guard":
                # Multi-guard: auto apply all enabled guardrails in app policy
                url = f"{self.api_base}/{self.api_version}/guardrails/apply"
            verbose_proxy_logger.debug("Highflame Guardrail: Calling URL: %s", url)
            response = await self.async_handler.post(
                url=url,
                headers=headers,
                json=dict(request),
            )
            verbose_proxy_logger.debug(
                "Highflame Guardrail: API response: %s", response.json()
            )
            response_data = response.json()
            if "assessments" not in response_data:
                response_data["assessments"] = []

            highflame_response = {"assessments": response_data.get("assessments", [])}
            status = "success"
            return highflame_response
        except Exception as e:
            verbose_proxy_logger.error(
                "Highflame Guardrail: API call failed: %s", e
            )
            status = "guardrail_failed_to_respond"
            exception_str = str(e)
            return {"assessments": []}
        finally:
            guardrail_json_response: Union[Exception, str, dict, List[dict]] = {}
            if status == "success" and highflame_response is not None:
                guardrail_json_response = dict(highflame_response)
            else:
                guardrail_json_response = exception_str

            clean_request_data = {
                "input": request.get("input", {}),
                "metadata": request.get("metadata", {}),
                "config": request.get("config", {}),
            }
            if "metadata" in clean_request_data and clean_request_data["metadata"]:
                clean_request_data["metadata"] = {
                    k: v
                    for k, v in clean_request_data["metadata"].items()
                    if k != "standard_logging_guardrail_information"
                }

            end_time = datetime.now()
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response=guardrail_json_response,
                request_data=clean_request_data,
            end_time = datetime.now()
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response=guardrail_json_response,
                request_data=clean_request_data,
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                duration=(end_time - start_time).total_seconds(),
                event_type=event_type,
            )

    def _process_assessments(
        self, assessments: List[Dict]
    ) -> Tuple[bool, bool, Optional[str], Optional[str]]:
        """
        Process Highflame assessments to determine if content should be rejected or transformed.

        Returns:
            Tuple of (should_reject, should_transform_content, reject_prompt, transformed_content)
        """
        should_reject = False
        should_transform_content = False
        reject_prompt = "Violated guardrail policy"
        transformed_content = None

        for assessment in assessments:
            verbose_proxy_logger.debug(
                "Highflame Guardrail: Processing assessment: %s", assessment
            )
            for assessment_type, assessment_data in assessment.items():
                verbose_proxy_logger.debug(
                    "Highflame Guardrail: Processing assessment_type: %s, data: %s",
                    assessment_type,
                    assessment_data,
                )

                results = assessment_data.get("results", {})
                strategy = results.get("strategy", "")

                # Check if this assessment indicates rejection
                if assessment_data.get("request_reject") is True:
                    should_reject = True
                    verbose_proxy_logger.debug(
                        "Highflame Guardrail: Request rejected by guardrail: %s (assessment_type: %s)",
                        self.guardrail_name,
                        assessment_type,
                    )
                    reject_prompt = str(results.get("reject_prompt", ""))
                    verbose_proxy_logger.debug(
                        "Highflame Guardrail: Extracted reject_prompt: '%s'",
                        reject_prompt,
                    )
                    break

                # Check if content transformation is needed (for DLP processors)
                elif (
                    strategy.lower() in ["mask", "redact", "replace"]
                    and "content" in results
                ):
                    should_transform_content = True
                    transformed_content = results.get("content")
                    verbose_proxy_logger.debug(
                        "Highflame Guardrail: Content transformation detected: strategy=%s",
                        strategy,
                    )
                    break

            if should_reject or should_transform_content:
                break

        return (
            should_reject,
            should_transform_content,
            reject_prompt,
            transformed_content,
        )

    def _apply_content_transformation(
        self, data: Dict, transformed_content: str
    ) -> None:
        """Apply content transformation (e.g. DLP masking/redaction) to the request data."""
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            set_last_user_message,
        )

        verbose_proxy_logger.debug(
            "Highflame Guardrail: Applying content transformation to messages"
        )
        try:
            data["messages"] = set_last_user_message(
                data["messages"], transformed_content
            )
            verbose_proxy_logger.debug(
                "Highflame Guardrail: Successfully updated messages with transformed content"
            )
        except Exception as e:
            verbose_proxy_logger.error(
                "Highflame Guardrail: Failed to update messages with transformed content: %s",
                e,
            )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: litellm.DualCache,
        data: Dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, Dict]]:
        """
        Pre-call hook for the Highflame guardrail.
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_last_user_message,
        )
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        verbose_proxy_logger.debug("Highflame Guardrail: pre_call_hook")

        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                "Highflame Guardrail: not running guardrail. Guardrail is disabled."
            )
            return data

        if "messages" not in data:
            return data

        text = get_last_user_message(data["messages"])
        if text is None:
            return data

        clean_metadata = {}
        if self.metadata:
            clean_metadata = {
                k: v
                for k, v in self.metadata.items()
                if k != "standard_logging_guardrail_information"
            }

        highflame_guard_request = HighflameGuardRequest(
            input=HighflameGuardInput(text=text),
            metadata=clean_metadata,
            config=self.config if self.config else {},
        )

        highflame_response = await self.call_highflame_guard(
            request=highflame_guard_request,
            event_type=GuardrailEventHooks.pre_call,
        )

        verbose_proxy_logger.debug(
            "Highflame Guardrail: Full response: %s", highflame_response
        )

        assessments = highflame_response.get("assessments", [])
        (
            should_reject,
            should_transform_content,
            reject_prompt,
            transformed_content,
        ) = self._process_assessments(assessments)

        verbose_proxy_logger.debug(
            "Highflame Guardrail: should_reject=%s, should_transform_content=%s, reject_prompt='%s'",
            should_reject,
            should_transform_content,
            reject_prompt,
        )

        # Handle content transformation (DLP masking/redaction)
        if should_transform_content and transformed_content is not None:
            self._apply_content_transformation(data, transformed_content)

        if should_reject:
            if not reject_prompt:
                reject_prompt = f"Request blocked by Highflame guardrails due to {self.guardrail_name} violation."

            verbose_proxy_logger.debug(
                "Highflame Guardrail: Blocking request with reject_prompt: '%s'",
                reject_prompt,
            )

            raise HTTPException(
                status_code=400,
                detail={
                    "error": reject_prompt,
                    "highflame_guardrail_response": highflame_response,
                },
            )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return data

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.highflame import (
            HighflameGuardrailConfigModel,
        )

        return HighflameGuardrailConfigModel
