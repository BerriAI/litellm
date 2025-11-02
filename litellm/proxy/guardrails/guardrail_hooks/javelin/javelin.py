from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Type, Union

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
from litellm.types.proxy.guardrails.guardrail_hooks.javelin import (
    JavelinGuardInput,
    JavelinGuardRequest,
    JavelinGuardResponse,
)
from litellm.types.utils import GuardrailStatus

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class JavelinGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        default_on: bool = True,
        guardrail_name: str = "trustsafety",
        javelin_guard_name: Optional[str] = None,
        api_version: str = "v1",
        metadata: Optional[Dict] = None,
        config: Optional[Dict] = None,
        application: Optional[str] = None,
        **kwargs,
    ):
        f"""
        Initialize the JavelinGuardrail class.

        This calls: {api_base}/{api_version}/guardrail/{guardrail_name}/apply

        Args:
            api_key: str = None,
            api_base: str = None,
            default_on: bool = True,
            api_version: str = "v1",
            guardrail_name: str = "trustsafety",
            metadata: Optional[Dict] = None,
            config: Optional[Dict] = None,
            application: Optional[str] = None,
        """

        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.javelin_api_key = api_key or get_secret_str("JAVELIN_API_KEY")
        self.api_base = (
            api_base
            or get_secret_str("JAVELIN_API_BASE")
            or "https://api-dev.javelin.live"
        )
        self.api_version = api_version
        self.guardrail_name = guardrail_name
        self.javelin_guard_name = javelin_guard_name or "javelin_guard"
        self.default_on = default_on
        self.metadata = metadata
        self.config = config
        self.application = application
        verbose_proxy_logger.debug(
            "Javelin Guardrail: Initialized with guardrail_name=%s, javelin_guard_name=%s, api_base=%s, api_version=%s",
            self.guardrail_name,
            self.javelin_guard_name,
            self.api_base,
            self.api_version,
        )

        super().__init__(guardrail_name=guardrail_name, default_on=default_on, **kwargs)

    async def call_javelin_guard(
        self,
        request: JavelinGuardRequest,
    ) -> JavelinGuardResponse:
        """
        Call the Javelin guard API.
        """
        start_time = datetime.now()
        # Create a new request with metadata if it's not already set
        if request.get("metadata") is None and self.metadata is not None:
            request = {**request, "metadata": self.metadata}
        headers = {
            "x-javelin-apikey": self.javelin_api_key,
        }
        if self.application:
            headers["x-javelin-application"] = self.application

        status: GuardrailStatus = "guardrail_failed_to_respond"
        javelin_response: Optional[JavelinGuardResponse] = None
        exception_str = ""

        try:
            url = f"{self.api_base}/{self.api_version}/guardrail/{self.javelin_guard_name}/apply"
            if self.javelin_guard_name == "javelin_guard":
                # auto apply all enabled guardrails in app policy, overwrite url
                url = f"{self.api_base}/{self.api_version}/guardrails/apply"
            verbose_proxy_logger.debug("Javelin Guardrail: Calling URL: %s", url)
            response = await self.async_handler.post(
                url=url,
                headers=headers,
                json=dict(request),
            )
            verbose_proxy_logger.debug(
                "Javelin Guardrail: Javelin guard API response: %s", response.json()
            )
            response_data = response.json()
            # Ensure the response has the required assessments field
            if "assessments" not in response_data:
                response_data["assessments"] = []

            javelin_response = {"assessments": response_data.get("assessments", [])}
            status = "success"
            return javelin_response
        except Exception as e:
            status = "guardrail_failed_to_respond"
            exception_str = str(e)
            return {"assessments": []}
        finally:
            ####################################################
            # Create Guardrail Trace for logging on Langfuse, Datadog, etc.
            ####################################################
            guardrail_json_response: Union[Exception, str, dict, List[dict]] = {}
            if status == "success" and javelin_response is not None:
                guardrail_json_response = dict(javelin_response)
            else:
                guardrail_json_response = exception_str

            # Create a clean request data copy for logging (without guardrail responses)
            clean_request_data = {
                "input": request.get("input", {}),
                "metadata": request.get("metadata", {}),
                "config": request.get("config", {}),
            }
            # Remove any existing guardrail logging information to prevent recursion
            if "metadata" in clean_request_data and clean_request_data["metadata"]:
                clean_request_data["metadata"] = {
                    k: v
                    for k, v in clean_request_data["metadata"].items()
                    if k != "standard_logging_guardrail_information"
                }

            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response=guardrail_json_response,
                request_data=clean_request_data,
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
            )

    def _process_assessments(
        self, assessments: List[Dict]
    ) -> Tuple[bool, bool, Optional[str], Optional[str]]:
        """
        Process Javelin assessments to determine if content should be rejected or transformed.

        Returns:
            Tuple of (should_reject, should_transform_content, reject_prompt, transformed_content)
        """
        should_reject = False
        should_transform_content = False
        reject_prompt = "Violated guardrail policy"
        transformed_content = None

        for assessment in assessments:
            verbose_proxy_logger.debug(
                "Javelin Guardrail: Processing assessment: %s", assessment
            )
            for assessment_type, assessment_data in assessment.items():
                verbose_proxy_logger.debug(
                    "Javelin Guardrail: Processing assessment_type: %s, data: %s",
                    assessment_type,
                    assessment_data,
                )

                results = assessment_data.get("results", {})
                strategy = results.get("strategy", "")

                # Check if this assessment indicates rejection
                if assessment_data.get("request_reject") is True:
                    should_reject = True
                    verbose_proxy_logger.debug(
                        "Javelin Guardrail: Request rejected by Javelin guardrail: %s (assessment_type: %s)",
                        self.guardrail_name,
                        assessment_type,
                    )

                    reject_prompt = str(results.get("reject_prompt", ""))

                    verbose_proxy_logger.debug(
                        "Javelin Guardrail: Extracted reject_prompt: '%s'",
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
                        "Javelin Guardrail: Content transformation detected: strategy=%s, content=%s",
                        strategy,
                        transformed_content,
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
        """Apply content transformation to the request data."""
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            set_last_user_message,
        )

        verbose_proxy_logger.debug(
            "Javelin Guardrail: Applying content transformation to messages"
        )
        try:
            # Update the messages with the transformed content
            data["messages"] = set_last_user_message(
                data["messages"], transformed_content
            )
            verbose_proxy_logger.debug(
                "Javelin Guardrail: Successfully updated messages with transformed content"
            )
        except Exception as e:
            verbose_proxy_logger.error(
                "Javelin Guardrail: Failed to update messages with transformed content: %s",
                e,
            )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: litellm.DualCache,
        data: Dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Optional[Union[Exception, str, Dict]]:
        """
        Pre-call hook for the Javelin guardrail.
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_last_user_message,
        )
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        verbose_proxy_logger.debug("Javelin Guardrail: pre_call_hook")
        verbose_proxy_logger.debug("Javelin Guardrail: Request data: %s", data)

        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                "Javelin Guardrail: not running guardrail. Guardrail is disabled."
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

        javelin_guard_request = JavelinGuardRequest(
            input=JavelinGuardInput(text=text),
            metadata=clean_metadata,
            config=self.config if self.config else {},
        )

        javelin_response = await self.call_javelin_guard(request=javelin_guard_request)

        # Debug: Log the full Javelin response
        verbose_proxy_logger.debug(
            "Javelin Guardrail: Full Javelin response: %s", javelin_response
        )

        # Process assessments to determine action
        assessments = javelin_response.get("assessments", [])
        (
            should_reject,
            should_transform_content,
            reject_prompt,
            transformed_content,
        ) = self._process_assessments(assessments)

        verbose_proxy_logger.debug(
            "Javelin Guardrail: should_reject=%s, should_transform_content=%s, reject_prompt='%s'",
            should_reject,
            should_transform_content,
            reject_prompt,
        )

        # Handle content transformation if needed
        if should_transform_content and transformed_content is not None:
            self._apply_content_transformation(data, transformed_content)

        if should_reject:
            if not reject_prompt:
                reject_prompt = f"Request blocked by Javelin guardrails due to {self.guardrail_name} violation."

            verbose_proxy_logger.debug(
                "Javelin Guardrail: Blocking request with reject_prompt: '%s'",
                reject_prompt,
            )

            # Raise HTTPException to prevent the request from going to the LLM
            raise HTTPException(
                status_code=400,
                detail={
                    "error": reject_prompt,
                    "javelin_guardrail_response": javelin_response,
                },
            )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return data

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        """
        Get the config model for the Javelin guardrail.
        """
        from litellm.types.proxy.guardrails.guardrail_hooks.javelin import (
            JavelinGuardrailConfigModel,
        )

        return JavelinGuardrailConfigModel
