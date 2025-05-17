import copy
import os
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple, Union

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import (
    LakeraAIRequest,
    LakeraAIResponse,
)


class LakeraAIGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        project_id: Optional[str] = None,
        payload: Optional[bool] = True,
        breakdown: Optional[bool] = True,
        metadata: Optional[Dict] = None,
        dev_info: Optional[bool] = True,
        **kwargs,
    ):
        """
        Initialize the LakeraAIGuardrail class.

        This calls: https://api.lakera.ai/v2/guard

        Args:
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
            project_id: Optional[str] = None,
            payload: Optional[bool] = True,
            breakdown: Optional[bool] = True,
            metadata: Optional[Dict] = None,
            dev_info: Optional[bool] = True,
        """
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.lakera_api_key = api_key or os.environ["LAKERA_API_KEY"]
        self.project_id = project_id
        self.api_base = (
            api_base or get_secret_str("LAKERA_API_BASE") or "https://api.lakera.ai"
        )
        self.payload: Optional[bool] = payload
        self.breakdown: Optional[bool] = breakdown
        self.metadata: Optional[Dict] = metadata
        self.dev_info: Optional[bool] = dev_info
        super().__init__(**kwargs)

    async def call_v2_guard(
        self,
        messages: List[AllMessageValues],
        request_data: Dict,
    ) -> Tuple[LakeraAIResponse, Dict]:
        """
        Call the Lakera AI v2 guard API.
        """
        status: Literal["success", "failure"] = "success"
        exception_str: str = ""
        start_time: datetime = datetime.now()
        lakera_response: Optional[LakeraAIResponse] = None
        request: Dict = {}
        masked_entity_count: Dict = {}
        try:
            request = dict(
                LakeraAIRequest(
                    messages=messages,
                    project_id=self.project_id,
                    payload=self.payload,
                    breakdown=self.breakdown,
                    metadata=self.metadata,
                    dev_info=self.dev_info,
                )
            )
            verbose_proxy_logger.debug("Lakera AI v2 guard request: %s", request)
            response = await self.async_handler.post(
                url=f"{self.api_base}/v2/guard",
                headers={"Authorization": f"Bearer {self.lakera_api_key}"},
                json=request,
            )
            verbose_proxy_logger.debug(
                "Lakera AI v2 guard response: %s", response.json()
            )
            lakera_response = LakeraAIResponse(**response.json())
            return lakera_response, masked_entity_count
        except Exception as e:
            status = "failure"
            exception_str = str(e)
            raise e
        finally:
            ####################################################
            # Create Guardrail Trace for logging on Langfuse, Datadog, etc.
            ####################################################
            guardrail_json_response: Union[Exception, str, dict, List[dict]] = {}
            if status == "success":
                copy_lakera_response_dict = (
                    dict(copy.deepcopy(lakera_response)) if lakera_response else {}
                )
                # payload contains PII, we don't want to log it
                copy_lakera_response_dict.pop("payload")
                guardrail_json_response = copy_lakera_response_dict
            else:
                guardrail_json_response = exception_str
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response=guardrail_json_response,
                guardrail_status=status,
                request_data=request_data,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
                masked_entity_count=masked_entity_count,
            )

    def _mask_pii_in_messages(
        self,
        messages: List[AllMessageValues],
        lakera_response: Optional[LakeraAIResponse],
        masked_entity_count: Dict,
    ) -> List[AllMessageValues]:
        """
        Return a copy of messages with any detected PII replaced by
        “[MASKED <TYPE>]” tokens.
        """
        payload = lakera_response.get("payload") if lakera_response else None
        if not payload:
            return messages

        # For each message, find its detections on the fly
        for idx, msg in enumerate(messages):
            content = msg.get("content", "")
            if not content:
                continue

            # For v1, we only support masking content strings
            if not isinstance(content, str):
                continue

            # Filter only detections for this message
            detected_modifications = [d for d in payload if d.get("message_id") == idx]
            if not detected_modifications:
                continue

            for modification in detected_modifications:
                start, end = modification.get("start", 0), modification.get("end", 0)

                # Extract the type (e.g. 'credit_card' → 'CREDIT_CARD')
                detector_type = modification.get("detector_type", "")
                if not detector_type:
                    continue

                typ = detector_type.split("/")[-1].upper() or "PII"
                mask = f"[MASKED {typ}]"
                if start is not None and end is not None:
                    content = self.mask_content_in_string(
                        content_string=content,
                        mask_string=mask,
                        start_index=start,
                        end_index=end,
                    )
                    masked_entity_count[typ] = masked_entity_count.get(typ, 0) + 1

            msg["content"] = content
        return messages

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
        ],
    ) -> Optional[Union[Exception, str, Dict]]:
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        verbose_proxy_logger.debug("Lakera AI: pre_call_hook")

        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                "Lakera AI: not running guardrail. Guardrail is disabled."
            )
            return data

        new_messages: Optional[List[AllMessageValues]] = data.get("messages")
        if new_messages is None:
            verbose_proxy_logger.warning(
                "Lakera AI: not running guardrail. No messages in data"
            )
            return data

        #########################################################
        ########## 1. Make the Lakera AI v2 guard API request ##########
        #########################################################
        lakera_guardrail_response, masked_entity_count = await self.call_v2_guard(
            messages=new_messages,
            request_data=data,
        )

        #########################################################
        ########## 2. Handle flagged content ##########
        #########################################################
        if lakera_guardrail_response.get("flagged") is True:
            # If only PII violations exist, mask the PII
            if self._is_only_pii_violation(lakera_guardrail_response):
                data["messages"] = self._mask_pii_in_messages(
                    messages=new_messages,
                    lakera_response=lakera_guardrail_response,
                    masked_entity_count=masked_entity_count,
                )
                verbose_proxy_logger.info(
                    "Lakera AI: Masked PII in messages instead of blocking request"
                )
            else:
                # If there are other violations or not set to mask PII, raise exception
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message="Lakera AI flagged this request. Please review the request and try again.",
                )

        #########################################################
        ########## 3. Add the guardrail to the applied guardrails header ##########
        #########################################################
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "responses",
        ],
    ):
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return

        new_messages: Optional[List[AllMessageValues]] = data.get("messages")
        if new_messages is None:
            verbose_proxy_logger.warning(
                "Lakera AI: not running guardrail. No messages in data"
            )
            return

        #########################################################
        ########## 1. Make the Lakera AI v2 guard API request ##########
        #########################################################
        lakera_guardrail_response, masked_entity_count = await self.call_v2_guard(
            messages=new_messages,
            request_data=data,
        )

        #########################################################
        ########## 2. Handle flagged content ##########
        #########################################################
        if lakera_guardrail_response.get("flagged") is True:
            # If only PII violations exist, mask the PII
            if self._is_only_pii_violation(lakera_guardrail_response):
                data["messages"] = self._mask_pii_in_messages(
                    messages=new_messages,
                    lakera_response=lakera_guardrail_response,
                    masked_entity_count=masked_entity_count,
                )
                verbose_proxy_logger.info(
                    "Lakera AI: Masked PII in messages instead of blocking request"
                )
            else:
                # If there are other violations or not set to mask PII, raise exception
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message="Lakera AI flagged this request. Please review the request and try again.",
                )

        #########################################################
        ########## 3. Add the guardrail to the applied guardrails header ##########
        #########################################################
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return data

    def _is_only_pii_violation(
        self, lakera_response: Optional[LakeraAIResponse]
    ) -> bool:
        """
        Returns True if there are only PII violations in the response.
        """
        if not lakera_response:
            return False

        for item in lakera_response.get("payload", []) or []:
            detector_type = item.get("detector_type", "") or ""
            if not detector_type.startswith("pii/"):
                return False
        return True
