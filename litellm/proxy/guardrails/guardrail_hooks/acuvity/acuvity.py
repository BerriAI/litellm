import os
from enum import Enum
from typing import List, Literal, Optional, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.utils import EmbeddingResponse, ImageResponse, ModelResponse, StreamingChoices

GUARDRAIL_NAME = "acuvity"
from dataclasses import dataclass, field
from typing import Dict, Set

from pydantic import BaseModel

from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

from .helper import Extraction, GuardName, ResponseHelper, get_apex_url_from_token


@dataclass
class GuardrailConfig:
    name: str
    threshold: Optional[str] = None
    matches: Optional[Dict] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'GuardrailConfig':
        name = data['name']
        if name and str(name).upper() not in GuardName:
            raise ValueError(f"invalid guard name {name}, must be either of {GuardName}")
        threshold_val = data.get('threshold', 0)
        if 0 < threshold_val > 1:
            raise ValueError(f"invalid threshold val {threshold_val}, must be between 0 and 1")
        return cls(
            name=name,
            threshold=data.get('threshold'),
            matches=data.get('matches')
        )

@dataclass
class Config:
    guardrails: List[GuardrailConfig] = field(default_factory=list)
    _redaction_keys: Set[str] = field(default_factory=set, init=False)

    def __init__(self, guardrails: List[GuardrailConfig]):
        self.guardrails = guardrails
        self._redaction_keys = set()

    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        return cls(
            guardrails=[GuardrailConfig.from_dict(g) for g in data.get('guardrails', [])]
        )

    @property
    def redaction_keys(self) -> set:
        redact_keys = set()
        for g in self.guardrails:
            if g.matches:
                for key, val in g.matches.items():
                    if isinstance(val, dict) and val.get('redact', False):
                        redact_keys.add(key)
        return redact_keys

# Define the scan request type
class ScanRequest(BaseModel):
    anonymization: Literal["FixedSize"]
    messages: List[str]
    redactions: List[str]
    type: Literal["Input", "Output"]

# Define the event type enum
class HookEventType(Enum):
    BEFORE_REQUEST = "beforeRequestHook"
    AFTER_REQUEST = "afterRequestHook"



class AcuvityGuardrail(CustomGuardrail):
    def __init__(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None,
        **kwargs,
    ):
        try:
            self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
            self.api_key = api_key or os.environ.get("ACUVITY_TOKEN")
            self.api_base = api_base or get_apex_url_from_token(self.api_key)

            if kwargs.get("event_hook") == "pre_call":
                self.pre_guard_config_dict = kwargs.pop("vendor_params", None)  # None as default if not found
                self.pre_guard_config = Config.from_dict(self.pre_guard_config_dict)
            if kwargs.get("event_hook") == "during_call":
                self.during_guard_config_dict = kwargs.pop("vendor_params", None)  # None as default if not found
                self.during_guard_config = Config.from_dict(self.during_guard_config_dict)
                if self.during_guard_config.redaction_keys:
                    raise ValueError("acuvity guard config cannot do redaction for during_call mode, please add redactions in the pre_call")
            if kwargs.get("event_hook") == "post_call":
                self.post_guard_config_dict = kwargs.pop("vendor_params", None)  # None as default if not found
                self.post_guard_config = Config.from_dict(self.post_guard_config_dict)
        except Exception as e:
            raise ValueError("Acuvity guard config cannot be parsed") from e

        super().__init__(**kwargs)

    async def post_acuvity_scan(
            self,
            base_url: str,
            api_key: str,
            text_array: List[str],
            event_type: HookEventType,
            redactions: List[str]
    ) -> dict:
        """
        Post a scan request to the Acuvity API

        Args:
            base_url: The base URL for the API
            api_key: The API key for authentication
            text_array: Array of texts to scan
            event_type: Type of hook event
            redactions: List of redactions to apply

        Returns:
            The API response as a dictionary
        """
        data = ScanRequest(
            anonymization="FixedSize",
            messages=text_array,
            redactions=redactions,
            type="Input" if event_type == HookEventType.BEFORE_REQUEST else "Output"
        )

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        response = await self.async_handler.post(
            f"{base_url}/_acuvity/scan",
            data=data.json(),  # Convert class to  JSON for serialization
            headers=headers
        )

        # Raise an exception for bad status codes
        response.raise_for_status()

        return response.json()

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
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
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Runs on request before the LLM API call

        Returns:
            The processed request, potentially with redacted content

        Raises:
            HTTPException: If guardrail policies are violated
        """
        redacted = False
        _messages = data.get("messages")
        if _messages:
            msgs = [message.get("content") for message in _messages if message.get("content") is not None]

            verbose_proxy_logger.debug("Acuvity pre_call data: %s", msgs)
            resp = await self.post_acuvity_scan(self.api_base, self.api_key ,msgs, HookEventType.BEFORE_REQUEST, list(self.pre_guard_config.redaction_keys))

            # Initialize guard results set
            guard_results = set()
            response_helper = ResponseHelper()

            # Process each extraction
            for index, extraction_dict in enumerate(resp.get('extractions', [])):
                # Evaluate config for current extraction
                extraction_obj = Extraction(**extraction_dict)
                current_results, detected_pii_extraction_vals = self.evaluate_all_guardrails(
                    extraction_obj,
                    self.pre_guard_config,
                    response_helper
                )
                # Update the main set with current results
                guard_results.update(current_results)

                if isinstance(_messages[index]["content"], str):
                    if _messages[index]["content"] != extraction_obj.data:
                        redacted = True
                        _messages[index]["content"] = extraction_obj.data

            data["messages"] = _messages
            if self._check_violations(guard_results, detected_pii_extraction_vals):
                raise HTTPException(
                    status_code=400, detail={
                        "error": "Violated guardrail policy",
                        "guard": guard_results
                    }
                )

            verbose_proxy_logger.info(
                f"Acuvity pre call processed message: {data['messages']}, redaction applied {redacted}"
            )
        return data

    @log_guardrail_information
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
        ],
    ):
        """
        For any triggered guards we raise exceptions.
        """
        _messages = data.get("messages")
        if _messages:
            msgs = [message.get("content") for message in _messages if message.get("content") is not None]

            verbose_proxy_logger.debug("Acuvity during_call data: %s", msgs)
            resp = await self.post_acuvity_scan(self.api_base, self.api_key ,msgs, HookEventType.BEFORE_REQUEST, list(self.during_guard_config.redaction_keys))

            # Initialize guard results set
            guard_results = set()
            response_helper = ResponseHelper()

            if resp.get('extractions'):
                    for index, extraction in enumerate(resp.get('extractions')):
                        extraction_obj = Extraction(**extraction)
                        current_results, _ = self.evaluate_all_guardrails(
                                                extraction_obj,
                                                self.during_guard_config,
                                                response_helper
                                            )
                        # Update the main set with current results
                        guard_results.update(current_results)

                    if len(guard_results) > 0:
                        raise HTTPException(
                            status_code=400, detail={
                                "error": "Violated guardrail policy",
                                "guard": guard_results
                            }
                        )
        pass

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[ModelResponse, EmbeddingResponse, ImageResponse],
    ):
        """
        Runs on response from LLM API call

        Returns:
            The processed response, potentially with redacted content

        Raises:
            HTTPException: If guardrail policies are violated
        """

        # we are intrested/supported only in text msgs and non streaming.
        _messages = self._extract_response_messages(response)
        redacted = False
        if _messages:
            verbose_proxy_logger.debug("Acuvity post_call data: %s", _messages)
            resp = await self.post_acuvity_scan(self.api_base,self.api_key , [_messages] , HookEventType.AFTER_REQUEST, list(self.post_guard_config.redaction_keys))

            # Initialize guard results set
            guard_results = set()
            response_helper = ResponseHelper()

            # Process each extraction
            for index, extraction in enumerate(resp.get('extractions', [])):
                # Evaluate parameters for current extraction
                extraction_obj = Extraction(**extraction)
                current_results, detected_pii_extraction_vals = self.evaluate_all_guardrails(
                    extraction_obj,
                    self.post_guard_config,
                    response_helper
                )
                # Update the main set with current results
                guard_results.update(current_results)

                #If no violations then, check if we need to replace the redacted response.
                if isinstance(response, ModelResponse) and not isinstance(response.choices[0], StreamingChoices):
                    if response.choices[0].message.content != extraction_obj.data:
                        redacted = True
                        response.choices[0].message.content = extraction_obj.data

            if self._check_violations(guard_results, detected_pii_extraction_vals):
                raise HTTPException(
                    status_code=400, detail={
                        "error": "Violated guardrail policy",
                        "guard": guard_results
                    }
                )
            verbose_proxy_logger.info(
                    f"Acuvity post call processed message: {response.choices[0]}, redaction applied {redacted}"
                )
        return response

    def _extract_response_messages(self, response_obj: Union[ModelResponse, EmbeddingResponse, ImageResponse]) -> Optional[str]:
        """
        Extracts the first non-streaming text message from a response object.

        Args:
            response_obj: The response object to process

        Returns:
            The first message content if available, None otherwise
        """
        if not isinstance(response_obj, ModelResponse):
            return None

        for choice in response_obj.choices:
            if (not isinstance(choice, StreamingChoices) and
                choice.message.content and
                isinstance(choice.message.content, str)):
                return choice.message.content
        return None

    def _check_violations(self, guard_results: set, detected_pii_extraction_vals: set) -> bool:
        # here we need to raise exception if
        # 1. If we only find PII, secrets guards
        if len(guard_results) == 1 and GuardName.PII_DETECTOR in guard_results:
            # then we check if any of the matches are only detection.
            if len([item for item in detected_pii_extraction_vals if item not in self.pre_guard_config.redaction_keys]) > 0:
                return True
        # 2. we see there more detections than PIIs, if it was only PII, secrets the guard_result len will be 1.
        elif len(guard_results) > 0:
            return True
        return False

    def evaluate_all_guardrails(
        self,
        extraction: Extraction,
        config: Config,
        response_helper: ResponseHelper
    ) -> Set[GuardName]:
        """
        Evaluate all guardrails for a given extraction
        """
        guard_types = set()
        detected_pii_extraction_vals = set()

        for guardrail in config.guardrails:
            try:
                guard_name = GuardName[guardrail.name.upper()]
                # default the threshold to 0.
                threshold = float(guardrail.threshold) if guardrail.threshold else 0.0

                if guardrail.matches:
                    # For guards with matches (like PII)
                    for match_name in guardrail.matches.keys():
                        result = response_helper.evaluate(
                            extraction,
                            guard_name,
                            threshold,
                            match_name
                        )
                        if result.matched:
                            if guard_name in [ GuardName.SECRETS_DETECTOR, GuardName.PII_DETECTOR]:
                                guard_types.add(GuardName.PII_DETECTOR)
                            else:
                                guard_types.add(guard_name)
                            detected_pii_extraction_vals.update(set(result.match_values))
                else:
                    # For guards without matches (like prompt_injection)
                    result = response_helper.evaluate(
                        extraction,
                        guard_name,
                        threshold
                    )
                    if result.matched:
                        guard_types.add(guard_name)

            except KeyError:
                verbose_proxy_logger.error("invalid guard name passed in guard config")
                continue  # Skip if guard name is not valid

        return guard_types, detected_pii_extraction_vals
