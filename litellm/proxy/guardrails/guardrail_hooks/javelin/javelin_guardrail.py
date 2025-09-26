# +-------------------------------------------------------------+
#
#           Use Javelin guardrails for your LLM calls
#
# +-------------------------------------------------------------+

import json
import os
import sys
from typing import Dict, List, Literal, Optional, Union

import httpx
from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client, httpxSpecialProvider
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_helpers import should_proceed_based_on_metadata
from litellm.secret_managers.main import get_secret
from litellm.types.guardrails import GuardrailEventHooks, Role, default_roles

GUARDRAIL_NAME = "javelin"

class JavelinGuardrail(CustomGuardrail):
    """
    Javelin Guardrail implementation for LiteLLM
    
    Integrates with Javelin's standalone guardrails API for:
    - Prompt injection detection
    - Trust & safety content filtering
    - Language detection
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        application_name: Optional[str] = None,
        guardrail_processor: str = "promptinjectiondetection",  # or "trustsafety", "lang_detector"
        **kwargs,
    ):
        """
        Initialize Javelin Guardrail
        
        Args:
            api_key: Javelin API key (can also be set via JAVELIN_API_KEY env var)
            api_base: Javelin API base URL (should include your domain)
            application_name: Application name for policy-specific rules
            guardrail_processor: Which processor to use ('promptinjectiondetection', 'trustsafety', 'lang_detector')
        """
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.javelin_api_key = api_key or get_secret("JAVELIN_API_KEY") or os.environ.get("JAVELIN_API_KEY")
        self.api_base = api_base or get_secret("JAVELIN_API_BASE") or os.environ.get("JAVELIN_API_BASE")
        self.application_name = application_name or get_secret("JAVELIN_APPLICATION_NAME") or os.environ.get("JAVELIN_APPLICATION_NAME")
        self.guardrail_processor = guardrail_processor
        
        if not self.javelin_api_key:
            raise ValueError("Javelin API key is required. Set JAVELIN_API_KEY environment variable or pass api_key parameter.")
            
        if not self.api_base:
            raise ValueError("Javelin API base URL is required. Set JAVELIN_API_BASE environment variable or pass api_base parameter.")
            
        # Remove trailing slash if present
        if isinstance(self.api_base, str):
            self.api_base = self.api_base.rstrip('/')
        
        super().__init__(
            guardrail_name=GUARDRAIL_NAME,
            supported_event_hooks=[
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
                GuardrailEventHooks.post_call,
            ],
            **kwargs
        )

    def _should_reject_request(self, response: dict) -> bool:
        """
        Determine if request should be rejected based on Javelin response
        
        Args:
            response: Response from Javelin API
            
        Returns:
            bool: True if request should be rejected
        """
        assessments = response.get("assessments", [])
        if not assessments:
            return False
            
        for assessment in assessments:
            # Check each processor type in the assessment
            for processor_name, processor_data in assessment.items():
                request_reject = processor_data.get("request_reject", False)
                if request_reject:
                    verbose_proxy_logger.warning(
                        f"Javelin guardrail {processor_name} flagged content for rejection: {processor_data}"
                    )
                    return True
        
        return False

    def _extract_text_from_data(self, data: dict) -> Optional[str]:
        """Extract text content from request data for analysis"""
        text_content = []
        
        # Handle chat completion messages
        if "messages" in data and isinstance(data["messages"], list):
            for message in data["messages"]:
                content = message.get("content")
                if content:
                    if isinstance(content, str):
                        text_content.append(content)
                    elif isinstance(content, list):
                        # Handle content with mixed types (text, images, etc.)
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_content.append(item.get("text", ""))
                            elif isinstance(item, str):
                                text_content.append(item)
        
        # Handle text completion
        elif "input" in data:
            input_data = data["input"]
            if isinstance(input_data, str):
                text_content.append(input_data)
            elif isinstance(input_data, list):
                text_content.extend([str(item) for item in input_data])
        
        # Handle prompt field
        elif "prompt" in data:
            prompt_data = data["prompt"]
            if isinstance(prompt_data, str):
                text_content.append(prompt_data)
            elif isinstance(prompt_data, list):
                text_content.extend([str(item) for item in prompt_data])
        
        return "\n".join(text_content) if text_content else None

    async def _call_javelin_guardrail(self, text: str, data: dict) -> dict:
        """
        Call Javelin guardrail API
        
        Args:
            text: Text content to analyze
            data: Original request data for context
            
        Returns:
            dict: Response from Javelin API
        """
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "x-javelin-apikey": self.javelin_api_key,
        }
        
        # Add application name header if specified
        if self.application_name:
            headers["x-javelin-application"] = self.application_name
        
        # Prepare request payload
        payload = {
            "input": {
                "text": text
            }
        }
        
        # Add any dynamic parameters from guardrail config
        extra_body = self.get_guardrail_dynamic_request_body_params(request_data=data)
        if extra_body:
            payload.update(extra_body)
        
        url = f"{self.api_base}/v1/guardrail/{self.guardrail_processor}/apply"
        
        verbose_proxy_logger.debug(
            f"Calling Javelin guardrail at {url} with payload: {json.dumps(payload, indent=2)}"
        )
        
        try:
            response = await self.async_handler.post(
                url=url,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            
            response_data = response.json()
            verbose_proxy_logger.debug(f"Javelin guardrail response: {json.dumps(response_data, indent=2)}")
            
            return response_data
            
        except httpx.HTTPStatusError as e:
            error_msg = f"Javelin guardrail API error: {e.response.status_code} - {e.response.text}"
            verbose_proxy_logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Javelin guardrail API error",
                    "details": error_msg
                }
            )
        except Exception as e:
            error_msg = f"Error calling Javelin guardrail: {str(e)}"
            verbose_proxy_logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Javelin guardrail error",
                    "details": error_msg
                }
            )

    async def _analyze_content(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "text_completion", 
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "responses",
            "mcp_call",
        ],
    ) -> None:
        """
        Core analysis method that extracts content and calls Javelin API
        """
        # Check if we should proceed based on metadata
        if (
            await should_proceed_based_on_metadata(
                data=data,
                guardrail_name=GUARDRAIL_NAME,
            )
            is False
        ):
            return

        # Extract text content from request
        text_content = self._extract_text_from_data(data)
        
        if not text_content or not text_content.strip():
            verbose_proxy_logger.debug("No text content found to analyze, skipping Javelin guardrail")
            return
            
        verbose_proxy_logger.debug(f"Analyzing content with Javelin {self.guardrail_processor} processor")
        
        # Call Javelin API
        response = await self._call_javelin_guardrail(text_content, data)
        
        # Check if content should be rejected
        if self._should_reject_request(response):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Content violated {self.guardrail_processor} policy",
                    "javelin_response": response
                }
            )

    @log_guardrail_information
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
        ],
    ) -> Optional[Union[Exception, str, Dict]]:
        """Pre-call hook for analyzing input before sending to LLM"""
        if not self._should_run_for_event(GuardrailEventHooks.pre_call, data):
            return None
            
        await self._analyze_content(data, user_api_key_dict, call_type)
        return None

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
            "responses",
            "mcp_call",
        ],
    ):
        """During-call moderation hook"""
        if not self._should_run_for_event(GuardrailEventHooks.during_call, data):
            return None
            
        await self._analyze_content(data, user_api_key_dict, call_type)
        return None

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: litellm.ModelResponse,
    ) -> Optional[litellm.ModelResponse]:
        """Post-call hook for analyzing LLM response"""
        if not self._should_run_for_event(GuardrailEventHooks.post_call, data):
            return response
            
        # Extract response content for analysis
        response_content = []
        
        if hasattr(response, 'choices') and response.choices:
            for choice in response.choices:
                if hasattr(choice, 'message') and choice.message:
                    content = choice.message.content
                    if content:
                        response_content.append(content)
                elif hasattr(choice, 'text'):
                    if choice.text:
                        response_content.append(choice.text)
        
        if response_content:
            # Create a temporary data structure for response analysis
            response_data = {
                "input": "\n".join(response_content)
            }
            
            # Use the same analysis method
            await self._analyze_content(response_data, user_api_key_dict, "responses")
        
        return response

    def _should_run_for_event(self, event_type: GuardrailEventHooks, data: dict) -> bool:
        """Check if guardrail should run for this event type"""
        from litellm.types.guardrails import GuardrailEventHooks as GEH
        
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return False
            
        return True
