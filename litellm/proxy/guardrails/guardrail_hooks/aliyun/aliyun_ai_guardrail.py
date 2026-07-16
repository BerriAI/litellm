#!/usr/bin/env python3
"""
Aliyun AI Security Guardrail Integration for LiteLLM
阿里云AI安全护栏集成
This guardrail scans prompts and responses using the Aliyun AI Security Guardrail API to detect:
- Content moderation violations
- Sensitive data (PII)
- Prompt injection attacks
- Malicious URLs
Documentation: https://help.aliyun.com/document_detail/2875413.html
Credentials:
Configured in config.yaml (litellm_params), support os.environ/ references:
- access_key_id: Aliyun Access Key ID
- access_key_secret: Aliyun Access Key Secret
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncGenerator, Literal
from urllib.parse import quote

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth

from .base import AliyunGuardrailBase

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.llms.openai import AllMessageValues
    from litellm.types.mcp import MCPPostCallResponseObject
    from litellm.types.proxy.guardrails.guardrail_hooks.aliyun.aliyun_ai_guardrail import (
        AliyunAIGuardrailResponse,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
    from litellm.types.utils import EmbeddingResponse, ImageResponse, ModelResponse

# Constants
ENCODING = "UTF-8"
ISO8601_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
ALGORITHM = "HmacSHA1"

# Region to endpoint mapping
REGION_ENDPOINTS = {
    "cn-shanghai": "green-cip.cn-shanghai.aliyuncs.com",
    "cn-beijing": "green-cip.cn-beijing.aliyuncs.com",
    "cn-hangzhou": "green-cip.cn-hangzhou.aliyuncs.com",
    "cn-shenzhen": "green-cip.cn-shenzhen.aliyuncs.com",
    "cn-chengdu": "green-cip.cn-chengdu.aliyuncs.com",
    "ap-southeast-1": "green-cip.ap-southeast-1.aliyuncs.com",
    "eu-central-1": "green-cip.eu-central-1.aliyuncs.com",
}

# Service codes for domestic (China) regions
SERVICE_INPUT_DOMESTIC = "query_security_check_pro"
SERVICE_OUTPUT_DOMESTIC = "response_security_check_pro"

# Service codes for international regions
SERVICE_INPUT_INTERNATIONAL = "query_security_check_cb"
SERVICE_OUTPUT_INTERNATIONAL = "response_security_check_cb"


# Detection types
CONTENT_MODERATION_TYPE = "contentModeration"
PROMPT_ATTACK_TYPE = "promptAttack"
SENSITIVE_DATA_TYPE = "sensitiveData"
MALICIOUS_URL_TYPE = "maliciousUrl"
MODEL_HALLUCINATION_TYPE = "modelHallucination"
CUSTOM_LABEL_TYPE = "customLabel"


def level_to_int(risk_level: str) -> int:
    """
    Convert risk level string to integer for comparison.
    Higher value = higher risk.
    Supports both standard risk levels (none/low/medium/high)
    and sensitive data levels (S0/S1/S2/S3/S4).
    Mapping:
    - none/S0 = 0 (no risk)
    - low/S1 = 1 (low risk)
    - medium/S2 = 2 (medium risk)
    - high/S3/S4 = 3 (high risk)
    """
    level_lower = risk_level.lower() if risk_level else "none"
    level_map = {
        # Standard risk levels
        "none": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
        # Sensitive data levels (mapped to standard levels)
        "s0": 0,  # No risk
        "s1": 1,  # Low risk
        "s2": 2,  # Medium risk
        "s3": 3,  # High risk
        "s4": 3,  # High risk (highest sensitive level)
    }
    return level_map.get(level_lower, 0)


# Protection level thresholds
# If detected_level >= threshold, then block
PROTECTION_LEVEL_THRESHOLD = {
    "low": 1,  # High protection: block low, medium, high (threshold=1, block if >=1)
    "medium": 2,  # Medium protection: block medium, high (threshold=2, block if >=2)
    "high": 3,  # Low protection: block high only (threshold=3, block if >=3)
    "max": 99,  # Observation mode: never block (threshold very high)
}


class AliyunAIGuardrail(AliyunGuardrailBase, CustomGuardrail):
    """
    LiteLLM Built-in Guardrail for Aliyun AI Security Guardrail.
    This guardrail scans prompts and responses using the Aliyun AI Security Guardrail API to detect
    malicious content, injection attempts, sensitive data, and policy violations.
    Configuration:
        guardrail_name: Name of the guardrail instance
        access_key_id: Aliyun Access Key ID
        access_key_secret: Aliyun Access Key Secret
        region_id: Aliyun region ID (default: cn-shanghai)
        default_on: Whether to enable by default
    """

    def __init__(
        self,
        guardrail_name: str,
        access_key_id: str | None = None,
        access_key_secret: str | None = None,
        region_id: str | None = None,
        level: str | None = None,
        max_text_length: int | None = None,
        stream_window_size: int | None = None,
        stream_slide_step: int | None = None,
        stream_first_check_step: int | None = None,
        service_input: str | None = None,
        service_output: str | None = None,
        service_mcp: str | None = None,
        **kwargs,
    ):
        """
        Initialize Aliyun AI Guardrail handler.
        Credentials (access_key_id / access_key_secret) are passed in from config.yaml
        via the guardrail loader.
        Args:
            region_id: Aliyun region ID (default: cn-shanghai)
            level: Protection level for risk filtering
                - "low": High protection, block all risks (low, medium, high, S1+)
                - "medium": Medium protection, block medium and high risks (medium, high, S2+)
                - "high": Low protection, block only high risks (high, S3+)
                - "max": Observation mode, no blocking
            service_input: Service code for input detection (default: query_security_check_pro)
            service_output: Service code for output detection (default: response_security_check_pro)
            service_mcp: Service code for MCP tool call detection, used by both
                pre_mcp_call and post_mcp_call (default: query_security_check_pro)
        """
        super().__init__(
            guardrail_name=guardrail_name,
            **kwargs,
        )
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        # Credentials come from config.yaml (access_key_id / access_key_secret)
        self.access_key_id = access_key_id or ""
        self.access_key_secret = access_key_secret or ""
        # Read region from config parameter
        self.region_id = region_id or "cn-shanghai"
        # Validate required credentials
        if not self.access_key_id:
            raise ValueError(
                "Aliyun AI Guardrail: ak is required. Set access_key_id in config.yaml (supports os.environ/ reference)"
            )
        if not self.access_key_secret:
            raise ValueError(
                "Aliyun AI Guardrail: sk is required. "
                "Set access_key_secret in config.yaml (supports os.environ/ reference)"
            )
        # Protection level: low/medium/high/max
        self.level = level or "medium"
        if self.level not in PROTECTION_LEVEL_THRESHOLD:
            raise ValueError(
                f"Aliyun AI Guardrail: Invalid level '{self.level}'. "
                f"Valid values are: {list(PROTECTION_LEVEL_THRESHOLD.keys())}"
            )
        self.max_text_length = max_text_length or 2000
        # Get endpoint from region
        self.endpoint = REGION_ENDPOINTS.get(self.region_id, REGION_ENDPOINTS["cn-shanghai"])
        self.service_url = f"https://{self.endpoint}"
        # Service codes - configurable, defaults to domestic
        self.service_input = service_input or SERVICE_INPUT_DOMESTIC
        self.service_output = service_output or SERVICE_OUTPUT_DOMESTIC
        self.service_mcp = service_mcp or SERVICE_INPUT_DOMESTIC
        # Sliding window parameters for streaming output checks
        self.stream_window_size = stream_window_size or 500
        self.stream_slide_step = stream_slide_step or 300
        self.stream_first_check_step = stream_first_check_step or 50
        verbose_proxy_logger.info(
            f"Initialized Aliyun AI Security Guardrail: {guardrail_name}, "
            f"region: {self.region_id}, level: {self.level}, "
            f"service_input: {self.service_input}, service_output: {self.service_output}, "
            f"service_mcp: {self.service_mcp}"
        )

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.aliyun.aliyun_ai_guardrail import (
            AliyunAIGuardrailConfigModel,
        )

        return AliyunAIGuardrailConfigModel

    @staticmethod
    def _format_iso8601_date() -> str:
        """Format current timestamp in ISO8601 format."""
        return datetime.now(timezone.utc).strftime(ISO8601_DATE_FORMAT)

    @staticmethod
    def _percent_encode(value: str | None) -> str:
        """URL encode a value according to Aliyun signature requirements."""
        if value is None:
            return ""
        return quote(value.encode(ENCODING), safe="~").replace("+", "%20").replace("*", "%2A")

    def _create_signature(self, string_to_sign: str) -> str:
        """Create HMAC-SHA1 signature for API request."""
        secret = self.access_key_secret + "&"
        signature = hmac.new(
            secret.encode(ENCODING),
            string_to_sign.encode(ENCODING),
            hashlib.sha1,
        ).digest()
        return base64.b64encode(signature).decode(ENCODING)

    def _create_string_to_sign(self, http_method: str, parameters: dict[str, str]) -> str:
        """Create the string to sign for API request."""
        sorted_keys = sorted(parameters.keys())
        canonicalized_query_string = ""
        for key in sorted_keys:
            canonicalized_query_string += "&" + self._percent_encode(key) + "=" + self._percent_encode(parameters[key])
        string_to_sign = (
            http_method + "&" + self._percent_encode("/") + "&" + self._percent_encode(canonicalized_query_string[1:])
        )
        return string_to_sign

    def _split_text(self, text: str, max_length: int = 2000) -> list[str]:
        """
        Split text into segments of maximum length, trying to preserve sentence boundaries.
        Args:
            text: Text to split
            max_length: Maximum length of each segment
        Returns:
            List of text segments
        """
        segments = []
        while len(text) > max_length:
            chunk = text[:max_length]
            # Find the last sentence boundary
            match = None
            for pattern in [r"[。！？；:\.?!]+"]:
                matches = list(re.finditer(pattern, chunk))
                if matches:
                    match = matches[-1]
            if match:
                cut_point = match.end()
            else:
                cut_point = max_length
            segments.append(text[:cut_point])
            text = text[cut_point:]
        if text:
            segments.append(text)
        return segments

    async def async_make_request(
        self,
        text: str | None = None,
        service_type: Literal["input", "output", "mcp"] = "input",
        image_urls: list[str] | None = None,
    ) -> AliyunAIGuardrailResponse:
        """
        Make a request to the Aliyun AI Security Guardrail API.
        Args:
            text: Text to check (optional when only images are checked)
            service_type: "input" for query_security_check, "output" for response_security_check,
                "mcp" for MCP tool call check (uses service_mcp config)
            image_urls: Public image URLs to check (optional)
        Returns:
            AliyunAIGuardrailResponse
        """
        from litellm.types.proxy.guardrails.guardrail_hooks.aliyun.aliyun_ai_guardrail import (
            AliyunAIGuardrailResponse,
        )

        # Build request parameters
        if service_type == "mcp":
            service_code = self.service_mcp
        elif service_type == "input":
            service_code = self.service_input
        else:
            service_code = self.service_output
        # Build ServiceParameters dynamically from the content + image combination
        service_parameters: dict[str, Any] = {"requestFrom": "LiteLLM"}
        if text:
            service_parameters["content"] = text
        if image_urls:
            service_parameters["imageUrls"] = image_urls
        parameters = {
            "Action": "MultiModalGuard",
            "Version": "2022-03-02",
            "AccessKeyId": self.access_key_id,
            "Timestamp": self._format_iso8601_date(),
            "SignatureMethod": "HMAC-SHA1",
            "SignatureVersion": "1.0",
            "SignatureNonce": str(uuid.uuid4()),
            "Format": "JSON",
            "Service": service_code,
            "ServiceParameters": json.dumps(service_parameters, ensure_ascii=False),
        }
        # Create signature
        string_to_sign = self._create_string_to_sign("POST", parameters)
        signature = self._create_signature(string_to_sign)
        parameters["Signature"] = signature
        verbose_proxy_logger.debug(
            "Aliyun AI Guardrail request: service=%s, text_length=%d, image_count=%d",
            service_code,
            len(text) if text else 0,
            len(image_urls) if image_urls else 0,
        )
        # Send request
        response = await self.async_handler.post(
            url=self.service_url,
            data=parameters,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30.0,
        )
        body = response.json()
        verbose_proxy_logger.debug("Aliyun AI Guardrail response: %s", body)
        # Check HTTP status
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail={"error": f"Aliyun AI Guardrail request failed. Status: {response.status_code}, Body: {body}"},
            )
        # Check API response code
        if body.get("Code") != 200:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Aliyun AI Guardrail API error. Code: {body.get('Code')}, Message: {body.get('Message')}"
                },
            )
        return AliyunAIGuardrailResponse(
            RequestId=body.get("RequestId", ""),
            Code=body.get("Code", 0),
            Message=body.get("Message"),
            Data=body.get("Data"),
        )

    def _should_block_by_level(self, detected_level: str) -> bool:
        """
        Check if the detected risk level should trigger blocking based on protection level.
        Logic: If detected_level_int >= threshold_int, then should block.
        Args:
            detected_level: Risk level from API response (none/low/medium/high or S0/S1/S2/S3/S4)
        Returns:
            True if should block, False otherwise
        """
        threshold = PROTECTION_LEVEL_THRESHOLD.get(self.level, 99)
        detected_int = level_to_int(detected_level)
        return detected_int >= threshold

    def _parse_response_and_check(
        self,
        response: AliyunAIGuardrailResponse,
        check_type: Literal["input", "output"],
    ) -> dict[str, Any]:
        """
        Parse the guardrail response and check if content should be blocked.
        Blocking logic:
        Check if detected level >= threshold based on protection level setting
        Args:
            response: The API response
            check_type: "input" or "output"
        Returns:
            Dict with parsed results
        Raises:
            HTTPException if content should be blocked
        """
        data = response.get("Data", {})
        if not data:
            return {"flagged": False, "suggestion": "pass", "details": {}, "message": ""}
        final_suggestion = data.get("Suggestion", "pass")
        detail_list = data.get("Detail") or []
        # Collect all detection results
        details: dict[str, dict[str, Any]] = {}
        desensitization = ""
        should_block = False
        blocked_type = ""
        blocked_level = ""
        block_message = ""
        for detail in detail_list:
            detection_type = detail.get("Type", "")
            detected_level = detail.get("Level", "none")
            suggestion = detail.get("Suggestion", "pass")
            results = detail.get("Result", [])
            # Store detail info
            details[detection_type] = {
                "level": detected_level,
                "suggestion": suggestion,
                "results": results,
            }
            # Extract desensitization text for sensitive data
            if detection_type == SENSITIVE_DATA_TYPE and results:
                for result in results:
                    ext = result.get("Ext", {})
                    if ext and ext.get("Desensitization"):
                        desensitization = ext.get("Desensitization", "")
                        break
            # Check if this detection should trigger blocking
            # Detected level must meet threshold based on protection level
            if not should_block and self._should_block_by_level(detected_level):
                should_block = True
                blocked_type = detection_type
                blocked_level = detected_level
                # Use the raw detection type returned by Aliyun as-is
                block_message = f"检测到{detection_type} (风险等级: {detected_level})"
        verbose_proxy_logger.debug(
            f"Aliyun AI Guardrail: level={self.level}, "
            f"check_type={check_type}, should_block={should_block}, "
            f"blocked_type={blocked_type}, blocked_level={blocked_level}"
        )
        if should_block:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Aliyun AI Guardrail: {block_message}",
                    "type": check_type,
                    "details": details,
                },
            )
        return {
            "flagged": final_suggestion != "pass",
            "suggestion": final_suggestion,
            "desensitization": desensitization,
            "details": details,
            "message": block_message,
        }

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: dict[str, Any],
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
            "responses",
        ],
    ) -> dict[str, Any] | None:
        """
        Pre-call hook to scan user prompts before sending to LLM.
        Raises HTTPException if content should be blocked.
        """
        verbose_proxy_logger.info(
            "Aliyun AI Guardrail: Running pre-call prompt scan, on call_type: %s",
            call_type,
        )
        # MCP tool call — use dedicated MCP check logic
        if call_type == "call_mcp_tool":
            return await self._mcp_pre_call_check(data)
        new_messages: list[AllMessageValues] | None = data.get("messages")
        if new_messages is None:
            verbose_proxy_logger.warning("Aliyun AI Guardrail: not running guardrail. No messages in data")
            return data
        user_prompt = self.get_user_prompt(new_messages)
        image_urls = self.get_image_urls(new_messages)
        if not user_prompt and not image_urls:
            verbose_proxy_logger.warning("Aliyun AI Guardrail: No user prompt or image found")
            return None
        verbose_proxy_logger.info(
            "Aliyun AI Guardrail: Pre-call scan started, prompt length: %d, image count: %d",
            len(user_prompt) if user_prompt else 0,
            len(image_urls),
        )
        # Split text if too long
        if user_prompt:
            if len(user_prompt) > self.max_text_length:
                segments = self._split_text(user_prompt, self.max_text_length)
            else:
                segments = [user_prompt]
        else:
            segments = []
        # Build request payloads as (text, image_urls) tuples.
        # Attach all image URLs to the first text segment so content + images
        # are checked together; remaining segments carry text only. When there
        # is no text, send a single image-only request.
        payloads: list[tuple] = []
        if segments:
            for idx, segment in enumerate(segments):
                payloads.append((segment, image_urls if idx == 0 and image_urls else None))
        elif image_urls:
            payloads.append((None, image_urls))
        # Check all payloads concurrently with limited concurrency
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests, MultiModalGuard API limit is 20

        async def check_with_semaphore(segment_text: str | None, segment_images: list[str] | None):
            async with semaphore:
                return await self.async_make_request(text=segment_text, service_type="input", image_urls=segment_images)

        responses = await asyncio.gather(*[check_with_semaphore(t, imgs) for t, imgs in payloads])
        for response in responses:
            self._parse_response_and_check(response, check_type="input")
        verbose_proxy_logger.info("Aliyun AI Guardrail: Pre-call scan passed")
        return None

    # ================================================================
    # MCP-specific guardrail methods
    # ================================================================

    async def _mcp_pre_call_check(self, data: dict) -> dict | None:
        """MCP pre-call: audit tool name + arguments before execution."""
        messages = data.get("messages", [])
        content = messages[0].get("content", "") if messages else ""
        verbose_proxy_logger.info(
            "Aliyun AI Guardrail: ★ MCP pre-call check started, content: %s",
            content,
        )
        if not content:
            return None
        if len(content) > self.max_text_length:
            segments = self._split_text(content, self.max_text_length)
        else:
            segments = [content]
        semaphore = asyncio.Semaphore(5)

        async def check(text: str):
            async with semaphore:
                return await self.async_make_request(text=text, service_type="mcp")

        responses = await asyncio.gather(*[check(s) for s in segments])
        for resp in responses:
            self._parse_response_and_check(resp, check_type="input")
        verbose_proxy_logger.info("Aliyun AI Guardrail: ★ MCP pre-call check passed")
        return None

    def _should_run_post_mcp_call(self) -> bool:
        """Check if post_mcp_call is configured in event_hook.

        Since GuardrailEventHooks enum does not include post_mcp_call,
        we cannot use should_run_guardrail(). This method manually checks
        whether the user configured 'post_mcp_call' in the guardrail mode.

        Returns True if post_mcp_call should run:
        - event_hook is None → run for all events
        - event_hook is a list containing 'post_mcp_call'
        - event_hook is a string equal to 'post_mcp_call'
        - event_hook is a Mode with 'post_mcp_call' in tags or default
        """
        from litellm.types.guardrails import Mode

        if self.event_hook is None:
            return True
        if isinstance(self.event_hook, list):
            return "post_mcp_call" in self.event_hook
        if isinstance(self.event_hook, Mode):
            for tag_value in self.event_hook.tags.values():
                if isinstance(tag_value, list):
                    if "post_mcp_call" in tag_value:
                        return True
                elif tag_value == "post_mcp_call":
                    return True
            if self.event_hook.default:
                default_list = (
                    self.event_hook.default if isinstance(self.event_hook.default, list) else [self.event_hook.default]
                )
                return "post_mcp_call" in default_list
            return False
        return self.event_hook == "post_mcp_call"

    async def async_post_mcp_tool_call_hook(
        self,
        kwargs: dict,
        response_obj: MCPPostCallResponseObject,
        start_time: datetime,
        end_time: datetime,
    ) -> MCPPostCallResponseObject | None:
        """MCP post-call: audit tool output after execution."""
        # Since GuardrailEventHooks enum has no post_mcp_call, the framework
        # always invokes this hook if implemented. We check config manually.
        if not self._should_run_post_mcp_call():
            verbose_proxy_logger.info("Aliyun AI Guardrail: Skipping post_mcp_call — not configured in event_hook")
            return None
        verbose_proxy_logger.info("Aliyun AI Guardrail: ★ MCP post-call check started")
        mcp_response = getattr(response_obj, "mcp_tool_call_response", None)
        if not mcp_response:
            return None
        texts: list[str] = []
        for item in mcp_response:
            if isinstance(item, dict):
                text = item.get("text", "")
            elif hasattr(item, "text"):
                text = item.text
            else:
                text = str(item)
            if text:
                texts.append(text)
        if not texts:
            return None
        combined_text = "\n".join(texts)
        verbose_proxy_logger.info(
            "Aliyun AI Guardrail: ★ MCP post-call check, response length: %d",
            len(combined_text),
        )
        if len(combined_text) > self.max_text_length:
            segments = self._split_text(combined_text, self.max_text_length)
        else:
            segments = [combined_text]
        semaphore = asyncio.Semaphore(5)

        async def check(text: str):
            async with semaphore:
                return await self.async_make_request(text=text, service_type="mcp")

        responses = await asyncio.gather(*[check(s) for s in segments])
        for resp in responses:
            self._parse_response_and_check(resp, check_type="output")
        verbose_proxy_logger.info("Aliyun AI Guardrail: ★ MCP post-call check passed")
        return None

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any | ModelResponse | EmbeddingResponse | ImageResponse,
    ) -> Any:
        """
        Post-call hook to scan LLM responses.
        Raises HTTPException if content should be blocked.
        """
        from litellm.types.utils import Choices, ModelResponse

        if isinstance(response, ModelResponse) and response.choices and isinstance(response.choices[0], Choices):
            content = response.choices[0].message.content or ""
            if content:
                verbose_proxy_logger.info(
                    "Aliyun AI Guardrail: Post-call scan started, response length: %d",
                    len(content),
                )
                # Split text if too long
                if len(content) > self.max_text_length:
                    segments = self._split_text(content, self.max_text_length)
                else:
                    segments = [content]
                # Check all segments concurrently with limited concurrency
                semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

                async def check_with_semaphore(segment: str):
                    async with semaphore:
                        return await self.async_make_request(text=segment, service_type="output")

                responses = await asyncio.gather(*[check_with_semaphore(segment) for segment in segments])
                for guardrail_response in responses:
                    self._parse_response_and_check(guardrail_response, check_type="output")
                verbose_proxy_logger.info("Aliyun AI Guardrail: Post-call scan passed")
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        """
        Process streaming response with sliding window guardrail checks.
        This method implements sliding window guardrail checks based on
        `stream_window_size` and `stream_slide_step`.
        It triggers a guardrail API call when:
        1. Every `stream_slide_step` new chars accumulate since the last check.
        2. Stream ends and there's remaining unchecked content.
        For example, if stream_window_size=2000, stream_slide_step=300:
        - At 300 chars: check chars 0-300 (window: last 2000)
        - At 600 chars: check chars 0-600 (window: last 2000)
        - At 2100 chars: check chars 100-2100 (window slides forward)
        - At 2400 chars: check chars 400-2400 (window slides forward)
        - When stream ends with 2500 chars: check chars 500-2500 (final window)
        """
        import litellm
        from litellm.types.utils import ModelResponseStream

        accumulated_text = ""
        last_check_position = 0  # Position (total length) when last check was triggered
        pending_chunks = []  # Buffer chunks until guardrail check passes
        chunk_count = 0
        is_first_check = True  # First check uses smaller threshold to reduce first-token latency
        verbose_proxy_logger.info(
            "Aliyun AI Guardrail: Streaming scan started, window=%d, step=%d, first_check_step=%d",
            self.stream_window_size,
            self.stream_slide_step,
            self.stream_first_check_step,
        )
        try:
            async for chunk in response:
                # Extract text from chunk
                chunk_text = ""
                if isinstance(chunk, ModelResponseStream):
                    chunk_text = litellm.get_response_string(response_obj=chunk) or ""
                    accumulated_text += chunk_text
                elif hasattr(chunk, "choices") and chunk.choices:
                    delta = getattr(chunk.choices[0], "delta", None)
                    if delta and hasattr(delta, "content") and delta.content:
                        chunk_text = delta.content
                        accumulated_text += delta.content
                chunk_count += 1
                # Buffer the chunk, don't yield until guardrail check passes
                pending_chunks.append(chunk)
                current_length = len(accumulated_text)
                # Sliding window: first check uses stream_first_check_step, subsequent checks use stream_slide_step
                new_chars_since_last_check = current_length - last_check_position
                check_threshold = self.stream_first_check_step if is_first_check else self.stream_slide_step
                if new_chars_since_last_check >= check_threshold:
                    # Send the most recent stream_window_size chars to the API
                    start = max(0, current_length - self.stream_window_size)
                    text_to_check = accumulated_text[start:current_length]
                    guardrail_response = await self.async_make_request(text=text_to_check, service_type="output")
                    self._parse_response_and_check(guardrail_response, check_type="output")
                    verbose_proxy_logger.info(
                        "Aliyun AI Guardrail: Streaming check passed at position %d", current_length
                    )
                    # Check passed - release all buffered chunks to the client
                    for pending_chunk in pending_chunks:
                        yield pending_chunk
                    pending_chunks.clear()
                    last_check_position = current_length
                    is_first_check = False
            # Stream ended - check any remaining unchecked content with a final window
            if len(accumulated_text) > last_check_position:
                start = max(0, len(accumulated_text) - self.stream_window_size)
                remaining_text = accumulated_text[start:]
                guardrail_response = await self.async_make_request(text=remaining_text, service_type="output")
                self._parse_response_and_check(guardrail_response, check_type="output")
            verbose_proxy_logger.info(
                "Aliyun AI Guardrail: Streaming scan completed, total length: %d", len(accumulated_text)
            )
            # Release any remaining buffered chunks
            for pending_chunk in pending_chunks:
                yield pending_chunk
            pending_chunks.clear()
        except HTTPException as e:
            # Yield an SSE error event to the client before ending the stream
            error_detail = e.detail if isinstance(e.detail, dict) else {"message": str(e.detail)}
            verbose_proxy_logger.info("Aliyun AI Guardrail: Streaming blocked at position %d", len(accumulated_text))
            yield f"data: {json.dumps({'error': error_detail})}\n\n"
            return
        except Exception as e:
            verbose_proxy_logger.error(f"Aliyun AI Guardrail streaming error: {str(e)}", exc_info=True)
            raise
