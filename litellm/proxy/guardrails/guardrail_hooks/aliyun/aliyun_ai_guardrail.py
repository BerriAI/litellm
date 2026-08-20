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
from collections.abc import AsyncGenerator, AsyncIterable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType

# `Any` is needed for the heterogeneous LLM payloads this guardrail walks (request bodies,
# streaming chunks, MCP tool results); every field it actually reads is probed defensively.
# `cast` re-labels the upstream message iterator, whose element type is untyped upstream.
from typing import TYPE_CHECKING, Any, Final, Literal, cast  # noqa: TID251  # see comment above
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

# Only the private iterator walks `messages` *and* the Responses API's `input` while keeping
# multimodal parts intact; the public helpers flatten to text, which would drop image URLs.
from litellm.proxy.guardrails._content_utils import (
    _iter_inspection_messages,  # pyright: ignore[reportPrivateUsage]  # see comment above
)
from litellm.types.utils import CallTypes

from .base import AliyunGuardrailBase

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.llms.openai import AllMessageValues
    from litellm.types.mcp import MCPPostCallResponseObject
    from litellm.types.proxy.guardrails.guardrail_hooks.aliyun.aliyun_ai_guardrail import (
        AliyunAIGuardrailResponse,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
    from litellm.types.utils import CallTypesLiteral, LLMResponseTypes

# Constants
ENCODING: Final = "UTF-8"
ISO8601_DATE_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"
ALGORITHM: Final = "HmacSHA1"

# Region to endpoint mapping
REGION_ENDPOINTS: Final = MappingProxyType(
    {
        "cn-shanghai": "green-cip.cn-shanghai.aliyuncs.com",
        "cn-beijing": "green-cip.cn-beijing.aliyuncs.com",
        "cn-hangzhou": "green-cip.cn-hangzhou.aliyuncs.com",
        "cn-shenzhen": "green-cip.cn-shenzhen.aliyuncs.com",
        "cn-chengdu": "green-cip.cn-chengdu.aliyuncs.com",
        "ap-southeast-1": "green-cip.ap-southeast-1.aliyuncs.com",
        "eu-central-1": "green-cip.eu-central-1.aliyuncs.com",
    }
)

# Service codes for domestic (China) regions
SERVICE_INPUT_DOMESTIC: Final = "query_security_check_pro"
SERVICE_OUTPUT_DOMESTIC: Final = "response_security_check_pro"

# Service codes for international regions
SERVICE_INPUT_INTERNATIONAL: Final = "query_security_check_cb"
SERVICE_OUTPUT_INTERNATIONAL: Final = "response_security_check_cb"


# Detection types
CONTENT_MODERATION_TYPE: Final = "contentModeration"
PROMPT_ATTACK_TYPE: Final = "promptAttack"
SENSITIVE_DATA_TYPE: Final = "sensitiveData"
MALICIOUS_URL_TYPE: Final = "maliciousUrl"
MODEL_HALLUCINATION_TYPE: Final = "modelHallucination"
CUSTOM_LABEL_TYPE: Final = "customLabel"

# Suggestion returned by Aliyun when it has decided the content must be rejected
BLOCK_SUGGESTION: Final = "block"

# An explicit upstream block that carries no parseable severity is treated as the most
# severe level, so it is still weighed against the configured threshold rather than
# being silently downgraded to "none".
UNRESOLVED_BLOCK_LEVEL: Final = "high"

# Risk level to integer, for threshold comparison. Covers both the standard levels
# (none/low/medium/high) and the sensitiveData levels (S0-S4).
RISK_LEVEL_TO_INT: Final = MappingProxyType(
    {
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
)


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
    level_lower: Final = risk_level.lower() if risk_level else "none"
    return RISK_LEVEL_TO_INT.get(level_lower, 0)


# Sentence-ending punctuation, used to cut long text on a boundary instead of mid-sentence
SENTENCE_BOUNDARY_PATTERN: Final = r"[。！？；:\.?!]+"

# Parsed result of a response Aliyun did not flag at all
PASS_RESULT: Final = MappingProxyType(
    {"flagged": False, "suggestion": "pass", "details": MappingProxyType({}), "message": ""}
)

# Protection level thresholds
# If detected_level >= threshold, then block
PROTECTION_LEVEL_THRESHOLD: Final = MappingProxyType(
    {
        "low": 1,  # High protection: block low, medium, high (threshold=1, block if >=1)
        "medium": 2,  # Medium protection: block medium, high (threshold=2, block if >=2)
        "high": 3,  # Low protection: block high only (threshold=3, block if >=3)
        "max": 99,  # Observation mode: never block (threshold very high)
    }
)


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
        **kwargs: object,  # kwargs-ok: forwarded verbatim to CustomGuardrail's own **kwargs contract
    ) -> None:
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
        self.access_key_id = access_key_id or ""
        self.access_key_secret = access_key_secret or ""
        self.region_id = region_id or "cn-shanghai"
        if not self.access_key_id:
            raise ValueError(
                "Aliyun AI Guardrail: ak is required. Set access_key_id in config.yaml (supports os.environ/ reference)"
            )
        if not self.access_key_secret:
            raise ValueError(
                "Aliyun AI Guardrail: sk is required. "
                "Set access_key_secret in config.yaml (supports os.environ/ reference)"
            )
        self.level = level or "medium"
        if self.level not in PROTECTION_LEVEL_THRESHOLD:
            raise ValueError(
                f"Aliyun AI Guardrail: Invalid level '{self.level}'. "
                f"Valid values are: {tuple(PROTECTION_LEVEL_THRESHOLD)}"
            )
        self.max_text_length = max_text_length or 2000
        self.endpoint = REGION_ENDPOINTS.get(self.region_id, REGION_ENDPOINTS["cn-shanghai"])
        self.service_url = f"https://{self.endpoint}"
        self.service_input = service_input or SERVICE_INPUT_DOMESTIC
        self.service_output = service_output or SERVICE_OUTPUT_DOMESTIC
        self.service_mcp = service_mcp or SERVICE_INPUT_DOMESTIC
        self.stream_window_size = stream_window_size or 500
        self.stream_slide_step = stream_slide_step or 300
        self.stream_first_check_step = stream_first_check_step or 50
        verbose_proxy_logger.info(
            "Initialized Aliyun AI Security Guardrail: %s, "
            "region: %s, level: %s, "
            "service_input: %s, service_output: %s, "
            "service_mcp: %s",
            guardrail_name,
            self.region_id,
            self.level,
            self.service_input,
            self.service_output,
            self.service_mcp,
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
        secret: Final = self.access_key_secret + "&"
        signature: Final = hmac.new(
            secret.encode(ENCODING),
            string_to_sign.encode(ENCODING),
            hashlib.sha1,
        ).digest()
        return base64.b64encode(signature).decode(ENCODING)

    def _create_string_to_sign(self, http_method: str, parameters: Mapping[str, str]) -> str:
        """Create the string to sign for API request."""
        canonicalized_query_string: Final = "".join(
            f"&{self._percent_encode(key)}={self._percent_encode(parameters[key])}" for key in sorted(parameters)
        )
        return (
            http_method + "&" + self._percent_encode("/") + "&" + self._percent_encode(canonicalized_query_string[1:])
        )

    @staticmethod
    def _iter_text_segments(text: str, max_length: int) -> Iterator[str]:
        """
        Yield ``text`` in segments of at most ``max_length`` characters.
        Each cut lands on the last sentence boundary inside the window, so a
        segment does not split a sentence in half when it can be avoided.
        """
        start = 0  # rebind-ok: cursor advances by one emitted segment per iteration
        while start < len(text):
            if len(text) - start <= max_length:
                yield text[start:]
                return
            window = text[start : start + max_length]
            boundaries = tuple(re.finditer(SENTENCE_BOUNDARY_PATTERN, window))
            cut_point = boundaries[-1].end() if boundaries else max_length
            yield window[:cut_point]
            start += cut_point

    def _split_text(self, text: str, max_length: int = 2000) -> tuple[str, ...]:
        """
        Split text into segments of maximum length, trying to preserve sentence boundaries.
        Args:
            text: Text to split
            max_length: Maximum length of each segment
        Returns:
            The text segments, in order
        """
        return tuple(self._iter_text_segments(text, max_length))

    @staticmethod
    def _build_service_parameters(text: str | None, image_urls: Sequence[str] | None) -> Mapping[str, Any]:
        """
        Build the ServiceParameters payload of one guardrail request.
        Args:
            text: The text to audit, when there is any
            image_urls: The image URLs to audit, when there are any
        Returns:
            The payload, as a real dict because json.dumps rejects a MappingProxyType
        """
        return dict(  # mutable-ok: json.dumps rejects a MappingProxyType
            (
                ("requestFrom", "LiteLLM"),
                *((("content", text),) if text else ()),
                *((("imageUrls", tuple(image_urls)),) if image_urls else ()),
            )
        )

    async def async_make_request(
        self,
        text: str | None = None,
        service_type: Literal["input", "output", "mcp"] = "input",
        image_urls: Sequence[str] | None = None,
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

        service_code: Final = (
            self.service_mcp
            if service_type == "mcp"
            else self.service_input
            if service_type == "input"
            else self.service_output
        )
        service_parameters: Final = self._build_service_parameters(text, image_urls)
        # httpx form-encodes `data=` from a dict, so this payload stays a real dict too
        parameters: Final[dict[str, str]] = {  # mutable-ok: httpx form-encodes `data=` from a dict
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
        string_to_sign: Final = self._create_string_to_sign("POST", parameters)
        parameters["Signature"] = self._create_signature(string_to_sign)
        verbose_proxy_logger.debug(
            "Aliyun AI Guardrail request: service=%s, text_length=%d, image_count=%d",
            service_code,
            len(text) if text else 0,
            len(image_urls) if image_urls else 0,
        )
        response: Final = await self.async_handler.post(
            url=self.service_url,
            data=parameters,
            headers={"Content-Type": "application/x-www-form-urlencoded"},  # mutable-ok: httpx wants a dict
            timeout=30.0,
        )
        body: Final = response.json()
        verbose_proxy_logger.debug("Aliyun AI Guardrail response: %s", body)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail={  # mutable-ok: HTTPException detail payload
                    "error": f"Aliyun AI Guardrail request failed. Status: {response.status_code}, Body: {body}"
                },
            )
        if body.get("Code") != 200:
            raise HTTPException(
                status_code=400,
                detail={  # mutable-ok: HTTPException detail payload
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
        threshold: Final = PROTECTION_LEVEL_THRESHOLD.get(self.level, 99)
        detected_int: Final = level_to_int(detected_level)
        return detected_int >= threshold

    def _resolve_detail_level(self, detail: Mapping[str, Any]) -> str:
        """
        Resolve the risk level of a single Detail entry.

        MultiModalGuard reports severity in two shapes: the ``_pro`` service codes
        return ``Detail[].Level``, while the documented response carries it as
        ``Detail[].Result[].RiskLevel``. Honouring only the former downgrades the
        latter to "none", which would let content Aliyun rejected pass through.
        Args:
            detail: A single entry of the response ``Detail`` list
        Returns:
            The risk level string to weigh against the configured threshold
        """
        level: Final = detail.get("Level")
        if isinstance(level, str) and level.strip():
            return level
        # Fall back to the highest RiskLevel reported across the individual results
        risk_levels: Final = tuple(
            risk_level
            for result in detail.get("Result") or ()
            if isinstance(result, dict)
            for risk_level in (result.get("RiskLevel"),)
            if isinstance(risk_level, str) and risk_level.strip()
        )
        if risk_levels:
            # max() keeps the first of equal-severity levels, as the previous scan did
            return max(risk_levels, key=level_to_int)
        # No parseable severity: never downgrade an explicit block to "none"
        if detail.get("Suggestion") == BLOCK_SUGGESTION:
            return UNRESOLVED_BLOCK_LEVEL
        return "none"

    @staticmethod
    def _resolve_desensitization(detail_list: Sequence[Any]) -> str:
        """
        Return the desensitized text reported for sensitive data.
        The first ``Desensitization`` within one detail's results wins, and a later
        sensitiveData detail overrides an earlier one.
        Args:
            detail_list: The response ``Detail`` entries
        Returns:
            The desensitized text, empty when none was reported
        """
        resolved = ""  # rebind-ok: a later sensitiveData detail overrides an earlier one
        for detail in detail_list:
            if detail.get("Type", "") != SENSITIVE_DATA_TYPE:
                continue
            for ext in (result.get("Ext") for result in detail.get("Result", ()) or ()):
                if ext and ext.get("Desensitization"):
                    resolved = ext.get("Desensitization", "")
                    break
        return resolved

    def _resolve_block_decision(self, detail_list: Sequence[Any], final_suggestion: str) -> tuple[str, str, str]:
        """
        Decide whether the audited content must be blocked.
        Args:
            detail_list: The response ``Detail`` entries
            final_suggestion: The overall ``Suggestion`` of the response
        Returns:
            (blocked_type, blocked_level, block_message); an empty message means pass
        """
        for detail in detail_list:
            detected_level = self._resolve_detail_level(detail)
            if self._should_block_by_level(detected_level):
                detection_type = detail.get("Type", "")
                return detection_type, detected_level, f"检测到{detection_type} (风险等级: {detected_level})"
        # Aliyun rejected the content overall but no single detection could be attributed
        # (e.g. an empty Detail list, or every entry reporting pass). Treat it as the most
        # severe level so the decision is not lost, still subject to the threshold.
        if final_suggestion == BLOCK_SUGGESTION and self._should_block_by_level(UNRESOLVED_BLOCK_LEVEL):
            return "", UNRESOLVED_BLOCK_LEVEL, f"阿里云返回阻断建议 (Suggestion: {final_suggestion})"
        return "", "", ""

    def _summarise_detail(self, detail: Mapping[str, Any]) -> Mapping[str, Any]:
        """Summarise one ``Detail`` entry for the parsed result payload."""
        return {  # mutable-ok: returned payload
            "level": self._resolve_detail_level(detail),
            "suggestion": detail.get("Suggestion", "pass"),
            "results": tuple(detail.get("Result") or ()),
        }

    def _parse_response_and_check(
        self,
        response: AliyunAIGuardrailResponse,
        check_type: Literal["input", "output"],
    ) -> Mapping[str, Any]:
        """
        Parse the guardrail response and check if content should be blocked.
        Blocking logic:
        Check if detected level >= threshold based on protection level setting
        Args:
            response: The API response
            check_type: "input" or "output"
        Returns:
            Mapping with parsed results
        Raises:
            HTTPException if content should be blocked
        """
        data: Final = response.get("Data")
        if not data:
            return PASS_RESULT
        final_suggestion: Final = data.get("Suggestion", "pass")
        detail_list: Final = tuple(data.get("Detail") or ())
        details: Final[dict[str, Mapping[str, Any]]] = {  # mutable-ok: returned payload
            detail.get("Type", ""): self._summarise_detail(detail) for detail in detail_list
        }
        desensitization: Final = self._resolve_desensitization(detail_list)
        blocked_type, blocked_level, block_message = self._resolve_block_decision(detail_list, final_suggestion)
        verbose_proxy_logger.debug(
            "Aliyun AI Guardrail: level=%s, check_type=%s, should_block=%s, blocked_type=%s, blocked_level=%s",
            self.level,
            check_type,
            bool(block_message),
            blocked_type,
            blocked_level,
        )
        if block_message:
            raise HTTPException(
                status_code=400,
                detail={  # mutable-ok: HTTPException detail payload
                    "error": f"Aliyun AI Guardrail: {block_message}",
                    "type": check_type,
                    "details": details,
                },
            )
        return {  # mutable-ok: returned payload
            "flagged": final_suggestion != "pass",
            "suggestion": final_suggestion,
            "desensitization": desensitization,
            "details": details,
            "message": block_message,
        }

    @staticmethod
    def _build_input_payloads(
        segments: Sequence[str], image_urls: Sequence[str]
    ) -> tuple[tuple[str | None, Sequence[str] | None], ...]:
        """
        Pair each text segment with the images to audit alongside it.
        Every image URL rides with the first text segment so content and images are
        checked together; later segments carry text only. With no text at all, a
        single image-only request is produced.
        Args:
            segments: The text segments to audit
            image_urls: The image URLs to audit
        Returns:
            (text, images) pairs, one per request to send
        """
        if segments:
            return tuple(
                (segment, image_urls if idx == 0 and image_urls else None) for idx, segment in enumerate(segments)
            )
        if image_urls:
            return ((None, image_urls),)
        return ()

    @staticmethod
    def _iter_responses_function_call_arguments(data: Mapping[str, object]) -> Iterator[str]:
        input_value: Final = data.get("input")
        if not isinstance(input_value, list):
            return
        for item in input_value:
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            arguments = item.get("arguments")
            if isinstance(arguments, str) and arguments:
                yield arguments

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, Any],  # mutable-ok: CustomLogger's hook contract
        call_type: CallTypesLiteral,
    ) -> dict[str, Any] | None:  # mutable-ok: CustomLogger's hook contract
        """
        Pre-call hook to scan user prompts before sending to LLM.
        Raises HTTPException if content should be blocked.
        """
        verbose_proxy_logger.info(
            "Aliyun AI Guardrail: Running pre-call prompt scan, on call_type: %s",
            call_type,
        )
        if call_type == CallTypes.call_mcp_tool.value:
            await self._mcp_pre_call_check(data)
            return None
        # Walks messages AND the Responses API's input, so /v1/responses is covered
        new_messages: Final = cast(  # cast-ok: the upstream iterator is typed as plain dicts
            "Sequence[AllMessageValues]", tuple(_iter_inspection_messages(data))
        )
        function_call_arguments: Final = tuple(self._iter_responses_function_call_arguments(data))
        if not new_messages and not function_call_arguments:
            verbose_proxy_logger.warning("Aliyun AI Guardrail: not running guardrail. No messages in data")
            return data
        message_prompt: Final = self.get_user_prompt(new_messages)
        user_prompt: Final = (
            "\n".join(text for text in (message_prompt, *function_call_arguments) if text).strip() or None
        )
        image_urls: Final = self.get_image_urls(new_messages)
        if not user_prompt and not image_urls:
            verbose_proxy_logger.warning("Aliyun AI Guardrail: No user prompt or image found")
            return None
        verbose_proxy_logger.info(
            "Aliyun AI Guardrail: Pre-call scan started, prompt length: %d, image count: %d",
            len(user_prompt) if user_prompt else 0,
            len(image_urls),
        )
        # _split_text already returns a single segment when the text fits
        segments: Final = self._split_text(user_prompt, self.max_text_length) if user_prompt else ()
        payloads: Final = self._build_input_payloads(segments, image_urls)
        semaphore: Final = asyncio.Semaphore(5)  # Max 5 concurrent requests, MultiModalGuard API limit is 20

        async def check_with_semaphore(
            segment_text: str | None, segment_images: Sequence[str] | None
        ) -> AliyunAIGuardrailResponse:
            async with semaphore:
                return await self.async_make_request(text=segment_text, service_type="input", image_urls=segment_images)

        responses: Final = await asyncio.gather(*(check_with_semaphore(t, imgs) for t, imgs in payloads))
        for response in responses:
            self._parse_response_and_check(response, check_type="input")
        verbose_proxy_logger.info("Aliyun AI Guardrail: Pre-call scan passed")
        return None

    # ================================================================
    # MCP-specific guardrail methods
    # ================================================================

    async def _mcp_pre_call_check(self, data: Mapping[str, Any]) -> None:
        """MCP pre-call: audit tool name + arguments before execution."""
        messages: Final = data.get("messages", ())
        content: Final = messages[0].get("content", "") if messages else ""
        verbose_proxy_logger.info(
            "Aliyun AI Guardrail: ★ MCP pre-call check started, content length: %d",
            len(content),
        )
        if not content:
            return
        # _split_text already returns a single segment when the text fits
        segments: Final = self._split_text(content, self.max_text_length)
        semaphore: Final = asyncio.Semaphore(5)

        async def check(text: str) -> AliyunAIGuardrailResponse:
            async with semaphore:
                return await self.async_make_request(text=text, service_type="mcp")

        responses: Final = await asyncio.gather(*(check(s) for s in segments))
        for resp in responses:
            self._parse_response_and_check(resp, check_type="input")
        verbose_proxy_logger.info("Aliyun AI Guardrail: ★ MCP pre-call check passed")

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
                default_value: Final = self.event_hook.default
                return "post_mcp_call" in (default_value if isinstance(default_value, list) else (default_value,))
            return False
        return self.event_hook == "post_mcp_call"

    async def async_post_mcp_tool_call_hook(
        self,
        kwargs: Mapping[str, Any],
        response_obj: MCPPostCallResponseObject,
        start_time: datetime,
        end_time: datetime,
    ) -> MCPPostCallResponseObject | None:
        """MCP post-call: audit tool output after execution.

        Raising from here does not block: the dispatcher treats every callback exception
        as a non-blocking logging error and hands the untouched tool result back. Both
        call sites also discard this hook's return value, so a violation has to be
        written into the live tool result carried by ``kwargs["original_response"]``.
        The replacement object is returned as well, to honour the hook's contract.
        """
        # Since GuardrailEventHooks enum has no post_mcp_call, the framework
        # always invokes this hook if implemented. We check config manually.
        if not self._should_run_post_mcp_call():
            verbose_proxy_logger.info("Aliyun AI Guardrail: Skipping post_mcp_call — not configured in event_hook")
            return None
        verbose_proxy_logger.info("Aliyun AI Guardrail: ★ MCP post-call check started")
        original_response: Final = kwargs.get("original_response")
        # The live tool result is preferred; the wrapped copy is the fallback.
        combined_text: Final = next(
            (
                text
                for text in (
                    self._extract_mcp_tool_text(candidate)
                    for candidate in (original_response, getattr(response_obj, "mcp_tool_call_response", None))
                    if candidate is not None
                )
                if text
            ),
            "",
        )
        if not combined_text:
            return None
        verbose_proxy_logger.info(
            "Aliyun AI Guardrail: ★ MCP post-call check, response length: %d",
            len(combined_text),
        )
        # _split_text already returns a single segment when the text fits
        segments: Final = self._split_text(combined_text, self.max_text_length)
        semaphore: Final = asyncio.Semaphore(5)

        async def check(text: str) -> AliyunAIGuardrailResponse:
            async with semaphore:
                return await self.async_make_request(text=text, service_type="mcp")

        try:
            responses: Final = await asyncio.gather(*(check(s) for s in segments))
            for resp in responses:
                self._parse_response_and_check(resp, check_type="output")
        except HTTPException as e:
            verbose_proxy_logger.warning(
                "Aliyun AI Guardrail: ★ MCP post-call blocked — tool output replaced with the violation detail"
            )
            return self._block_mcp_tool_output(
                detail=e.detail,
                response_obj=response_obj,
                original_response=original_response,
            )
        except Exception as e:  # noqa: BLE001 - fail closed on any guardrail failure
            # Raising here is swallowed by the dispatcher as a non-blocking logging
            # error, which would hand the unaudited tool output straight to the
            # client. Fail closed, matching every other path of this integration.
            verbose_proxy_logger.error(
                "Aliyun AI Guardrail: ★ MCP post-call check failed, blocking unaudited tool output: %s",
                e,
                exc_info=True,
            )
            return self._block_mcp_tool_output(
                detail={  # mutable-ok: HTTPException detail payload
                    "error": f"Aliyun AI Guardrail 调用失败，工具输出未经审核，已拦截: {e!s}"
                },  # mutable-ok: detail payload
                response_obj=response_obj,
                original_response=original_response,
            )
        verbose_proxy_logger.info("Aliyun AI Guardrail: ★ MCP post-call check passed")
        return None

    @staticmethod
    def _find_coerced_field(payload: Sequence[object], field: str) -> object:
        """
        Return one field of a CallToolResult that was coerced into (field, value) pairs.
        Args:
            payload: The coerced pairs
            field: The field name to recover
        Returns:
            The field's value, or None when it is absent
        """
        return next(
            (entry[1] for entry in payload if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == field),
            None,
        )

    @classmethod
    def _iter_mcp_content_items(cls, payload: object) -> tuple[object, ...]:
        """
        Normalise an MCP tool result into its content items.
        Args:
            payload: A CallToolResult, its content list, or a raw response body
        Returns:
            The content items to audit, empty when none can be located
        """
        if isinstance(payload, str):
            return (payload,)
        content: Final = getattr(payload, "content", None)
        if isinstance(content, list):
            return tuple(content)
        if isinstance(payload, dict):
            inner: Final = payload.get("content")
            return tuple(inner) if isinstance(inner, list) else (payload,)
        if isinstance(payload, list):
            # MCPPostCallResponseObject declares mcp_tool_call_response as a list, so a
            # CallToolResult handed to it is coerced by iterating the model into
            # (field, value) pairs. Recover the real content instead of auditing reprs.
            coerced: Final = cls._find_coerced_field(payload, "content")
            return tuple(coerced) if isinstance(coerced, list) else tuple(payload)
        return ()

    @classmethod
    def _extract_mcp_item_text(cls, item: object) -> str:
        """
        Collect the text of a single MCP content item.
        Args:
            item: A content item, as a string, dict or MCP content model
        Returns:
            The item's text, empty when it carries none
        """
        if isinstance(item, str):
            return item
        direct_text: Final = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
        if direct_text:
            return str(direct_text)
        # An EmbeddedResource keeps its payload one level down, on the resource.
        resource: Final = item.get("resource") if isinstance(item, dict) else getattr(item, "resource", None)
        resource_text: Final = resource.get("text") if isinstance(resource, dict) else getattr(resource, "text", None)
        return str(resource_text or "")

    @classmethod
    def _extract_mcp_structured_content(cls, payload: object) -> str:
        """
        Serialise a tool result's ``structuredContent``, when it carries one.
        Args:
            payload: A CallToolResult, a raw response body, or its coerced form
        Returns:
            The serialised structured payload, empty when there is none
        """
        structured: Final = (
            payload.get("structuredContent")
            if isinstance(payload, dict)
            # MCPPostCallResponseObject coerces a CallToolResult into (field, value) pairs.
            else cls._find_coerced_field(payload, "structuredContent")
            if isinstance(payload, list)
            else getattr(payload, "structuredContent", None)
        )
        if structured is None:
            return ""
        if isinstance(structured, str):
            return structured
        return json.dumps(structured, ensure_ascii=False, default=str)

    def _extract_mcp_tool_text(self, payload: object) -> str:
        """
        Collect the textual output of an MCP tool result.
        A tool result carries text beyond ``content[].text``: ``structuredContent``
        holds an arbitrary JSON payload and an EmbeddedResource keeps its text on
        ``resource.text``. Auditing only the plain text items would let a tool
        return prohibited content in either field unchecked.
        """
        item_texts: Final = tuple(
            text for text in map(self._extract_mcp_item_text, self._iter_mcp_content_items(payload)) if text
        )
        structured_text: Final = self._extract_mcp_structured_content(payload)
        return "\n".join((*item_texts, structured_text) if structured_text else item_texts)

    @staticmethod
    def _replace_tool_output_in_place(target: object, blocked_content: Sequence[Any]) -> bool:
        """
        Overwrite an MCP tool result's content with ``blocked_content``, in place.
        The in-place stores are the point: both dispatch sites hand the caller's own
        tool result to the client, so the violation has to be written into it.
        """
        if target is None:
            return False
        from litellm.proxy._experimental.mcp_server.utils import (
            set_mcp_tool_result_structured_content,
        )

        content: Final = getattr(target, "content", None)
        if isinstance(content, list):
            content[:] = blocked_content
            set_mcp_tool_result_structured_content(target, None)
            if hasattr(target, "isError"):
                try:
                    # Kept duck-typed on purpose: any MCP result shape carrying `isError`
                    # must be flagged, not just the SDK's CallToolResult.
                    target.isError = True  # pyright: ignore[reportAttributeAccessIssue]  # rebind-ok: flag the caller's result
                except (AttributeError, TypeError, ValueError):
                    pass
            return True
        if isinstance(target, list):
            target[:] = blocked_content  # rebind-ok: the caller's result must be overwritten
            return True
        if isinstance(target, dict):
            result: Final = target.get("result")
            if isinstance(result, dict) and isinstance(result.get("content"), list):
                result["content"] = list(blocked_content)  # mutable-ok: the replaced field must stay a JSON list
                set_mcp_tool_result_structured_content(result, None)
                return True
            if isinstance(target.get("content"), list):
                blocked: Final = list(blocked_content)  # mutable-ok: must stay a JSON list
                target["content"] = blocked  # rebind-ok: overwrite the caller's result
                set_mcp_tool_result_structured_content(target, None)
                return True
        return False

    def _block_mcp_tool_output(
        self,
        detail: object,
        response_obj: MCPPostCallResponseObject,
        original_response: object,
    ) -> MCPPostCallResponseObject:
        """Replace blocked MCP tool output, both in place and as the returned object."""
        from mcp.types import TextContent

        from litellm.types.llms.base import HiddenParams
        from litellm.types.mcp import MCPPostCallResponseObject as _MCPPostCallResponseObject

        payload: Final = detail if isinstance(detail, dict) else {"error": str(detail)}  # mutable-ok: JSON payload
        blocked_content: Final = (TextContent(type="text", text=json.dumps(payload, ensure_ascii=False)),)
        for target in (original_response, getattr(response_obj, "mcp_tool_call_response", None)):
            self._replace_tool_output_in_place(target, blocked_content)
        hidden_params: Final = getattr(response_obj, "hidden_params", None)
        return _MCPPostCallResponseObject(
            mcp_tool_call_response=blocked_content,
            hidden_params=hidden_params if isinstance(hidden_params, HiddenParams) else HiddenParams(),
        )

    @staticmethod
    def _iter_function_call_text(function: object) -> Iterator[str]:
        """Yield the name and arguments of a tool/function call."""
        for field in ("name", "arguments"):
            value = getattr(function, field, None)
            if value:
                yield str(value)

    @staticmethod
    def _iter_output_item_text(output_items: Iterable[object] | None) -> Iterator[str]:
        """Yield the text carried by Responses API output items."""
        for output_item in output_items or ():
            content_parts = getattr(output_item, "content", None)
            if content_parts:
                for content_part in content_parts:
                    text = getattr(content_part, "text", None)
                    if text:
                        yield str(text)
            else:
                text = getattr(output_item, "text", None)
                if text:
                    yield str(text)

    @classmethod
    def _iter_completion_text(cls, message: object) -> Iterator[str]:
        """
        Yield every text field a chat completion message or streaming delta carries.
        Args:
            message: A choice's ``message`` (non-streaming) or ``delta`` (streaming)
        """
        for field in ("content", "reasoning_content"):
            value = getattr(message, field, None)
            if value:
                yield str(value)
        for call in getattr(message, "tool_calls", None) or ():
            yield from cls._iter_function_call_text(getattr(call, "function", None))
        yield from cls._iter_function_call_text(getattr(message, "function_call", None))

    @classmethod
    def _extract_response_text(cls, response: object) -> str:
        """
        Collect every text field a non-streaming response can carry.
        Mirrors ``_extract_stream_chunk_text``: auditing only ``message.content``
        would release tool call arguments, reasoning text and every
        /v1/responses body to the client unchecked whenever ``stream=False``.
        Args:
            response: A non-streaming response object
        Returns:
            The concatenated text to audit, empty when the response carries none
        """
        return "\n".join(
            (
                # Responses API bodies keep their text in output items, not in choices.
                *cls._iter_output_item_text(getattr(response, "output", None)),
                *(
                    text
                    for choice in getattr(response, "choices", None) or ()
                    for text in cls._iter_completion_text(getattr(choice, "message", None))
                ),
            )
        )

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,  # mutable-ok: CustomLogger's hook contract
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        """
        Post-call hook to scan LLM responses.
        Raises HTTPException if content should be blocked.
        """
        # Every choice is scanned: n>1 responses would otherwise return unchecked content
        content: Final = self._extract_response_text(response)
        if content:
            verbose_proxy_logger.info(
                "Aliyun AI Guardrail: Post-call scan started, response length: %d",
                len(content),
            )
            # _split_text already returns a single segment when the text fits
            segments: Final = self._split_text(content, self.max_text_length)
            semaphore: Final = asyncio.Semaphore(5)  # Max 5 concurrent requests

            async def check_with_semaphore(segment: str) -> AliyunAIGuardrailResponse:
                async with semaphore:
                    return await self.async_make_request(text=segment, service_type="output")

            responses: Final = await asyncio.gather(*(check_with_semaphore(segment) for segment in segments))
            for guardrail_response in responses:
                self._parse_response_and_check(guardrail_response, check_type="output")
            verbose_proxy_logger.info("Aliyun AI Guardrail: Post-call scan passed")
        return response

    @classmethod
    def _extract_stream_chunk_text(cls, chunk: object) -> str:
        """
        Collect every text field a streaming chunk can carry.
        Covers both chat completion chunks and Responses API streaming events.
        Auditing only ``delta.content`` would release tool call arguments,
        reasoning text and every /v1/responses chunk to the client unchecked.
        Args:
            chunk: A streaming chunk
        Returns:
            The concatenated text to audit, empty when the chunk carries none
        """
        # Responses API events carry text outside of choices: either a plain
        # string delta, or the assembled output of a terminal event.
        event_delta: Final = getattr(chunk, "delta", None)
        return "".join(
            (
                *((event_delta,) if isinstance(event_delta, str) and event_delta else ()),
                *cls._iter_output_item_text(getattr(getattr(chunk, "response", None), "output", None)),
                *(
                    text
                    for choice in getattr(chunk, "choices", None) or ()
                    for text in cls._iter_completion_text(getattr(choice, "delta", None))
                ),
            )
        )

    @staticmethod
    def _as_error_payload(detail: object) -> Mapping[str, Any]:
        """Normalise an HTTPException detail into a JSON object."""
        return detail if isinstance(detail, dict) else {"message": str(detail)}  # mutable-ok: JSON payload

    def _stream_check_windows(self, text: str, last_check_position: int) -> tuple[str, ...]:
        current_length: Final = len(text)
        window_size: Final = self.stream_window_size
        latest_window_start: Final = max(0, current_length - window_size)
        if current_length - last_check_position <= window_size:
            return (text[latest_window_start:],)
        step: Final = max(1, min(self.stream_slide_step, window_size))
        overlap: Final = window_size - step
        first_window_start: Final = max(0, last_check_position - overlap)
        window_starts: Final = (*range(first_window_start, latest_window_start, step), latest_window_start)
        return tuple(text[start : min(start + window_size, current_length)] for start in window_starts)

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncIterable[Any],
        request_data: Mapping[str, Any],
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
        # Sliding-window state: every name below advances as the stream is consumed.
        accumulated_text = ""  # rebind-ok: grows with each chunk
        last_check_position = 0  # rebind-ok: position (total length) of the last check
        pending_chunks: Final[list[Any]] = []  # mutable-ok: buffer held back until a check passes
        is_first_check = True  # rebind-ok: the first check uses a smaller threshold
        verbose_proxy_logger.info(
            "Aliyun AI Guardrail: Streaming scan started, window=%d, step=%d, first_check_step=%d",
            self.stream_window_size,
            self.stream_slide_step,
            self.stream_first_check_step,
        )
        try:
            async for chunk in response:
                chunk_text = self._extract_stream_chunk_text(chunk)
                accumulated_text += chunk_text
                # Buffer the chunk, don't yield until guardrail check passes
                pending_chunks.append(chunk)
                current_length = len(accumulated_text)
                new_chars_since_last_check = current_length - last_check_position
                check_threshold = self.stream_first_check_step if is_first_check else self.stream_slide_step
                if new_chars_since_last_check >= check_threshold:
                    check_windows = self._stream_check_windows(accumulated_text, last_check_position)
                    for text_to_check in check_windows:
                        guardrail_response = await self.async_make_request(text=text_to_check, service_type="output")
                        self._parse_response_and_check(guardrail_response, check_type="output")
                    verbose_proxy_logger.info(
                        "Aliyun AI Guardrail: Streaming check passed at position %d", current_length
                    )
                    for pending_chunk in pending_chunks:
                        yield pending_chunk
                    pending_chunks.clear()
                    last_check_position = current_length
                    is_first_check = False
            # Stream ended - check any remaining unchecked content with a final window
            if len(accumulated_text) > last_check_position:
                final_windows: Final = self._stream_check_windows(accumulated_text, last_check_position)
                for remaining_text in final_windows:
                    final_response = await self.async_make_request(text=remaining_text, service_type="output")
                    self._parse_response_and_check(final_response, check_type="output")
            verbose_proxy_logger.info(
                "Aliyun AI Guardrail: Streaming scan completed, total length: %d", len(accumulated_text)
            )
            for pending_chunk in pending_chunks:
                yield pending_chunk
            pending_chunks.clear()
        except HTTPException as e:
            detail_payload: Final = self._as_error_payload(e.detail)
            verbose_proxy_logger.info("Aliyun AI Guardrail: Streaming blocked at position %d", len(accumulated_text))
            payload: Final = json.dumps({"error": detail_payload}, ensure_ascii=False)  # mutable-ok: SSE payload
            yield f"data: {payload}\n\n"
            return
        except Exception as e:
            verbose_proxy_logger.error("Aliyun AI Guardrail streaming error: %s", e, exc_info=True)
            raise
