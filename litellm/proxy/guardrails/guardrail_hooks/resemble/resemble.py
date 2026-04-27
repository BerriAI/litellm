# +-------------------------------------------------------------+
#
#           Use Resemble AI Detect for your LLM calls
#                     https://www.resemble.ai/
#
#   Scans audio, video, and image URLs referenced in requests
#   for deepfake / synthetic content. Blocks requests whose
#   media inputs Resemble labels as fake or whose aggregated
#   score exceeds the configured threshold.
#
# +-------------------------------------------------------------+

import asyncio
import os
import re
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

from fastapi import HTTPException

from litellm import DualCache
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
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )


# Matches HTTPS URLs ending in common audio/video/image extensions. Query
# strings and fragments are allowed. Kept intentionally simple — multimodal
# content parts and metadata lookups handle the non-URL-in-text cases.
MEDIA_URL_REGEX = re.compile(
    r"https?://[^\s<>\"')\]}]+?\.(?:mp3|wav|m4a|flac|ogg|opus|aac|webm|mp4|mov|avi|mkv|jpg|jpeg|png|webp|gif)(?:\?[^\s<>\"')\]}]*)?",
    re.IGNORECASE,
)

RESEMBLE_DEFAULT_API_BASE = "https://app.resemble.ai/api/v2"


class ResembleGuardrailMissingSecrets(Exception):
    """Raised when the Resemble API key is not configured."""

    pass


class ResembleGuardrailAPIError(Exception):
    """Raised when the Resemble API returns an unexpected error."""

    pass


class ResembleGuardrail(CustomGuardrail):
    """
    Resemble AI Detect guardrail for LiteLLM.

    Extracts media URLs from LLM request inputs (multimodal content parts,
    regex from text, or metadata) and submits them to Resemble Detect. Blocks
    the request when the returned label is ``fake`` or when the aggregated
    score exceeds the configured threshold.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        threshold: Optional[float] = None,
        media_type: Optional[Literal["audio", "video", "image"]] = None,
        audio_source_tracing: Optional[bool] = None,
        use_reverse_search: Optional[bool] = None,
        zero_retention_mode: Optional[bool] = None,
        metadata_key: Optional[str] = None,
        poll_interval_seconds: Optional[float] = None,
        poll_timeout_seconds: Optional[float] = None,
        fail_closed: Optional[bool] = None,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        resolved_api_key = api_key or os.environ.get("RESEMBLE_API_KEY")
        if not resolved_api_key:
            raise ResembleGuardrailMissingSecrets(
                "Couldn't get Resemble API key. Set the `RESEMBLE_API_KEY` "
                "environment variable or pass `api_key` in the guardrail config."
            )
        self.api_key: str = resolved_api_key

        self.api_base: str = (
            api_base or os.environ.get("RESEMBLE_API_BASE") or RESEMBLE_DEFAULT_API_BASE
        ).rstrip("/")

        self.threshold: float = threshold if threshold is not None else 0.5
        self.media_type: Optional[str] = media_type
        self.audio_source_tracing: bool = bool(audio_source_tracing)
        self.use_reverse_search: bool = bool(use_reverse_search)
        self.zero_retention_mode: bool = bool(zero_retention_mode)
        self.metadata_key: str = metadata_key or "mediaUrl"
        self.poll_interval_seconds: float = (
            poll_interval_seconds if poll_interval_seconds is not None else 2.0
        )
        self.poll_timeout_seconds: float = (
            poll_timeout_seconds if poll_timeout_seconds is not None else 60.0
        )
        self.fail_closed: bool = bool(fail_closed)

        verbose_proxy_logger.debug(
            "Resemble guardrail initialized: name=%s threshold=%s media_type=%s "
            "audio_source_tracing=%s use_reverse_search=%s zero_retention_mode=%s "
            "fail_closed=%s",
            kwargs.get("guardrail_name", "unknown"),
            self.threshold,
            self.media_type,
            self.audio_source_tracing,
            self.use_reverse_search,
            self.zero_retention_mode,
            self.fail_closed,
        )

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Hook entrypoints
    # ------------------------------------------------------------------

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
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Union[Exception, str, dict, None]:
        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        await self._scan_request(data)
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
            "responses",
            "mcp_call",
            "anthropic_messages",
        ],
    ):
        event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        await self._scan_request(data)
        return data

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply Resemble Detect through LiteLLM's generic guardrail endpoint.

        The generic path supplies normalized texts/images/structured messages
        instead of the raw chat request shape, so collect URLs from both the
        normalized inputs and the original request data before scanning.
        """
        media_urls = self._extract_media_urls_from_guardrail_inputs(
            inputs=inputs,
            request_data=request_data,
        )
        if not media_urls:
            verbose_proxy_logger.debug(
                "Resemble guardrail: no media URL found in %s inputs — passing through",
                input_type,
            )
            return inputs

        for media_url in media_urls:
            await self._scan_single_url(media_url)
        return inputs

    # ------------------------------------------------------------------
    # Core scan logic
    # ------------------------------------------------------------------

    async def _scan_request(self, data: dict) -> None:
        media_urls = self._extract_media_urls(data)
        if not media_urls:
            verbose_proxy_logger.debug(
                "Resemble guardrail: no media URL found in request — passing through"
            )
            return

        for media_url in media_urls:
            await self._scan_single_url(media_url)

    async def _scan_single_url(self, media_url: str) -> None:
        try:
            item = await self._create_and_poll_detection(media_url)
        except HTTPException:
            raise
        except Exception as exc:
            self._handle_api_error(exc, media_url)
            return

        if item.get("status") == "failed":
            self._handle_api_error(
                ResembleGuardrailAPIError(
                    f"Resemble detection failed: {item.get('error_message') or 'unknown reason'}"
                ),
                media_url,
            )
            return

        evaluation = self._evaluate_detection(item)
        if not evaluation["verdict"]:
            detail: Dict[str, Any] = {
                "error": "Resemble Detect flagged media as synthetic",
                "resemble": {
                    "uuid": item.get("uuid"),
                    "media_url": media_url,
                    "media_type": item.get("media_type"),
                    "label": evaluation["label"],
                    "score": evaluation["score"],
                    "threshold": self.threshold,
                    "reason": evaluation["reason"],
                    "audio_source_tracing": item.get("audio_source_tracing"),
                },
            }
            raise HTTPException(status_code=400, detail=detail)

        verbose_proxy_logger.debug(
            "Resemble guardrail: passed (label=%s score=%s threshold=%s uuid=%s)",
            evaluation["label"],
            evaluation["score"],
            self.threshold,
            item.get("uuid"),
        )

    def _handle_api_error(self, exc: Exception, media_url: str) -> None:
        verbose_proxy_logger.warning(
            "Resemble guardrail API error for %s: %s", media_url, exc
        )
        if self.fail_closed:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Resemble Detect API call failed",
                    "resemble": {
                        "media_url": media_url,
                        "reason": str(exc),
                    },
                },
            )
        # Fail open — swallow the error so the LLM request proceeds.

    # ------------------------------------------------------------------
    # URL extraction
    # ------------------------------------------------------------------

    def _extract_media_urls(self, data: dict) -> List[str]:
        """
        Collect every media URL referenced in the request.

        Scans all multimodal content parts across every message, then every
        URL in the joined request text, then the metadata fallback. Returns
        a de-duplicated list preserving discovery order.

        Returning only the first match (the previous behaviour) let callers
        sneak a synthetic URL past the guardrail by placing a benign one
        earlier in the content array.
        """
        seen: Dict[str, None] = {}

        def _add(url: Optional[str]) -> None:
            if isinstance(url, str) and url and url not in seen:
                seen[url] = None

        messages = data.get("messages") or []
        for url in self._urls_from_content_parts(messages):
            _add(url)
        for url in self._urls_from_text(messages, data):
            _add(url)
        for url in self._urls_from_metadata(data):
            _add(url)

        return list(seen.keys())

    def _extract_media_urls_from_guardrail_inputs(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
    ) -> List[str]:
        """Collect media URLs from generic guardrail inputs and request data."""
        seen: Dict[str, None] = {}

        def _add(url: Optional[str]) -> None:
            if isinstance(url, str) and url and url not in seen:
                seen[url] = None

        for url in inputs.get("images", []) or []:
            _add(url)

        for text in inputs.get("texts", []) or []:
            if isinstance(text, str):
                for match in MEDIA_URL_REGEX.finditer(text):
                    _add(match.group(0))

        structured_messages = inputs.get("structured_messages", []) or []
        for url in self._urls_from_content_parts(structured_messages):
            _add(url)
        for url in self._urls_from_text(structured_messages, {}):
            _add(url)

        for url in self._extract_media_urls(request_data):
            _add(url)

        return list(seen.keys())

    @staticmethod
    def _urls_from_content_parts(messages: Any) -> List[str]:
        """Pull every URL out of multimodal content parts across all messages."""
        urls: List[str] = []
        if not isinstance(messages, list):
            return urls
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "input_audio":
                    audio = part.get("input_audio")
                    if isinstance(audio, dict) and isinstance(audio.get("url"), str):
                        urls.append(audio["url"])
                elif part_type == "image_url":
                    image_url = part.get("image_url")
                    if isinstance(image_url, dict) and isinstance(
                        image_url.get("url"), str
                    ):
                        urls.append(image_url["url"])
                elif part_type in ("image", "document"):
                    source = part.get("source")
                    if (
                        isinstance(source, dict)
                        and source.get("type") == "url"
                        and isinstance(source.get("url"), str)
                    ):
                        urls.append(source["url"])
        return urls

    @staticmethod
    def _urls_from_text(messages: Any, data: dict) -> List[str]:
        """Find every media URL embedded in plain text fields (messages, prompt, input)."""
        text_chunks: List[str] = []
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    text_chunks.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "text"
                            and isinstance(part.get("text"), str)
                        ):
                            text_chunks.append(part["text"])
        prompt = data.get("prompt")
        if isinstance(prompt, str):
            text_chunks.append(prompt)
        input_value = data.get("input")
        if isinstance(input_value, str):
            text_chunks.append(input_value)

        joined = "\n".join(text_chunks)
        return [match.group(0) for match in MEDIA_URL_REGEX.finditer(joined)]

    def _urls_from_metadata(self, data: dict) -> List[str]:
        """Metadata fallback — accepts a single URL string or a list of URLs."""
        metadata = data.get("metadata") or {}
        candidate = metadata.get(self.metadata_key)
        if isinstance(candidate, str):
            return [candidate]
        if isinstance(candidate, list):
            return [entry for entry in candidate if isinstance(entry, str)]
        return []

    # ------------------------------------------------------------------
    # Resemble API
    # ------------------------------------------------------------------

    async def _create_and_poll_detection(self, media_url: str) -> Dict[str, Any]:
        create_payload: Dict[str, Any] = {"url": media_url}
        if self.media_type:
            create_payload["media_type"] = self.media_type
        if self.audio_source_tracing:
            create_payload["audio_source_tracing"] = True
        if self.use_reverse_search:
            create_payload["use_reverse_search"] = True
        if self.zero_retention_mode:
            create_payload["zero_retention_mode"] = True

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        create_response = await self.async_handler.post(
            url=f"{self.api_base}/detect",
            headers=headers,
            json=create_payload,
            timeout=10.0,
        )
        create_response.raise_for_status()
        create_body = create_response.json()

        item = create_body.get("item") or {}
        uuid = item.get("uuid")
        if not uuid:
            raise ResembleGuardrailAPIError(
                "Resemble /detect response is missing item.uuid — cannot poll"
            )

        # If the server returned a completed item synchronously, skip polling.
        if item.get("status") == "completed" and self._item_has_metrics(item):
            return item

        return await self._poll_detection(uuid, headers)

    @staticmethod
    def _item_has_metrics(item: Dict[str, Any]) -> bool:
        """True when the detect item carries the scoring fields we evaluate."""
        return bool(
            item.get("metrics")
            or item.get("image_metrics")
            or item.get("video_metrics")
        )

    async def _poll_detection(
        self, uuid: str, headers: Dict[str, str]
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + self.poll_timeout_seconds
        poll_url = f"{self.api_base}/detect/{uuid}"

        while time.monotonic() < deadline:
            response = await self.async_handler.get(
                url=poll_url,
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
            body = response.json()
            item = body.get("item") or {}
            if not item:
                raise ResembleGuardrailAPIError(
                    f"Resemble GET /detect/{uuid} returned no item"
                )
            status = item.get("status")
            # "failed" is terminal; "completed" is only terminal once metrics
            # have landed — otherwise a racy API response could slip a
            # metric-less item past _evaluate_detection (which would treat
            # missing metrics as "unknown/0.0" and silently pass the request).
            if status == "failed":
                return item
            if status == "completed" and self._item_has_metrics(item):
                return item
            await asyncio.sleep(self.poll_interval_seconds)

        raise ResembleGuardrailAPIError(
            f"Resemble detection timed out after {self.poll_timeout_seconds}s (uuid={uuid})"
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_detection(self, item: Dict[str, Any]) -> Dict[str, Any]:
        label, score = self._extract_label_and_score(item)
        is_fake = label == "fake" or score >= self.threshold
        reason = (
            f"Resemble Detect flagged media as {label} (score={score}, "
            f"threshold={self.threshold})"
            if is_fake
            else f"Resemble Detect passed: label={label}, score={score}"
        )
        return {
            "verdict": not is_fake,
            "label": label,
            "score": score,
            "reason": reason,
        }

    def _extract_label_and_score(self, item: Dict[str, Any]) -> Tuple[str, float]:
        metrics = item.get("metrics")
        if isinstance(metrics, dict):
            return (
                str(metrics.get("label") or "unknown").lower(),
                self._coerce_score(metrics.get("aggregated_score")),
            )

        image_metrics = item.get("image_metrics")
        if isinstance(image_metrics, dict):
            return (
                str(image_metrics.get("label") or "unknown").lower(),
                self._coerce_score(image_metrics.get("score")),
            )

        video_metrics = item.get("video_metrics")
        if isinstance(video_metrics, dict):
            return (
                str(video_metrics.get("label") or "unknown").lower(),
                self._coerce_score(video_metrics.get("score")),
            )

        return ("unknown", 0.0)

    @staticmethod
    def _coerce_score(score: Any) -> float:
        return float(score if score is not None else 0)

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.resemble import (
            ResembleGuardrailConfigModel,
        )

        return ResembleGuardrailConfigModel
