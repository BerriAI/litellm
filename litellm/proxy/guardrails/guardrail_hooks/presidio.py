# +-----------------------------------------------+
# |                                               |
# |               PII Masking                     |
# |         with Microsoft Presidio               |
# |   https://github.com/BerriAI/litellm/issues/  |
# +-----------------------------------------------+
#
#  Tell us how we can improve! - Krrish & Ishaan


import asyncio
import json
import threading
import re
from datetime import datetime
from contextlib import asynccontextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

import aiohttp

import litellm  # noqa: E401
from litellm._uuid import uuid
from litellm import get_secret
from litellm._logging import verbose_proxy_logger
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

from litellm.caching.caching import DualCache
from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import (
    GuardrailEventHooks,
    LitellmParams,
    Mode,
    PiiAction,
    PiiEntityType,
    PresidioPerRequestConfig,
)
from litellm.types.proxy.guardrails.guardrail_hooks.presidio import (
    PresidioAnalyzeRequest,
    PresidioAnalyzeResponseItem,
)
from litellm.types.utils import GuardrailStatus, StreamingChoices
from litellm.utils import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream,
)


# Max trailing alphabetic chars to allow when matching corrupted uuid-style placeholders
# (e.g. LLM outputs "<PERSON>...fa9den" instead of "<PERSON>...fa9d"). Tune down (e.g. 3–5)
# if LLM rarely adds more than a few chars to reduce false matches.
_MAX_TRAILING_CHARS_CORRUPTED_PLACEHOLDER = 15
# Matches placeholders generated in anonymize_text as "<ENTITY_TYPE>" + str(uuid.uuid4()),
# i.e. no underscore and a full UUID4 suffix. str(uuid.uuid4()) currently produces
# lowercase hex only; uppercase [A-F] is accepted here for robustness if the UUID
# generator changes or a caller normalizes case differently. Keep this in sync with
# the placeholder generator in anonymize_text.
_UUID_SUFFIX_PLACEHOLDER_RE = re.compile(
    r"^<[^>]+>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _ensure_event_hook_includes_post_call(
    event_hook: Optional[Union[GuardrailEventHooks, List[Any], Mode, str]],
    include_pre_call_on_none: bool = False,
) -> Optional[Union[Mode, List[str], str]]:
    post_call = GuardrailEventHooks.post_call.value

    def _hook_value(hook: Any) -> str:
        if isinstance(hook, GuardrailEventHooks):
            return hook.value
        return str(hook)

    def _normalize_hook_list(hooks: List[Any]) -> List[str]:
        normalized: List[str] = []
        for hook in hooks:
            hook_value = _hook_value(hook)
            if hook_value not in normalized:
                normalized.append(hook_value)
        if post_call not in normalized:
            normalized.append(post_call)
        return normalized

    if event_hook is None:
        if include_pre_call_on_none:
            return [GuardrailEventHooks.pre_call.value, post_call]
        return post_call
    if isinstance(event_hook, Mode):
        mode_copy = event_hook.model_copy(deep=True)
        mode_copy.tags = {
            tag: _normalize_hook_list(values if isinstance(values, list) else [values])
            for tag, values in mode_copy.tags.items()
        }
        # Preserve tag-only Mode semantics: if default is None, untagged requests do
        # not run the guardrail. We only expand post_call for explicitly configured
        # tag/default hooks, rather than turning an absent default into a new one.
        if mode_copy.default is not None:
            mode_copy.default = _normalize_hook_list(
                mode_copy.default
                if isinstance(mode_copy.default, list)
                else [mode_copy.default]
            )
        return mode_copy
    if isinstance(event_hook, list):
        return _normalize_hook_list(event_hook)

    hook_value = _hook_value(event_hook)
    if hook_value == post_call:
        return hook_value
    return [hook_value, post_call]


def _get_corrupted_placeholder_pattern(key: str) -> re.Pattern:
    return re.compile(
        re.escape(key)
        + rf"[a-zA-Z0-9]{{1,{_MAX_TRAILING_CHARS_CORRUPTED_PLACEHOLDER}}}"
        + r"(?![a-zA-Z0-9])"
    )


def _replace_pii_tokens_in_text(text: str, pii_tokens: Dict[str, str]) -> str:
    """
    Replace PII placeholders in text with original values. Handles LLM corruption
    of uuid-style placeholders (e.g. <PERSON>uuid becomes <PERSON>uiden or
    <PERSON>uuid2) by matching key + trailing alphanumeric chars.
    """
    if not pii_tokens:
        return text
    corrupted_placeholder_patterns = {
        key: _get_corrupted_placeholder_pattern(key)
        for key in pii_tokens
        if _UUID_SUFFIX_PLACEHOLDER_RE.match(key) is not None
    }
    consumed_tokens = set()
    # Do regex pass first for uuid-style keys so "key+trailing" is replaced in one go.
    # If we did exact replace first, we'd replace the key and leave trailing chars (e.g. "Jane Doeen").
    for key, pattern in corrupted_placeholder_patterns.items():
        text, replacement_count = pattern.subn(pii_tokens[key], text)
        if replacement_count > 0:
            consumed_tokens.add(key)
    # Replace longer keys first so "<PERSON>uuid" is replaced before "<PERSON>"
    for key, value in sorted(pii_tokens.items(), key=lambda x: -len(x[0])):
        if key in text:
            consumed_tokens.add(key)
        text = text.replace(key, value)
    # Fallback for tokens truncated by max_tokens: if the end of the text is a
    # sufficiently long prefix of a placeholder, replace that suffix with the
    # original value so standard and Anthropic response paths behave the same.
    for token, original_text in pii_tokens.items():
        if token in consumed_tokens:
            continue
        if token in text:
            continue
        min_overlap = max(1, min(20, len(token) // 2))
        # Re-capture the latest text on each outer iteration. If a prior token
        # replacement shortened or lengthened the response, the next token's
        # suffix scan must use the updated string length and tail positions.
        current_text = text
        current_len = len(current_text)
        scan_start = max(0, current_len - len(token))
        for i in range(scan_start, current_len):
            sub = current_text[i:]
            if token.startswith(sub) and len(sub) >= min_overlap:
                # Safe to break: at most one suffix of `current_text` can be the
                # active truncated match for this token, and the next outer-loop
                # iteration will re-snapshot `text` after this replacement.
                text = current_text[:i] + original_text
                break
    return text


def _score_analyze_span_for_unmask_pairing(
    span: Dict[str, Any]
) -> Tuple[int, float, int]:
    """Score span quality for replace-token pairing: prefer longer, then higher-score, then earlier span."""
    start = cast(int, span["start"])
    end = cast(int, span["end"])
    raw_score = span.get("score")
    score = (
        float(raw_score)
        if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool)
        else -1.0
    )
    return (end - start, score, -start)


def _to_int_offset(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _dedupe_overlapping_analyze_spans_for_unmask(
    analyze_spans_sorted: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Collapse exact-duplicate and overlapping analyze spans for output_parse_pii mapping.
    Overlap clusters keep the most representative span to reduce mismatched pairing
    when analyzer returns nested/overlapping entities for the same text region.
    """
    if not analyze_spans_sorted:
        return []

    deduped_exact_spans: List[Dict[str, Any]] = []
    seen_spans = set()
    for span in analyze_spans_sorted:
        span_key = (cast(int, span["start"]), cast(int, span["end"]))
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)
        deduped_exact_spans.append(span)

    collapsed_spans: List[Dict[str, Any]] = []
    overlap_cluster: List[Dict[str, Any]] = []
    cluster_end = -1

    for span in deduped_exact_spans:
        span_start = cast(int, span["start"])
        span_end = cast(int, span["end"])
        if not overlap_cluster:
            overlap_cluster = [span]
            cluster_end = span_end
            continue

        # Overlap only when current start is inside the previous cluster.
        if span_start < cluster_end:
            overlap_cluster.append(span)
            cluster_end = max(cluster_end, span_end)
            continue

        collapsed_spans.append(
            max(overlap_cluster, key=_score_analyze_span_for_unmask_pairing)
        )
        overlap_cluster = [span]
        cluster_end = span_end

    if overlap_cluster:
        collapsed_spans.append(
            max(overlap_cluster, key=_score_analyze_span_for_unmask_pairing)
        )

    return collapsed_spans


class _OPTIONAL_PresidioPIIMasking(CustomGuardrail):
    user_api_key_cache = None
    ad_hoc_recognizers = None

    # Class variables or attributes
    def __init__(
        self,
        mock_testing: bool = False,
        mock_redacted_text: Optional[dict] = None,
        presidio_analyzer_api_base: Optional[str] = None,
        presidio_anonymizer_api_base: Optional[str] = None,
        output_parse_pii: Optional[bool] = False,
        apply_to_output: bool = False,
        presidio_ad_hoc_recognizers: Optional[str] = None,
        logging_only: Optional[bool] = None,
        pii_entities_config: Optional[
            Dict[Union[PiiEntityType, str], PiiAction]
        ] = None,
        presidio_language: Optional[str] = None,
        presidio_score_thresholds: Optional[
            Dict[Union[PiiEntityType, str], float]
        ] = None,
        presidio_entities_deny_list: Optional[List[Union[PiiEntityType, str]]] = None,
        **kwargs,
    ):
        if logging_only is True:
            self.logging_only = True
            kwargs["event_hook"] = GuardrailEventHooks.logging_only
        super().__init__(**kwargs)
        self.guardrail_provider = "presidio"
        # Deprecated request state. Keep attribute for backward compatibility with
        # tests/instrumentation, but request processing uses request-local mappings.
        self.pii_tokens: dict = {}
        self.mock_redacted_text = mock_redacted_text
        self.output_parse_pii = output_parse_pii or False
        self.apply_to_output = apply_to_output

        # When output_parse_pii or apply_to_output is enabled, the guardrail must
        # also run on post_call to unmask/mask the response.  Expand the event_hook
        # so should_run_guardrail returns True for both pre_call and post_call.
        if (self.output_parse_pii or self.apply_to_output) and not logging_only:
            self.event_hook = _ensure_event_hook_includes_post_call(
                self.event_hook,
                include_pre_call_on_none=self.output_parse_pii,
            )
        self.pii_entities_config: Dict[Union[PiiEntityType, str], PiiAction] = (
            pii_entities_config or {}
        )
        self.presidio_score_thresholds: Dict[Union[PiiEntityType, str], float] = (
            presidio_score_thresholds or {}
        )
        self.presidio_entities_deny_list: List[Union[PiiEntityType, str]] = (
            presidio_entities_deny_list or []
        )
        self.presidio_language = presidio_language or "en"
        # Shared HTTP session to prevent memory leaks (issue #14540)
        self._http_session: Optional[aiohttp.ClientSession] = None
        # Lock to prevent race conditions when creating session under concurrent load
        # Note: asyncio.Lock() can be created without an event loop; it only needs one when awaited
        self._session_lock: asyncio.Lock = asyncio.Lock()

        # Track main thread ID to safely identity when we are running in main loop vs background thread

        self._main_thread_id = threading.get_ident()

        # Loop-bound session cache for background threads
        self._loop_sessions: Dict[asyncio.AbstractEventLoop, aiohttp.ClientSession] = {}

        if mock_testing is True:  # for testing purposes only
            return

        ad_hoc_recognizers = presidio_ad_hoc_recognizers
        if ad_hoc_recognizers is not None:
            try:
                with open(ad_hoc_recognizers, "r") as file:
                    self.ad_hoc_recognizers = json.load(file)
            except FileNotFoundError:
                raise Exception(f"File not found. file_path={ad_hoc_recognizers}")
            except json.JSONDecodeError as e:
                raise Exception(
                    f"Error decoding JSON file: {str(e)}, file_path={ad_hoc_recognizers}"
                )
            except Exception as e:
                raise Exception(
                    f"An error occurred: {str(e)}, file_path={ad_hoc_recognizers}"
                )
        self.validate_environment(
            presidio_analyzer_api_base=presidio_analyzer_api_base,
            presidio_anonymizer_api_base=presidio_anonymizer_api_base,
        )

    def validate_environment(
        self,
        presidio_analyzer_api_base: Optional[str] = None,
        presidio_anonymizer_api_base: Optional[str] = None,
    ):
        self.presidio_analyzer_api_base: Optional[
            str
        ] = presidio_analyzer_api_base or get_secret(
            "PRESIDIO_ANALYZER_API_BASE", None
        )  # type: ignore
        self.presidio_anonymizer_api_base: Optional[
            str
        ] = presidio_anonymizer_api_base or litellm.get_secret(
            "PRESIDIO_ANONYMIZER_API_BASE", None
        )  # type: ignore

        if self.presidio_analyzer_api_base is None:
            raise Exception("Missing `PRESIDIO_ANALYZER_API_BASE` from environment")
        if not self.presidio_analyzer_api_base.endswith("/"):
            self.presidio_analyzer_api_base += "/"
        if not (
            self.presidio_analyzer_api_base.startswith("http://")
            or self.presidio_analyzer_api_base.startswith("https://")
        ):
            # add http:// if unset, assume communicating over private network - e.g. render
            self.presidio_analyzer_api_base = (
                "http://" + self.presidio_analyzer_api_base
            )

        if self.presidio_anonymizer_api_base is None:
            raise Exception("Missing `PRESIDIO_ANONYMIZER_API_BASE` from environment")
        if not self.presidio_anonymizer_api_base.endswith("/"):
            self.presidio_anonymizer_api_base += "/"
        if not (
            self.presidio_anonymizer_api_base.startswith("http://")
            or self.presidio_anonymizer_api_base.startswith("https://")
        ):
            # add http:// if unset, assume communicating over private network - e.g. render
            self.presidio_anonymizer_api_base = (
                "http://" + self.presidio_anonymizer_api_base
            )

    @asynccontextmanager
    async def _get_session_iterator(
        self,
    ) -> AsyncGenerator[aiohttp.ClientSession, None]:
        """
        Async context manager for yielding an HTTP session.

        Logic:
        1. If running in the main thread (where the object was initialized/destined to live normally),
           use the shared `self._http_session` (protected by a lock).
        2. If running in a background thread (e.g. logging hook), use a cached session for that loop.
        """
        current_loop = asyncio.get_running_loop()

        # Check if we are in the stored main thread
        if threading.get_ident() == self._main_thread_id:
            # Main thread -> use shared session
            async with self._session_lock:
                if self._http_session is None or self._http_session.closed:
                    self._http_session = aiohttp.ClientSession()
                yield self._http_session
        else:
            # Background thread/loop -> use loop-bound session cache
            # This avoids "attached to a different loop" or "no running event loop" errors
            # when accessing the shared session created in the main loop
            if (
                current_loop not in self._loop_sessions
                or self._loop_sessions[current_loop].closed
            ):
                self._loop_sessions[current_loop] = aiohttp.ClientSession()
            yield self._loop_sessions[current_loop]

    async def _close_http_session(self) -> None:
        """Close all cached HTTP sessions."""
        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

        for session in self._loop_sessions.values():
            if not session.closed:
                await session.close()
        self._loop_sessions.clear()

    def __del__(self):
        """Cleanup: we try to close, but doing async cleanup in __del__ is risky."""
        pass

    def _has_block_action(self) -> bool:
        """Return True if pii_entities_config has any BLOCK action (fail-closed on analyzer errors)."""
        if not self.pii_entities_config:
            return False
        return any(
            action == PiiAction.BLOCK for action in self.pii_entities_config.values()
        )

    def _get_presidio_analyze_request_payload(
        self,
        text: str,
        presidio_config: Optional[PresidioPerRequestConfig],
        request_data: dict,
    ) -> PresidioAnalyzeRequest:
        """
        Construct the payload for the Presidio analyze request

        API Ref: https://microsoft.github.io/presidio/api-docs/api-docs.html#tag/Analyzer/paths/~1analyze/post
        """
        analyze_payload: PresidioAnalyzeRequest = PresidioAnalyzeRequest(
            text=text,
            language=self.presidio_language,
        )
        ##################################################################
        ###### Check if user has configured any params for this guardrail
        ################################################################
        if self.ad_hoc_recognizers is not None:
            analyze_payload["ad_hoc_recognizers"] = self.ad_hoc_recognizers

        if self.pii_entities_config:
            analyze_payload["entities"] = list(self.pii_entities_config.keys())

        ##################################################################
        ######### End of adding config params
        ##################################################################

        # Check if client side request passed any dynamic params
        if presidio_config and presidio_config.language:
            analyze_payload["language"] = presidio_config.language

        casted_analyze_payload: dict = cast(dict, analyze_payload)
        casted_analyze_payload.update(
            self.get_guardrail_dynamic_request_body_params(request_data=request_data)
        )
        return cast(PresidioAnalyzeRequest, casted_analyze_payload)

    async def analyze_text(
        self,
        text: str,
        presidio_config: Optional[PresidioPerRequestConfig],
        request_data: dict,
    ) -> Union[List[PresidioAnalyzeResponseItem], Dict]:
        """
        Send text to the Presidio analyzer endpoint and get analysis results
        """
        try:
            # Skip empty or whitespace-only text to avoid Presidio errors
            # Common in tool/function calling where assistant content is empty
            if not text or len(text.strip()) == 0:
                verbose_proxy_logger.debug(
                    "Skipping Presidio analysis for empty/whitespace-only text"
                )
                return []

            if self.mock_redacted_text is not None:
                return self.mock_redacted_text

            # Use shared session to prevent memory leak (issue #14540)
            async with self._get_session_iterator() as session:
                # Make the request to /analyze
                analyze_url = f"{self.presidio_analyzer_api_base}analyze"

                analyze_payload: PresidioAnalyzeRequest = (
                    self._get_presidio_analyze_request_payload(
                        text=text,
                        presidio_config=presidio_config,
                        request_data=request_data,
                    )
                )

                verbose_proxy_logger.debug(
                    "Making request to: %s with payload: %s",
                    analyze_url,
                    analyze_payload,
                )

                def _fail_on_invalid_response(
                    reason: str,
                ) -> List[PresidioAnalyzeResponseItem]:
                    should_fail_closed = (
                        bool(self.pii_entities_config)
                        or self.output_parse_pii
                        or self.apply_to_output
                    )
                    if should_fail_closed:
                        raise GuardrailRaisedException(
                            guardrail_name=self.guardrail_name,
                            message=f"Presidio analyzer returned invalid response; cannot verify PII when PII protection is configured: {reason}",
                            should_wrap_with_default_message=False,
                        )
                    verbose_proxy_logger.warning(
                        "Presidio analyzer %s, returning empty list", reason
                    )
                    return []

                async with session.post(
                    analyze_url,
                    json=analyze_payload,
                    headers={"Accept": "application/json"},
                ) as response:
                    # Validate HTTP status
                    if response.status >= 400:
                        error_body = await response.text()
                        return _fail_on_invalid_response(
                            f"HTTP {response.status} from Presidio analyzer: {error_body[:200]}"
                        )

                    # Validate Content-Type is JSON
                    content_type = getattr(
                        response,
                        "content_type",
                        response.headers.get("Content-Type", ""),
                    )
                    if "application/json" not in content_type:
                        error_body = await response.text()
                        return _fail_on_invalid_response(
                            f"expected application/json Content-Type but received '{content_type}'; body: '{error_body[:200]}'"
                        )

                    analyze_results = await response.json()
                    verbose_proxy_logger.debug("analyze_results: %s", analyze_results)

                # Handle error responses from Presidio (e.g., {'error': 'No text provided'})
                # Presidio may return a dict instead of a list when errors occur

                if isinstance(analyze_results, dict):
                    if "error" in analyze_results:
                        return _fail_on_invalid_response(
                            f"error: {analyze_results.get('error')}"
                        )
                    # If it's a dict but not an error, try to process it as a single item
                    verbose_proxy_logger.debug(
                        "Presidio returned dict (not list), attempting to process as single item"
                    )
                    try:
                        return [PresidioAnalyzeResponseItem(**analyze_results)]
                    except Exception as e:
                        return _fail_on_invalid_response(
                            f"failed to parse dict response: {e}"
                        )

                # Handle unexpected types (str, None, etc.) - e.g. from malformed/error
                if not isinstance(analyze_results, list):
                    return _fail_on_invalid_response(
                        f"unexpected type {type(analyze_results).__name__} (expected list or dict), response: {str(analyze_results)[:200]}"
                    )

                # Normal case: list of results
                final_results = []
                for item in analyze_results:
                    if not isinstance(item, dict):
                        verbose_proxy_logger.warning(
                            "Skipping invalid Presidio result item (expected dict, got %s): %s",
                            type(item).__name__,
                            str(item)[:100],
                        )
                        continue
                    try:
                        final_results.append(PresidioAnalyzeResponseItem(**item))
                    except Exception as e:
                        verbose_proxy_logger.warning(
                            "Failed to parse Presidio result item: %s (error: %s)",
                            item,
                            e,
                        )
                        continue
                return final_results
        except GuardrailRaisedException:
            # Re-raise GuardrailRaisedException without wrapping
            raise
        except Exception as e:
            # Sanitize exception to avoid leaking the original text (which may
            # contain API keys or other secrets) in error responses.
            raise Exception(f"Presidio PII analysis failed: {type(e).__name__}") from e

    async def anonymize_text(
        self,
        text: str,
        analyze_results: Any,
        output_parse_pii: bool,
        masked_entity_count: Dict[str, int],
        pii_tokens: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Send analysis results to the Presidio anonymizer endpoint to get redacted text
        """
        try:
            # If there are no detections after filtering, return the original text
            if isinstance(analyze_results, list) and len(analyze_results) == 0:
                return text

            # Use shared session to prevent memory leak (issue #14540)
            async with self._get_session_iterator() as session:
                # Make the request to /anonymize
                anonymize_url = f"{self.presidio_anonymizer_api_base}anonymize"
                verbose_proxy_logger.debug("Making request to: %s", anonymize_url)
                anonymize_payload = {
                    "text": text,
                    "analyzer_results": analyze_results,
                }

                async with session.post(
                    anonymize_url,
                    json=anonymize_payload,
                    headers={"Accept": "application/json"},
                ) as response:
                    # Validate HTTP status
                    if response.status >= 400:
                        error_body = await response.text()
                        raise Exception(
                            f"Presidio anonymizer returned HTTP {response.status}: {error_body[:200]}"
                        )

                    # Validate Content-Type is JSON
                    content_type = getattr(
                        response,
                        "content_type",
                        response.headers.get("Content-Type", ""),
                    )
                    if "application/json" not in content_type:
                        error_body = await response.text()
                        raise Exception(
                            f"Presidio anonymizer returned non-JSON Content-Type '{content_type}'; body: '{error_body[:200]}'"
                        )

                    redacted_text = await response.json()

            if output_parse_pii and pii_tokens is None:
                verbose_proxy_logger.warning(
                    "Presidio output_parse_pii enabled but pii_tokens is None; "
                    "token mappings will be discarded and response unmasking may fail."
                )
            token_store = pii_tokens if pii_tokens is not None else {}
            if redacted_text is not None:
                verbose_proxy_logger.debug("redacted_text: %s", redacted_text)
                items = redacted_text["items"]
                for item in items:
                    entity_type = item.get("entity_type", None)
                    if entity_type is not None:
                        masked_entity_count[entity_type] = (
                            masked_entity_count.get(entity_type, 0) + 1
                        )

                # output_parse_pii is designed for replace-and-unmask flow.
                # If Presidio returns non-replace operators, use Presidio's text as-is
                # to avoid incorrect offset arithmetic on mixed operator outputs.
                if output_parse_pii:
                    replace_items = [i for i in items if i.get("operator") == "replace"]
                    non_replace_items = [
                        i for i in items if i.get("operator") != "replace"
                    ]
                    if non_replace_items:
                        replace_entity_types = [
                            str(entity_type)
                            for entity_type in (
                                i.get("entity_type") for i in replace_items
                            )
                            if entity_type is not None
                        ]
                        verbose_proxy_logger.warning(
                            "Presidio output_parse_pii fallback: detected non-replace "
                            "operators (%s); %d replace-operator entities (%s) will also "
                            "NOT be unmasked; returning redacted_text without unmask mapping.",
                            sorted({str(i.get("operator")) for i in non_replace_items}),
                            len(replace_items),
                            replace_entity_types,
                        )
                        return redacted_text["text"]

                    if not isinstance(analyze_results, list):
                        verbose_proxy_logger.warning(
                            "Presidio output_parse_pii fallback: analyze_results is not a list; "
                            "returning redacted_text without unmask mapping."
                        )
                        return redacted_text["text"]

                    analyze_spans = []
                    for r in analyze_results:
                        if not isinstance(r, dict):
                            continue
                        start_v = _to_int_offset(r.get("start"))
                        end_v = _to_int_offset(r.get("end"))
                        if start_v is None or end_v is None:
                            continue
                        analyze_spans.append(
                            {
                                "start": cast(int, start_v),
                                "end": cast(int, end_v),
                                "entity_type": r.get("entity_type"),
                                "score": r.get("score"),
                            }
                        )
                    # `replace_items` starts are in anonymized-output coordinates, while
                    # `analyze_spans` starts are in original-input coordinates. We still
                    # sort both lists left-to-right and pair them positionally, relying on
                    # Presidio preserving entity order across analyze/anonymize results.
                    # If that ordering ever changes, this logic would need stronger
                    # placeholder-to-span matching than simple positional pairing.
                    replace_items_sorted = sorted(
                        replace_items, key=lambda i: i["start"]
                    )
                    analyze_spans_sorted = sorted(
                        analyze_spans, key=lambda r: r["start"]
                    )
                    analyze_spans_sorted = _dedupe_overlapping_analyze_spans_for_unmask(
                        analyze_spans_sorted
                    )

                    if len(replace_items_sorted) != len(analyze_spans_sorted):
                        # Best-effort mapping: preserve left-to-right order and prefer
                        # matching entity_type before falling back to next available span.
                        if len(analyze_spans_sorted) >= len(replace_items_sorted):
                            best_effort_spans: List[Dict[str, Any]] = []
                            used_span_indices = set()
                            last_used_index = -1
                            for replace_item in replace_items_sorted:
                                replace_entity = replace_item.get("entity_type")
                                replace_entity_str = (
                                    str(
                                        getattr(replace_entity, "value", replace_entity)
                                    )
                                    if replace_entity is not None
                                    else None
                                )
                                selected_index: Optional[int] = None

                                for idx, span in enumerate(analyze_spans_sorted):
                                    if (
                                        idx in used_span_indices
                                        or idx <= last_used_index
                                    ):
                                        continue
                                    span_entity = span.get("entity_type")
                                    span_entity_str = (
                                        str(getattr(span_entity, "value", span_entity))
                                        if span_entity is not None
                                        else None
                                    )
                                    if span_entity_str == replace_entity_str:
                                        selected_index = idx
                                        break

                                if selected_index is None:
                                    for idx, _ in enumerate(analyze_spans_sorted):
                                        if (
                                            idx in used_span_indices
                                            or idx <= last_used_index
                                        ):
                                            continue
                                        selected_index = idx
                                        break

                                if selected_index is None:
                                    break

                                used_span_indices.add(selected_index)
                                last_used_index = selected_index
                                best_effort_spans.append(
                                    analyze_spans_sorted[selected_index]
                                )

                            if len(best_effort_spans) == len(replace_items_sorted):
                                verbose_proxy_logger.warning(
                                    "Presidio output_parse_pii best-effort mapping: replace item count (%s) "
                                    "does not match analyze span count (%s); using ordered span pairing.",
                                    len(replace_items_sorted),
                                    len(analyze_spans_sorted),
                                )
                                analyze_spans_sorted = best_effort_spans
                            else:
                                verbose_proxy_logger.warning(
                                    "Presidio output_parse_pii fallback: replace item count (%s) "
                                    "does not match analyze span count (%s); returning redacted_text.",
                                    len(replace_items_sorted),
                                    len(analyze_spans_sorted),
                                )
                                return redacted_text["text"]
                        else:
                            verbose_proxy_logger.warning(
                                "Presidio output_parse_pii fallback: replace item count (%s) "
                                "does not match analyze span count (%s); returning redacted_text.",
                                len(replace_items_sorted),
                                len(analyze_spans_sorted),
                            )
                            return redacted_text["text"]

                    # Build unique placeholders in original-text coordinates (from
                    # analyze results) so replacement offsets stay stable.
                    new_text = text
                    # Both lists are already sorted left-to-right in their own
                    # coordinate systems (analyze: original input, anonymize:
                    # anonymized output). Positional zipping is still valid
                    # because Presidio preserves entity order across both
                    # endpoints even when anonymized-output offsets shift.
                    for analyze_span, replace_item in zip(
                        reversed(analyze_spans_sorted), reversed(replace_items_sorted)
                    ):
                        start = cast(int, analyze_span["start"])
                        end = cast(int, analyze_span["end"])
                        if not (isinstance(start, int) and isinstance(end, int)):
                            raise RuntimeError(
                                "Presidio output_parse_pii: unexpected non-int span offsets "
                                f"start={start!r} end={end!r}"
                            )
                        replacement = cast(str, replace_item["text"])
                        replacement = f"{replacement}{str(uuid.uuid4())}"
                        token_store[replacement] = text[start:end]
                        verbose_proxy_logger.debug(
                            "Presidio output_parse_pii pair: placeholder=%s entity=%s original=%r",
                            replacement,
                            analyze_span.get("entity_type"),
                            text[start:end],
                        )
                        # Safe to splice against `new_text` using original-text offsets:
                        # upstream span dedupe removes overlaps, and right-to-left
                        # replacement keeps positions to the left of `start` stable.
                        new_text = new_text[:start] + replacement + new_text[end:]
                    return new_text

                # output_parse_pii disabled: return Presidio's redacted text directly.
                return redacted_text["text"]
            else:
                raise Exception("Invalid anonymizer response: received None")
        except Exception as e:
            # Sanitize exception to avoid leaking the original text (which may
            # contain API keys or other secrets) in error responses.
            error_str = str(e)
            if (
                "Invalid anonymizer response" in error_str
                or "Presidio anonymizer returned" in error_str
            ):
                raise
            raise Exception(
                f"Presidio PII anonymization failed: {type(e).__name__}"
            ) from e

    def filter_analyze_results_by_score(
        self, analyze_results: Union[List[PresidioAnalyzeResponseItem], Dict]
    ) -> Union[List[PresidioAnalyzeResponseItem], Dict]:
        """
        Drop detections that fall below configured per-entity score thresholds
        or match an entity type in the deny list.
        """
        if not self.presidio_score_thresholds and not self.presidio_entities_deny_list:
            return analyze_results

        if not isinstance(analyze_results, list):
            return analyze_results

        filtered_results: List[PresidioAnalyzeResponseItem] = []
        deny_list_strings = [
            getattr(x, "value", str(x)) for x in self.presidio_entities_deny_list
        ]
        for item in analyze_results:
            entity_type = item.get("entity_type")

            str_entity_type = str(
                getattr(entity_type, "value", entity_type)
                if entity_type is not None
                else entity_type
            )
            if entity_type and str_entity_type in deny_list_strings:
                continue

            if self.presidio_score_thresholds:
                score = item.get("score")
                threshold = None
                if entity_type is not None:
                    threshold = self.presidio_score_thresholds.get(entity_type)
                if threshold is None:
                    threshold = self.presidio_score_thresholds.get("ALL")

                if threshold is not None:
                    if score is None or score < threshold:
                        continue

            filtered_results.append(item)

        return filtered_results

    def raise_exception_if_blocked_entities_detected(
        self, analyze_results: Union[List[PresidioAnalyzeResponseItem], Dict]
    ):
        """
        Raise an exception if blocked entities are detected
        """
        if self.pii_entities_config is None:
            return

        if isinstance(analyze_results, Dict):
            # if mock testing is enabled, analyze_results is a dict
            # we don't need to raise an exception in this case
            return

        for result in analyze_results:
            entity_type = result.get("entity_type")

            if entity_type:
                # Check if entity_type is in config (supports both enum and string)
                if (
                    entity_type in self.pii_entities_config
                    and self.pii_entities_config[entity_type] == PiiAction.BLOCK
                ):
                    raise BlockedPiiEntityError(
                        entity_type=entity_type,
                        guardrail_name=self.guardrail_name,
                    )

    async def check_pii(
        self,
        text: str,
        output_parse_pii: bool,
        presidio_config: Optional[PresidioPerRequestConfig],
        request_data: dict,
        pii_tokens: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Calls Presidio Analyze + Anonymize endpoints for PII Analysis + Masking
        """
        start_time = datetime.now()
        analyze_results: Optional[Union[List[PresidioAnalyzeResponseItem], Dict]] = None
        status: GuardrailStatus = "success"
        masked_entity_count: Dict[str, int] = {}
        exception_str: str = ""
        try:
            if self.mock_redacted_text is not None:
                redacted_text = self.mock_redacted_text
            else:
                # First get analysis results
                analyze_results = await self.analyze_text(
                    text=text,
                    presidio_config=presidio_config,
                    request_data=request_data,
                )

                verbose_proxy_logger.debug("analyze_results: %s", analyze_results)

                # Apply score threshold filtering if configured
                analyze_results = self.filter_analyze_results_by_score(
                    analyze_results=analyze_results
                )

                ####################################################
                # Blocked Entities check
                ####################################################
                self.raise_exception_if_blocked_entities_detected(
                    analyze_results=analyze_results
                )

                # Then anonymize the text using the analysis results
                anonymized_text = await self.anonymize_text(
                    text=text,
                    analyze_results=analyze_results,
                    output_parse_pii=output_parse_pii,
                    masked_entity_count=masked_entity_count,
                    pii_tokens=pii_tokens,
                )
                return anonymized_text
            return redacted_text["text"]
        except Exception as e:
            status = "guardrail_failed_to_respond"
            exception_str = str(e)
            raise e
        finally:
            ####################################################
            # Create Guardrail Trace for logging on Langfuse, Datadog, etc.
            ####################################################
            guardrail_json_response: Union[Exception, str, dict, List[dict]] = {}
            if status == "success":
                if isinstance(analyze_results, List):
                    guardrail_json_response = [dict(item) for item in analyze_results]
            else:
                guardrail_json_response = exception_str
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider=self.guardrail_provider,
                guardrail_json_response=guardrail_json_response,
                request_data=request_data,
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
                masked_entity_count=masked_entity_count,
            )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        """
        - Check if request turned off pii
            - Check if user allowed to turn off pii (key permissions -> 'allow_pii_controls')

        - Take the request data
        - Call /analyze -> get the results
        - Call /anonymize w/ the analyze results -> get the redacted text

        For multiple messages in /chat/completions, we'll need to call them in parallel.
        """

        try:
            request_pii_tokens: Dict[str, str] = {}
            content_safety = data.get("content_safety", None)
            verbose_proxy_logger.debug("content_safety: %s", content_safety)
            presidio_config = self.get_presidio_settings_from_request_data(data)
            messages = data.get("messages", None)
            if messages is None:
                return data
            tasks = []
            task_mappings: List[
                Tuple[int, Optional[int]]
            ] = []  # Track (message_index, content_index) for each task

            for msg_idx, m in enumerate(messages):
                content = m.get("content", None)
                if content is None:
                    continue
                if isinstance(content, str):
                    tasks.append(
                        self.check_pii(
                            text=content,
                            output_parse_pii=self.output_parse_pii,
                            presidio_config=presidio_config,
                            request_data=data,
                            pii_tokens=request_pii_tokens,
                        )
                    )
                    task_mappings.append(
                        (msg_idx, None)
                    )  # None indicates string content
                elif isinstance(content, list):
                    for content_idx, c in enumerate(content):
                        text_str = c.get("text", None)
                        if text_str is None:
                            continue
                        tasks.append(
                            self.check_pii(
                                text=text_str,
                                output_parse_pii=self.output_parse_pii,
                                presidio_config=presidio_config,
                                request_data=data,
                                pii_tokens=request_pii_tokens,
                            )
                        )
                        task_mappings.append((msg_idx, int(content_idx)))

            responses = await asyncio.gather(*tasks)

            # Map responses back to the correct message and content item
            for task_idx, r in enumerate(responses):
                mapping = task_mappings[task_idx]
                msg_idx = cast(int, mapping[0])
                content_idx_optional = cast(Optional[int], mapping[1])
                content = messages[msg_idx].get("content", None)
                if content is None:
                    continue
                if isinstance(content, str) and content_idx_optional is None:
                    messages[msg_idx][
                        "content"
                    ] = r  # replace content with redacted string
                elif isinstance(content, list) and content_idx_optional is not None:
                    messages[msg_idx]["content"][content_idx_optional]["text"] = r

            verbose_proxy_logger.debug(
                f"Presidio PII Masking: Redacted pii message: {data['messages']}"
            )
            data["messages"] = messages
            # Store pii_tokens in request data so post_call can unmask using the same
            # request's mappings (guardrail instance is shared across requests).
            if self.output_parse_pii and request_pii_tokens:
                data.setdefault("_presidio_pii_tokens", {})[self.guardrail_name] = dict(
                    request_pii_tokens
                )
            return data
        except Exception as e:
            raise e

    def logging_hook(
        self, kwargs: dict, result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        from concurrent.futures import ThreadPoolExecutor

        def run_in_new_loop():
            """Run the coroutine in a new event loop within this thread."""
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(
                    self.async_logging_hook(
                        kwargs=kwargs, result=result, call_type=call_type
                    )
                )
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

        try:
            # First, try to get the current event loop
            _ = asyncio.get_running_loop()
            # If we're already in an event loop, run in a separate thread
            # to avoid nested event loop issues
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_new_loop)
                return future.result()

        except RuntimeError:
            # No running event loop, we can safely run in this thread
            return run_in_new_loop()

    async def async_logging_hook(
        self, kwargs: dict, result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """
        Masks the input before logging to langfuse, datadog, etc.
        """
        if (
            call_type == "completion" or call_type == "acompletion"
        ):  # /chat/completions requests
            messages: Optional[List] = kwargs.get("messages", None)
            tasks = []
            task_mappings: List[
                Tuple[int, Optional[int]]
            ] = []  # Track (message_index, content_index) for each task

            if messages is None:
                return kwargs, result

            presidio_config = self.get_presidio_settings_from_request_data(kwargs)

            for msg_idx, m in enumerate(messages):
                content = m.get("content", None)
                if content is None:
                    continue
                if isinstance(content, str):
                    tasks.append(
                        self.check_pii(
                            text=content,
                            output_parse_pii=False,
                            presidio_config=presidio_config,
                            request_data=kwargs,
                        )
                    )  # need to pass separately b/c presidio has context window limits
                    task_mappings.append(
                        (msg_idx, None)
                    )  # None indicates string content
                elif isinstance(content, list):
                    for content_idx, c in enumerate(content):
                        text_str = c.get("text", None)
                        if text_str is None:
                            continue
                        tasks.append(
                            self.check_pii(
                                text=text_str,
                                output_parse_pii=False,
                                presidio_config=presidio_config,
                                request_data=kwargs,
                            )
                        )
                        task_mappings.append((msg_idx, int(content_idx)))

            responses = await asyncio.gather(*tasks)

            # Map responses back to the correct message and content item
            for task_idx, r in enumerate(responses):
                mapping = task_mappings[task_idx]
                msg_idx = cast(int, mapping[0])
                content_idx_optional = cast(Optional[int], mapping[1])
                content = messages[msg_idx].get("content", None)
                if content is None:
                    continue
                if isinstance(content, str) and content_idx_optional is None:
                    messages[msg_idx][
                        "content"
                    ] = r  # replace content with redacted string
                elif isinstance(content, list) and content_idx_optional is not None:
                    messages[msg_idx]["content"][content_idx_optional]["text"] = r

            verbose_proxy_logger.debug(
                f"Presidio PII Masking: Redacted pii message: {messages}"
            )
            kwargs["messages"] = messages

        return kwargs, result

    async def async_post_call_success_hook(  # type: ignore
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[ModelResponse, EmbeddingResponse, ImageResponse],
    ):
        """
        Output parse the response object to replace the masked tokens with user sent values
        """
        verbose_proxy_logger.debug(
            f"PII Masking Args: self.output_parse_pii={self.output_parse_pii}; type of response={type(response)}"
        )

        if self.apply_to_output is True:
            if self._is_anthropic_message_response(response):
                return await self._process_anthropic_response_for_pii(
                    response=cast(dict, response), request_data=data, mode="mask"
                )
            return await self._mask_output_response(
                response=response, request_data=data
            )

        if self.output_parse_pii is False and litellm.output_parse_pii is False:
            return response

        # Use only request-scoped pii_tokens; do not fall back to self.pii_tokens
        # or we may use stale mappings from a previous request (e.g. wrong name).
        pii_tokens = data.get("_presidio_pii_tokens", {}).get(self.guardrail_name, {})
        if not pii_tokens:
            return response

        attempted_non_streaming_unmask = False
        defer_cleanup_to_streaming_hook = isinstance(response, ModelResponseStream) or (
            isinstance(response, ModelResponse)
            and len(response.choices) > 0
            and isinstance(response.choices[0], StreamingChoices)
        )
        try:
            if isinstance(response, ModelResponse) and not isinstance(
                response.choices[0], StreamingChoices
            ):  # /chat/completions requests
                attempted_non_streaming_unmask = True
                await self._process_response_for_pii(
                    response=response,
                    request_data=data,
                    mode="unmask",
                )
            elif self._is_anthropic_message_response(response):
                attempted_non_streaming_unmask = True
                await self._process_anthropic_response_for_pii(
                    response=cast(dict, response), request_data=data, mode="unmask"
                )
        finally:
            if attempted_non_streaming_unmask or not defer_cleanup_to_streaming_hook:
                self._clear_request_scoped_pii_tokens(data)
        return response

    @staticmethod
    def _unmask_pii_text(text: str, pii_tokens: Dict[str, str]) -> str:
        """
        Replace PII tokens in *text* with their original values.

        Includes a fallback for tokens that were truncated by ``max_tokens``:
        if the *end* of ``text`` matches the *beginning* of a token and the
        overlap is long enough, the truncated suffix is replaced with the
        original value.  The minimum overlap length is
        ``min(20, len(token) // 2)`` to reduce the risk of false positives
        when multiple tokens share a common prefix.
        """
        return _replace_pii_tokens_in_text(text, pii_tokens)

    @staticmethod
    def _is_anthropic_message_response(response: Any) -> bool:
        """Check if the response is an Anthropic native message dict."""
        return (
            isinstance(response, dict)
            and response.get("type") == "message"
            and isinstance(response.get("content"), list)
        )

    async def _process_anthropic_response_for_pii(
        self,
        response: dict,
        request_data: dict,
        mode: Literal["mask", "unmask"],
    ) -> dict:
        """
        Process an Anthropic native message dict for PII masking/unmasking.
        Handles content blocks with type == "text".
        """
        pii_tokens = (
            request_data.get("_presidio_pii_tokens", {}).get(self.guardrail_name, {})
            if request_data
            else {}
        )
        if not pii_tokens and mode == "unmask":
            verbose_proxy_logger.debug(
                "No pii_tokens found in request_data — nothing to unmask (anthropic response)"
            )
        presidio_config = self.get_presidio_settings_from_request_data(
            request_data or {}
        )

        content = response.get("content")
        if not isinstance(content, list):
            return response

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text_value = block.get("text")
            if text_value is None:
                continue
            if mode == "unmask":
                block["text"] = self._unmask_pii_text(text_value, pii_tokens)
            elif mode == "mask":
                block["text"] = await self.check_pii(
                    text=text_value,
                    output_parse_pii=False,
                    presidio_config=presidio_config,
                    request_data=request_data,
                )

        return response

    async def _process_response_for_pii(
        self,
        response: ModelResponse,
        request_data: dict,
        mode: Literal["mask", "unmask"],
    ) -> ModelResponse:
        """
        Helper to recursively process a ModelResponse for PII.
        Handles all choices and tool calls.
        """
        pii_tokens = (
            request_data.get("_presidio_pii_tokens", {}).get(self.guardrail_name, {})
            if request_data
            else {}
        )
        if not pii_tokens and mode == "unmask":
            verbose_proxy_logger.debug(
                "No pii_tokens found in request_data — nothing to unmask"
            )
        presidio_config = self.get_presidio_settings_from_request_data(
            request_data or {}
        )

        for choice in response.choices:
            message = getattr(choice, "message", None)
            if message is None:
                continue

            # 1. Process content
            content = getattr(message, "content", None)
            if isinstance(content, str):
                if mode == "unmask":
                    message.content = _replace_pii_tokens_in_text(content, pii_tokens)
                elif mode == "mask":
                    message.content = await self.check_pii(
                        text=content,
                        output_parse_pii=False,
                        presidio_config=presidio_config,
                        request_data=request_data,
                    )
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text_value = item.get("text")
                    if text_value is None:
                        continue
                    if mode == "unmask":
                        item["text"] = _replace_pii_tokens_in_text(
                            text_value, pii_tokens
                        )
                    elif mode == "mask":
                        item["text"] = await self.check_pii(
                            text=text_value,
                            output_parse_pii=False,
                            presidio_config=presidio_config,
                            request_data=request_data,
                        )

            # 2. Process tool calls
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    function = getattr(tool_call, "function", None)
                    if function and hasattr(function, "arguments"):
                        args = function.arguments
                        if isinstance(args, str):
                            if mode == "unmask":
                                function.arguments = _replace_pii_tokens_in_text(
                                    args, pii_tokens
                                )
                            elif mode == "mask":
                                function.arguments = await self.check_pii(
                                    text=args,
                                    output_parse_pii=False,
                                    presidio_config=presidio_config,
                                    request_data=request_data,
                                )

            # 3. Process legacy function calls
            function_call = getattr(message, "function_call", None)
            if function_call and hasattr(function_call, "arguments"):
                args = function_call.arguments
                if isinstance(args, str):
                    if mode == "unmask":
                        function_call.arguments = _replace_pii_tokens_in_text(
                            args, pii_tokens
                        )
                    elif mode == "mask":
                        function_call.arguments = await self.check_pii(
                            text=args,
                            output_parse_pii=False,
                            presidio_config=presidio_config,
                            request_data=request_data,
                        )
        return response

    async def _mask_output_response(
        self,
        response: Union[ModelResponse, EmbeddingResponse, ImageResponse],
        request_data: dict,
    ):
        """
        Apply Presidio masking on model responses (non-streaming).
        """
        if not isinstance(response, ModelResponse):
            return response

        # skip streaming here; handled in async_post_call_streaming_iterator_hook
        if isinstance(response, ModelResponseStream):
            return response

        await self._process_response_for_pii(
            response=response,
            request_data=request_data,
            mode="mask",
        )
        return response

    async def _stream_apply_output_masking(
        self,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Union[ModelResponseStream, bytes], None]:
        """Apply Presidio masking to streaming output (apply_to_output=True path)."""
        from litellm.llms.base_llm.base_model_iterator import (
            convert_model_response_to_streaming,
        )
        from litellm.main import stream_chunk_builder
        from litellm.types.utils import ModelResponse

        all_chunks: List[ModelResponseStream] = []
        try:
            async for chunk in response:
                if isinstance(chunk, ModelResponseStream):
                    all_chunks.append(chunk)
                elif isinstance(chunk, bytes):
                    yield chunk  # type: ignore[misc]
                    continue

            if not all_chunks:
                verbose_proxy_logger.warning(
                    "Presidio apply_to_output: streaming response contained only "
                    "bytes chunks (Anthropic native SSE). Output PII masking was "
                    "skipped for this response."
                )
                return

            assembled_model_response = stream_chunk_builder(
                chunks=all_chunks, messages=request_data.get("messages")
            )

            if not isinstance(assembled_model_response, ModelResponse):
                for chunk in all_chunks:
                    yield chunk
                return

            await self._process_response_for_pii(
                response=assembled_model_response,
                request_data=request_data,
                mode="mask",
            )

            mock_response_stream = convert_model_response_to_streaming(
                assembled_model_response
            )
            yield mock_response_stream

        except Exception as e:
            verbose_proxy_logger.error(f"Error masking streaming PII output: {str(e)}")
            for chunk in all_chunks:
                yield chunk

    async def _stream_pii_unmasking(
        self,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Union[ModelResponseStream, bytes], None]:
        """Apply PII unmasking to streaming output (output_parse_pii=True path)."""
        from litellm.llms.base_llm.base_model_iterator import (
            convert_model_response_to_streaming,
        )
        from litellm.main import stream_chunk_builder
        from litellm.types.utils import ModelResponse

        remaining_chunks: List[ModelResponseStream] = []
        try:
            async for chunk in response:
                if isinstance(chunk, ModelResponseStream):
                    remaining_chunks.append(chunk)
                elif isinstance(chunk, bytes):
                    yield chunk  # type: ignore[misc]
                    continue

            if not remaining_chunks:
                return

            assembled_model_response = stream_chunk_builder(
                chunks=remaining_chunks, messages=request_data.get("messages")
            )

            if not isinstance(assembled_model_response, ModelResponse):
                for chunk in remaining_chunks:
                    yield chunk
                return

            self._preserve_usage_from_last_chunk(
                assembled_model_response, remaining_chunks
            )

            await self._process_response_for_pii(
                response=assembled_model_response,
                request_data=request_data,
                mode="unmask",
            )

            mock_response_stream = convert_model_response_to_streaming(
                assembled_model_response
            )
            yield mock_response_stream

        except Exception as e:
            verbose_proxy_logger.error(f"Error in PII streaming processing: {str(e)}")
            for chunk in remaining_chunks:
                yield chunk
        finally:
            self._clear_request_scoped_pii_tokens(request_data)

    async def async_post_call_streaming_iterator_hook(  # type: ignore[override]
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Union[ModelResponseStream, bytes], None]:
        """
        Process streaming response chunks to unmask PII tokens when needed.

        Note: the return type includes `bytes` because Anthropic native SSE
        streaming sends raw bytes chunks that pass through untransformed.
        The base class declares ModelResponseStream only.
        """
        if self.apply_to_output:
            async for chunk in self._stream_apply_output_masking(
                response, request_data
            ):
                yield chunk
            return

        pii_tokens = (
            request_data.get("_presidio_pii_tokens", {}).get(self.guardrail_name, {})
            if request_data
            else {}
        )
        if not pii_tokens and request_data:
            verbose_proxy_logger.debug(
                "No pii_tokens found in request_data for streaming unmask path"
            )
        if not (self.output_parse_pii and pii_tokens):
            async for chunk in response:
                yield chunk
            return

        async for chunk in self._stream_pii_unmasking(response, request_data):
            yield chunk

    @staticmethod
    def _preserve_usage_from_last_chunk(
        assembled_model_response: Any,
        chunks: List[Any],
    ) -> None:
        """Copy usage metadata from the last chunk when stream_chunk_builder misses it."""
        if not getattr(assembled_model_response, "usage", None) and chunks:
            last_chunk_usage = getattr(chunks[-1], "usage", None)
            if last_chunk_usage:
                setattr(assembled_model_response, "usage", last_chunk_usage)

    def get_presidio_settings_from_request_data(
        self, data: dict
    ) -> Optional[PresidioPerRequestConfig]:
        if "metadata" in data:
            _metadata = data.get("metadata", None)
            if _metadata is None:
                return None
            _guardrail_config = _metadata.get("guardrail_config")
            if _guardrail_config:
                _presidio_config = PresidioPerRequestConfig(**_guardrail_config)
                return _presidio_config

        return None

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except Exception:
            pass

    def _clear_request_scoped_pii_tokens(self, data: dict) -> None:
        pii_token_map = data.get("_presidio_pii_tokens")
        if not isinstance(pii_token_map, dict):
            return
        pii_token_map.pop(self.guardrail_name, None)
        if not pii_token_map:
            data.pop("_presidio_pii_tokens", None)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: "GenericGuardrailAPIInputs",
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> "GenericGuardrailAPIInputs":
        """
        UI will call this function to check:
            1. If the connection to the guardrail is working
            2. When Testing the guardrail with some text, this function will be called with the input text and returns a text after applying the guardrail
        """
        texts = inputs.get("texts", [])

        request_pii_tokens: Dict[str, str] = {}
        response_pii_tokens = (
            request_data.get("_presidio_pii_tokens", {}).get(self.guardrail_name, {})
            if request_data
            else {}
        )
        new_texts = []
        if input_type == "response" and response_pii_tokens:
            for text in texts:
                new_texts.append(_replace_pii_tokens_in_text(text, response_pii_tokens))
            if request_data is not None:
                self._clear_request_scoped_pii_tokens(request_data)
        else:
            for text in texts:
                modified_text = await self.check_pii(
                    text=text,
                    output_parse_pii=self.output_parse_pii,
                    presidio_config=None,
                    request_data=request_data or {},
                    pii_tokens=request_pii_tokens,
                )
                new_texts.append(modified_text)
        inputs["texts"] = new_texts
        # When using unified guardrail path, pre_call uses apply_guardrail instead of
        # async_pre_call_hook; store pii_tokens in request_data so post_call can unmask.
        if (
            input_type == "request"
            and self.output_parse_pii
            and request_pii_tokens
            and request_data is not None
        ):
            request_data.setdefault("_presidio_pii_tokens", {})[
                self.guardrail_name
            ] = dict(request_pii_tokens)
        return inputs

    def update_in_memory_litellm_params(self, litellm_params: LitellmParams) -> None:
        """
        Update the guardrails litellm params in memory
        """
        super().update_in_memory_litellm_params(litellm_params)
        if litellm_params.pii_entities_config:
            self.pii_entities_config = litellm_params.pii_entities_config
        if litellm_params.presidio_score_thresholds:
            self.presidio_score_thresholds = litellm_params.presidio_score_thresholds
        if litellm_params.presidio_entities_deny_list:
            self.presidio_entities_deny_list = (
                litellm_params.presidio_entities_deny_list
            )
