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
from contextlib import asynccontextmanager
from datetime import datetime
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
        self.pii_tokens: dict = (
            {}
        )  # mapping of PII token to original text - only used with Presidio `replace` operation
        self.mock_redacted_text = mock_redacted_text
        self.output_parse_pii = output_parse_pii or False
        self.apply_to_output = apply_to_output

        # When output_parse_pii or apply_to_output is enabled, the guardrail must
        # also run on post_call to unmask/mask the response.  Expand the event_hook
        # so should_run_guardrail returns True for both pre_call and post_call.
        if (self.output_parse_pii or self.apply_to_output) and not logging_only:
            current_hook = self.event_hook
            if isinstance(current_hook, str) and current_hook != "post_call":
                self.event_hook = cast(
                    List[GuardrailEventHooks], [current_hook, "post_call"]
                )
            elif isinstance(current_hook, list) and "post_call" not in current_hook:
                self.event_hook = cast(
                    List[GuardrailEventHooks], current_hook + ["post_call"]
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

    async def _post_presidio_anonymize(self, text: str, analyze_results: Any) -> Any:
        """POST to Presidio anonymize; returns parsed JSON body."""
        # Use shared session to prevent memory leak (issue #14540)
        async with self._get_session_iterator() as session:
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
                if response.status >= 400:
                    error_body = await response.text()
                    raise Exception(
                        f"Presidio anonymizer returned HTTP {response.status}: {error_body[:200]}"
                    )
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
                return await response.json()

    def _finalize_presidio_anonymize_simple(
        self,
        redacted_text: Dict[str, Any],
        masked_entity_count: Dict[str, int],
    ) -> str:
        # No need to build numbered tokens — just use Presidio's
        # already-anonymized text directly.  The old code incorrectly
        # applied anonymizer item positions (which reference the
        # *output* text) to the *original* text, causing offset errors.
        for item in redacted_text.get("items", []):
            entity_type = item.get("entity_type", None)
            if entity_type is not None:
                masked_entity_count[entity_type] = (
                    masked_entity_count.get(entity_type, 0) + 1
                )
        return redacted_text["text"]

    def _finalize_presidio_anonymize_numbered_tokens(
        self,
        text: str,
        analyze_results: Any,
        request_data: Optional[Dict],
        masked_entity_count: Dict[str, int],
    ) -> str:
        # output_parse_pii is True — we need sequentially numbered
        # tokens and a pii_tokens mapping for later unmasking.
        # Use analyze_results positions (which reference the ORIGINAL
        # text) instead of anonymizer items (which reference the output).
        new_text = text
        if request_data is None:
            verbose_proxy_logger.warning(
                "Presidio anonymize_text called without request_data — "
                "PII tokens cannot be stored per-request. "
                "This may indicate a missing caller update."
            )
            request_data = {}
        if not request_data.get("metadata"):
            request_data["metadata"] = {}
        if "pii_tokens" not in request_data["metadata"]:
            request_data["metadata"]["pii_tokens"] = {}
        pii_tokens = request_data["metadata"]["pii_tokens"]

        # Assign sequence numbers in forward (left-to-right) order so
        # that <PERSON_1> is the first entity in the text, etc.
        sorted_forward = sorted(analyze_results, key=lambda x: x["start"])
        seq_map = {}
        for idx, ar in enumerate(sorted_forward, start=1):
            seq_map[(ar["start"], ar["end"])] = idx

        # Apply replacements in reverse order by start position so
        # that replacing later spans first does not shift earlier
        # coordinates in the original text.
        for ar in reversed(sorted_forward):
            start = ar["start"]
            end = ar["end"]
            entity_type = ar["entity_type"]
            replacement = f"<{entity_type}>"
            seq = seq_map[(start, end)]
            if replacement.endswith(">"):
                replacement = f"{replacement[:-1]}_{seq}>"
            else:
                replacement = f"{replacement}_{seq}"
            pii_tokens[replacement] = text[start:end]
            new_text = new_text[:start] + replacement + new_text[end:]
            masked_entity_count[entity_type] = (
                masked_entity_count.get(entity_type, 0) + 1
            )
        return new_text

    async def anonymize_text(
        self,
        text: str,
        analyze_results: Any,
        output_parse_pii: bool,
        masked_entity_count: Dict[str, int],
        request_data: Optional[Dict] = None,
    ) -> str:
        """
        Send analysis results to the Presidio anonymizer endpoint to get redacted text
        """
        try:
            # If there are no detections after filtering, return the original text
            if isinstance(analyze_results, list) and len(analyze_results) == 0:
                return text

            redacted_text = await self._post_presidio_anonymize(text, analyze_results)
            if redacted_text is None:
                raise Exception("Invalid anonymizer response: received None")

            verbose_proxy_logger.debug("redacted_text: %s", redacted_text)

            if not output_parse_pii:
                return self._finalize_presidio_anonymize_simple(
                    redacted_text, masked_entity_count
                )

            return self._finalize_presidio_anonymize_numbered_tokens(
                text, analyze_results, request_data, masked_entity_count
            )
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
                    request_data=request_data,
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
            content_safety = data.get("content_safety", None)
            verbose_proxy_logger.debug("content_safety: %s", content_safety)
            presidio_config = self.get_presidio_settings_from_request_data(data)
            messages = data.get("messages", None)
            if messages is None:
                return data
            tasks = []
            task_mappings: List[Tuple[int, Optional[int]]] = (
                []
            )  # Track (message_index, content_index) for each task

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
            task_mappings: List[Tuple[int, Optional[int]]] = (
                []
            )  # Track (message_index, content_index) for each task

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

        if isinstance(response, ModelResponse) and not isinstance(
            response.choices[0], StreamingChoices
        ):  # /chat/completions requests
            await self._process_response_for_pii(
                response=response,
                request_data=data,
                mode="unmask",
            )
        elif self._is_anthropic_message_response(response):
            await self._process_anthropic_response_for_pii(
                response=cast(dict, response), request_data=data, mode="unmask"
            )
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
        for token, original_text in pii_tokens.items():
            if token in text:
                text = text.replace(token, original_text)
            else:
                # FALLBACK: Handle truncated tokens (token cut off by max_tokens)
                # Only check at the very end of the text.
                min_overlap = min(20, len(token) // 2)
                for i in range(max(0, len(text) - len(token)), len(text)):
                    sub = text[i:]
                    if token.startswith(sub) and len(sub) >= min_overlap:
                        text = text[:i] + original_text
                        break
        return text

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
        metadata = (request_data.get("metadata") or {}) if request_data else {}
        pii_tokens = metadata.get("pii_tokens", {})
        if not pii_tokens and mode == "unmask":
            verbose_proxy_logger.debug(
                "No pii_tokens in metadata for Anthropic response unmask"
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
        metadata = (request_data.get("metadata") or {}) if request_data else {}
        pii_tokens = metadata.get("pii_tokens", {})
        if not pii_tokens and mode == "unmask":
            verbose_proxy_logger.debug(
                "No pii_tokens found in request_data['metadata'] — nothing to unmask"
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
                    message.content = self._unmask_pii_text(content, pii_tokens)
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
                        item["text"] = self._unmask_pii_text(text_value, pii_tokens)
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
                                function.arguments = self._unmask_pii_text(
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
                        function_call.arguments = self._unmask_pii_text(
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

    # Window used to flush buffered text for incremental Presidio masking on
    # streaming outputs. Tuned to keep TTFT low while still giving Presidio
    # enough context to detect multi-word PII entities. ``_PRESIDIO_TAIL_OVERLAP``
    # is the rolling tail retained across flushes so a PII span crossing a
    # window boundary is still re-analyzed in the next call.
    _PRESIDIO_STREAM_FLUSH_CHARS = 120
    _PRESIDIO_TAIL_OVERLAP = 32

    @staticmethod
    def _build_text_only_chunk(
        template: "ModelResponseStream",
        choice_index: int,
        content: str,
    ) -> "ModelResponseStream":
        """Construct a synthetic streaming chunk that carries only a text delta.

        Used to flush trailing buffered text after the upstream stream has
        finished but the per-choice text buffer still holds characters that
        were withheld for boundary-safe masking/unmasking.
        """
        from litellm.types.utils import (
            Delta,
            ModelResponseStream,
            StreamingChoices,
        )

        return ModelResponseStream(
            id=template.id,
            object="chat.completion.chunk",
            created=template.created,
            model=template.model,
            choices=[
                StreamingChoices(
                    index=choice_index,
                    delta=Delta(content=content),
                )
            ],
        )

    @staticmethod
    def _safe_unmask_prefix_len(
        buf: str,
        pii_tokens: "Dict[str, str]",
        max_token_len: int,
    ) -> int:
        """Return the largest prefix length of ``buf`` we can safely emit
        without splitting a pii token across the prefix/tail boundary.

        The prefix is shrunk further if it ends with a non-trivial prefix of
        any known token (which would indicate a token still mid-arrival in
        the next chunk). Otherwise the trailing ``max_token_len`` characters
        are held back, which is sufficient when the buffer does not end in a
        partial token prefix.
        """
        if not buf:
            return 0
        candidate = len(buf) - max_token_len
        if candidate <= 0:
            return 0
        # Iteratively walk candidate back while it ends in a partial token prefix.
        # In the worst case this only shrinks once per token, so total work is
        # bounded by O(sum(len(token))) per call.
        while candidate > 0:
            safe_view = buf[:candidate]
            shrunk = False
            for token in pii_tokens:
                tlen = len(token)
                if tlen <= 1:
                    continue
                # Longest suffix of safe_view that matches a non-trivial prefix
                # of ``token``.
                max_k = min(len(safe_view), tlen - 1)
                for k in range(max_k, 0, -1):
                    if safe_view.endswith(token[:k]):
                        candidate -= k
                        shrunk = True
                        break
                if shrunk:
                    break
            if not shrunk:
                break
        return max(0, candidate)

    async def _emit_masked_tool_call_args(
        self,
        template: "ModelResponseStream",
        tool_arg_buffer: "Dict[Tuple[int, int], str]",
        mask_fn,
    ):
        """Emit a synthetic tool-call chunk with masked arguments for each
        (choice_index, tool_call_index) entry in ``tool_arg_buffer`` and
        clear the buffer.

        We send one chunk per tool call so the client reassembles each set of
        arguments correctly. The legacy ``function_call`` slot
        (``tool_call_index == -1``) is emitted on ``delta.function_call``.

        ``mask_fn`` may be sync or async — we await coroutine results when
        the unmask path passes a plain ``lambda``.
        """
        import inspect

        from litellm.types.utils import (
            ChatCompletionDeltaToolCall,
            Delta,
            Function,
            ModelResponseStream,
            StreamingChoices,
        )

        for (choice_idx, tc_idx), args in list(tool_arg_buffer.items()):
            if not args:
                continue
            result = mask_fn(args)
            if inspect.isawaitable(result):
                masked = await result
            else:
                masked = result
            if tc_idx == -1:
                delta = Delta(function_call=Function(arguments=masked))
            else:
                delta = Delta(
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            index=tc_idx,
                            function=Function(arguments=masked),
                        )
                    ]
                )
            yield ModelResponseStream(
                id=template.id,
                object="chat.completion.chunk",
                created=template.created,
                model=template.model,
                choices=[StreamingChoices(index=choice_idx, delta=delta)],
            )
            tool_arg_buffer[(choice_idx, tc_idx)] = ""

    @staticmethod
    def _find_safe_flush_boundary(text: str, max_window: int) -> int:
        """Return the index (exclusive) of a safe place to cut ``text`` for flushing.

        Prefers whitespace or sentence-ending punctuation inside the first
        ``max_window`` characters so multi-word entities aren't split across
        a flush boundary. Returns 0 if no such boundary is found.
        """
        boundary_chars = " \t\n.!?,;:"
        upper = min(len(text), max_window)
        for i in range(upper, 0, -1):
            if text[i - 1] in boundary_chars:
                return i
        return 0

    async def _mask_emit_safe_prefix(
        self,
        buf: str,
        flush_chars: int,
        tail_overlap: int,
        mask_fn,
    ):
        """Compute the safe prefix to flush, mask it, and return (masked, remainder)."""
        flushable = buf[:-tail_overlap] if tail_overlap else buf
        safe_len = self._find_safe_flush_boundary(flushable, flush_chars)
        if safe_len == 0:
            safe_len = max(0, len(buf) - tail_overlap)
        if not safe_len:
            return None, buf
        masked = await mask_fn(buf[:safe_len])
        return masked, buf[safe_len:]

    async def _mask_apply_to_choice_delta(
        self,
        choice,
        buf: str,
        had_text: bool,
        is_final: bool,
        flush_chars: int,
        tail_overlap: int,
        mask_fn,
    ) -> str:
        """Apply incremental masking to a single ``StreamingChoices`` delta in
        place. Returns the new per-choice buffer after handling this chunk."""
        delta = choice.delta
        if is_final:
            if buf:
                delta.content = await mask_fn(buf)
                return ""
            return buf
        if had_text and len(buf) >= flush_chars + tail_overlap:
            masked, remainder = await self._mask_emit_safe_prefix(
                buf, flush_chars, tail_overlap, mask_fn
            )
            if masked is not None:
                delta.content = masked
                return remainder
            delta.content = ""
            return buf
        if had_text:
            delta.content = ""
        return buf

    @staticmethod
    def _accumulate_tool_call_arg_fragments(
        delta: Any,
        tool_arg_buffer: Dict[Tuple[int, int], str],
        choice_index: int,
    ) -> None:
        """Strip in-flight ``function.arguments`` fragments from ``delta`` and
        accumulate them in ``tool_arg_buffer`` so they can be masked / unmasked
        as a single coherent string at end-of-stream.

        Other fields on the tool-call delta (``id``, ``type``, ``function.name``)
        are left intact so the client still sees the OpenAI-shaped tool-call
        lifecycle (call start, name, argument fragments redacted, end).
        """
        tool_calls = getattr(delta, "tool_calls", None) or []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            args = getattr(fn, "arguments", None)
            if args:
                key = (choice_index, getattr(tc, "index", 0) or 0)
                tool_arg_buffer[key] = tool_arg_buffer.get(key, "") + args
                fn.arguments = ""
        legacy = getattr(delta, "function_call", None)
        if legacy is not None:
            args = getattr(legacy, "arguments", None)
            if args:
                # Use a synthetic tool index of -1 for the legacy function_call slot.
                key = (choice_index, -1)
                tool_arg_buffer[key] = tool_arg_buffer.get(key, "") + args
                legacy.arguments = ""

    async def _mask_process_model_chunk(
        self,
        chunk: "ModelResponseStream",
        text_buffer: Dict[int, str],
        tool_arg_buffer: Dict[Tuple[int, int], str],
        flush_chars: int,
        tail_overlap: int,
        mask_fn,
    ) -> None:
        """Apply incremental Presidio masking to every choice on ``chunk``.

        Mutates the chunk's ``delta.content`` in place and updates the per-
        choice ``text_buffer`` rolling window. Also strips streamed
        ``tool_calls`` / ``function_call`` argument fragments into
        ``tool_arg_buffer`` so they are masked once the call completes (see
        ``_emit_masked_tool_calls``).
        """
        for choice in chunk.choices or []:
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            is_final = getattr(choice, "finish_reason", None) is not None
            if content:
                text_buffer[choice.index] = text_buffer.get(choice.index, "") + content
            text_buffer[choice.index] = await self._mask_apply_to_choice_delta(
                choice,
                text_buffer.get(choice.index, ""),
                bool(content),
                is_final,
                flush_chars,
                tail_overlap,
                mask_fn,
            )
            self._accumulate_tool_call_arg_fragments(
                delta, tool_arg_buffer, choice.index
            )

    def _flush_buffer_unmasked(
        self,
        template: "ModelResponseStream",
        text_buffer: Dict[int, str],
    ):
        """Yield synthetic chunks for any buffered text, unmasked, in choice order.

        Used when a mixed stream (ModelResponseStream + unknown event) forces
        us off the masking pipeline; we cannot reconstruct Presidio context
        cleanly, so we flush whatever is in the buffer as-is and let the
        operator know via the “mixed stream detected” warning. The buffer
        is cleared in place.
        """
        flushed = []
        for idx, remaining in list(text_buffer.items()):
            if remaining:
                flushed.append(self._build_text_only_chunk(template, idx, remaining))
                text_buffer[idx] = ""
        return flushed

    async def _stream_apply_output_masking(
        self,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Union[ModelResponseStream, bytes], None]:
        """Apply Presidio masking to streaming output incrementally.

        For each ``ModelResponseStream`` chunk:
          * Non-text deltas (role-only, ``tool_calls``, ``finish_reason``,
            ``function_call``) are forwarded immediately so the client sees the
            full OpenAI streaming lifecycle without buffering.
          * Text deltas are appended to a per-choice rolling buffer. When the
            buffer exceeds ``_PRESIDIO_STREAM_FLUSH_CHARS + _PRESIDIO_TAIL_OVERLAP``
            (or the chunk carries a ``finish_reason``), the safe prefix is sent
            through Presidio analyze + anonymize and emitted in-place of the
            current chunk's ``delta.content``. The tail is retained so the next
            window can still detect a PII span that crossed the flush point.
          * ``bytes`` chunks (Anthropic native SSE) and unknown event objects
            (e.g. ``/v1/responses`` events) pass through untouched. If text was
            already buffered when an unknown event arrives, the buffered text
            is flushed unmasked because partial Presidio context cannot
            safely be reconstructed downstream.

        This replaces the prior ``stream_chunk_builder`` +
        ``convert_model_response_to_streaming`` pipeline which emitted a
        single SSE chunk at end-of-stream and made TTFT equal to total
        generation time.
        """
        from litellm.types.utils import ModelResponseStream

        presidio_config = self.get_presidio_settings_from_request_data(request_data)
        flush_chars = self._PRESIDIO_STREAM_FLUSH_CHARS
        tail_overlap = self._PRESIDIO_TAIL_OVERLAP

        text_buffer: Dict[int, str] = {}
        # (choice_index, tool_call_index) → accumulated arguments fragment.
        # tool_call_index == -1 is reserved for the legacy ``function_call`` slot.
        tool_arg_buffer: Dict[Tuple[int, int], str] = {}
        template: Optional[ModelResponseStream] = None
        saw_model_response = False
        saw_unknown_event = False
        passthrough_active = False  # Set after an unknown event — all subsequent
        # ModelResponseStream chunks are forwarded unmasked to match the
        # warning message contract (see Greptile review on PR #29134).

        async def _mask(text: str) -> str:
            try:
                return await self.check_pii(
                    text=text,
                    output_parse_pii=False,
                    presidio_config=presidio_config,
                    request_data=request_data,
                )
            except Exception as exc:  # noqa: BLE001
                verbose_proxy_logger.error(
                    "Presidio mid-stream masking failed; emitting unmasked text: %s",
                    exc,
                )
                return text

        try:
            async for chunk in response:
                if isinstance(chunk, bytes):
                    yield chunk  # type: ignore[misc]
                    continue

                if not isinstance(chunk, ModelResponseStream):
                    if any(text_buffer.values()) and template is not None:
                        verbose_proxy_logger.warning(
                            "Presidio apply_to_output: mixed stream detected "
                            "(ModelResponseStream + unknown event). Flushing "
                            "buffered text without PII masking and switching "
                            "to transparent passthrough."
                        )
                        for flush in self._flush_buffer_unmasked(template, text_buffer):
                            yield flush
                    saw_unknown_event = True
                    passthrough_active = True
                    yield chunk
                    continue

                if passthrough_active:
                    # Honor the "switched to transparent passthrough" warning:
                    # once we have seen an unknown event we cannot reconstruct
                    # the masking context for subsequent ModelResponseStream
                    # chunks, so forward them unmasked.
                    yield chunk
                    continue

                saw_model_response = True
                template = chunk
                await self._mask_process_model_chunk(
                    chunk,
                    text_buffer,
                    tool_arg_buffer,
                    flush_chars,
                    tail_overlap,
                    _mask,
                )
                yield chunk

            if template is not None and not passthrough_active:
                for idx, remaining in text_buffer.items():
                    if remaining:
                        masked = await _mask(remaining)
                        yield self._build_text_only_chunk(template, idx, masked)
                async for tc_chunk in self._emit_masked_tool_call_args(
                    template, tool_arg_buffer, _mask
                ):
                    yield tc_chunk

            if saw_unknown_event:
                verbose_proxy_logger.warning(
                    "Presidio apply_to_output: streaming response contained "
                    "unknown event objects (e.g. /v1/responses events). Output "
                    "PII masking was skipped for this response."
                )
                return

            if not saw_model_response and not saw_unknown_event:
                verbose_proxy_logger.warning(
                    "Presidio apply_to_output: streaming response contained only "
                    "bytes chunks (Anthropic native SSE). Output PII masking was "
                    "skipped for this response."
                )

        except Exception as e:  # noqa: BLE001
            verbose_proxy_logger.error(f"Error masking streaming PII output: {str(e)}")
            if template is not None:
                for idx, remaining in text_buffer.items():
                    if remaining:
                        yield self._build_text_only_chunk(template, idx, remaining)

    async def _stream_pii_unmasking(
        self,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Union[ModelResponseStream, bytes], None]:
        """Apply PII unmasking to streaming output incrementally.

        Unmasking is pure local string replacement against the known
        ``pii_tokens`` dict (no network call), so each chunk is processed as
        it arrives:

          * Non-text deltas pass through immediately.
          * For text deltas we keep a trailing tail equal to the longest known
            token length, so a placeholder token (e.g. ``<PERSON_1>``) split
            across two SSE chunks is still detected and replaced when the
            second half arrives.
          * The final chunk (``finish_reason`` set) or end-of-stream flushes
            the remaining tail through ``_unmask_pii_text``.

        Replaces the prior reassemble-and-emit-one-chunk approach which
        collapsed streaming responses into a single end-of-stream SSE event.
        """
        from litellm.types.utils import ModelResponseStream

        metadata = (request_data.get("metadata") or {}) if request_data else {}
        pii_tokens: Dict[str, str] = metadata.get("pii_tokens") or {}
        max_token_len = max((len(t) for t in pii_tokens.keys()), default=0)

        text_buffer: Dict[int, str] = {}
        tool_arg_buffer: Dict[Tuple[int, int], str] = {}
        template: Optional[ModelResponseStream] = None

        try:
            async for chunk in response:
                if isinstance(chunk, bytes):
                    yield chunk  # type: ignore[misc]
                    continue
                if not isinstance(chunk, ModelResponseStream):
                    if template is not None and any(text_buffer.values()):
                        for idx, remaining in list(text_buffer.items()):
                            if remaining:
                                yield self._build_text_only_chunk(
                                    template,
                                    idx,
                                    self._unmask_pii_text(remaining, pii_tokens),
                                )
                                text_buffer[idx] = ""
                    yield chunk
                    continue

                template = chunk

                for choice in chunk.choices or []:
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    content = getattr(delta, "content", None)
                    is_final = getattr(choice, "finish_reason", None) is not None
                    if content:
                        text_buffer[choice.index] = (
                            text_buffer.get(choice.index, "") + content
                        )
                    buf = text_buffer.get(choice.index, "")

                    if is_final:
                        if buf:
                            delta.content = self._unmask_pii_text(buf, pii_tokens)
                            text_buffer[choice.index] = ""
                    elif content:
                        safe_len = self._safe_unmask_prefix_len(
                            buf, pii_tokens, max_token_len
                        )
                        safe = buf[:safe_len]
                        delta.content = (
                            self._unmask_pii_text(safe, pii_tokens) if safe else ""
                        )
                        text_buffer[choice.index] = buf[safe_len:]
                    self._accumulate_tool_call_arg_fragments(
                        delta, tool_arg_buffer, choice.index
                    )

                yield chunk

            if template is not None:
                for idx, remaining in text_buffer.items():
                    if remaining:
                        yield self._build_text_only_chunk(
                            template,
                            idx,
                            self._unmask_pii_text(remaining, pii_tokens),
                        )
                async for tc_chunk in self._emit_masked_tool_call_args(
                    template,
                    tool_arg_buffer,
                    lambda t: self._unmask_pii_text(t, pii_tokens),
                ):
                    yield tc_chunk

        except Exception as e:  # noqa: BLE001
            verbose_proxy_logger.error(f"Error in PII streaming processing: {str(e)}")
            if template is not None:
                for idx, remaining in text_buffer.items():
                    if remaining:
                        yield self._build_text_only_chunk(template, idx, remaining)

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

        metadata = (request_data.get("metadata") or {}) if request_data else {}
        pii_tokens = metadata.get("pii_tokens", {})
        if not pii_tokens and request_data:
            verbose_proxy_logger.debug(
                "No pii_tokens in request_data['metadata'] for streaming unmask path"
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

        # When input_type is "response" and pii_tokens are available,
        # unmask the text instead of masking it.
        metadata = (request_data.get("metadata") or {}) if request_data else {}
        pii_tokens = metadata.get("pii_tokens", {})

        new_texts = []
        if input_type == "response" and pii_tokens:
            for text in texts:
                new_texts.append(self._unmask_pii_text(text, pii_tokens))
        else:
            for text in texts:
                modified_text = await self.check_pii(
                    text=text,
                    output_parse_pii=self.output_parse_pii,
                    presidio_config=None,
                    request_data=request_data or {},
                )
                new_texts.append(modified_text)
        inputs["texts"] = new_texts
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
