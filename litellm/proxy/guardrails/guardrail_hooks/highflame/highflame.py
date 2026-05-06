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

        All requests are sent to Highflame Shield's POST /v1/guard endpoint.
        ``api_base`` should be a Shield-compatible host — either the public
        Shield API (``https://shield.api.highflame.ai``) or a customer-specific
        Highflame gateway URL. Paths are always ``/v1/guard``; the
        ``api_version`` field is preserved for backward compatibility but is
        not used to construct the request URL (Shield is unversioned at the
        path level).

        See https://docs.highflame.ai/ for details on Shield guard responses
        and per-policy configuration.

        Args:
            api_key: API key for Highflame service.
            api_base: Base URL for Highflame Shield (or a Shield-compatible
                gateway). Defaults to ``https://api.highflame.ai``.
            default_on: Whether the guardrail is enabled by default.
            guardrail_name: Name used within litellm for this guardrail instance.
            highflame_guard_name: Logical Highflame guard name — drives the
                synthesized assessment shape returned to downstream LiteLLM
                code. One of ``trustsafety``, ``promptinjectiondetection``,
                ``lang_detector``, ``dlp_gcp``, ``highflame_guard``.
            api_version: Preserved for backward compatibility. Not used in the
                request URL.
            metadata: Additional metadata to send with requests. If it
                contains ``account_id`` / ``project_id`` (or the
                ``highflame_`` prefixed variants), they are forwarded as
                ``X-Account-ID`` / ``X-Project-ID`` headers to Shield.
            config: Configuration parameters for the guardrail.
            application: Application name for policy-specific guardrails.
                Forwarded as ``x-highflame-application`` header.
        """

        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.highflame_api_key = api_key or get_secret_str("HIGHFLAME_API_KEY")
        self.api_base = (
            api_base
            or get_secret_str("HIGHFLAME_API_BASE")
            or "https://api.highflame.ai"
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

    def _build_shield_headers(self) -> Dict[str, str]:
        """
        Build the headers for a Shield ``/v1/guard`` request.

        ``X-Product: guardrails`` selects the Cedar policy namespace on
        Shield. Tenant headers (``X-Account-ID`` / ``X-Project-ID``) are
        forwarded only when present in ``self.metadata``; Shield will
        otherwise validate based on the API key alone.
        """
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Product": "guardrails",
        }
        if self.highflame_api_key:
            headers["x-highflame-apikey"] = self.highflame_api_key
        if self.application:
            headers["x-highflame-application"] = self.application

        if self.metadata:
            account_id = self.metadata.get("account_id") or self.metadata.get(
                "highflame_account_id"
            )
            project_id = self.metadata.get("project_id") or self.metadata.get(
                "highflame_project_id"
            )
            if account_id:
                headers["X-Account-ID"] = str(account_id)
            if project_id:
                headers["X-Project-ID"] = str(project_id)

        return headers

    def _synthesize_guard_response(
        self, decision: str, reason: str
    ) -> HighflameGuardResponse:
        """
        Build a ``HighflameGuardResponse`` from Shield's ``decision`` /
        ``reason`` so that downstream LiteLLM code typed against the
        existing per-guard TypedDicts (and the ``_process_assessments``
        consumer) keeps working unchanged.

        The synthesized shape always includes ``request_reject`` and a
        ``results.reject_prompt`` — those are the only fields
        ``_process_assessments`` actually reads. Per-guard-specific fields
        (categories, category_scores, lang/prob, content/strategy) are
        populated with neutral defaults so the response still validates
        as a per-guard TypedDict shape at the dict level.
        """
        is_deny = decision == "deny"
        guard_name = self.highflame_guard_name

        results: Dict[str, object] = {"reject_prompt": reason if is_deny else ""}

        if guard_name == "promptinjectiondetection":
            results["categories"] = {
                "prompt_injection": is_deny,
                "jailbreak": False,
            }
            results["category_scores"] = {
                "prompt_injection": 1.0 if is_deny else 0.0,
                "jailbreak": 0.0,
            }
        elif guard_name == "trustsafety":
            results["categories"] = {
                "violence": is_deny,
                "weapons": False,
                "hate_speech": False,
                "crime": False,
                "sexual": False,
                "profanity": False,
            }
            results["category_scores"] = {
                "violence": 1.0 if is_deny else 0.0,
                "weapons": 0.0,
                "hate_speech": 0.0,
                "crime": 0.0,
                "sexual": 0.0,
                "profanity": 0.0,
            }
        elif guard_name == "lang_detector":
            results["lang"] = ""
            results["prob"] = 1.0 if is_deny else 0.0
        elif guard_name == "dlp_gcp":
            # ``strategy: inspect`` ensures the existing ``_process_assessments``
            # logic does not attempt content transformation when Shield only
            # returns a deny/allow decision. Customers wanting redaction
            # should use Shield's ``mode=modify`` flow, which is not yet
            # surfaced through this integration.
            results["strategy"] = "inspect"

        assessment: Dict[str, object] = {
            "request_reject": is_deny,
            "results": results,
        }

        return {"assessments": [{guard_name: assessment}]}

    def _build_shield_request_body(
        self, request: HighflameGuardRequest
    ) -> Dict[str, object]:
        """
        Build the Shield ``/v1/guard`` request body from a LiteLLM
        ``HighflameGuardRequest``. ``content`` comes from the request input;
        everything else is fixed for prompt-time evaluation.
        """
        input_text = ""
        request_input = request.get("input") or {}
        if isinstance(request_input, dict):
            input_text = request_input.get("text", "") or ""

        request_metadata = request.get("metadata") or {}
        session_id: Optional[str] = None
        if isinstance(request_metadata, dict):
            session_id = (
                request_metadata.get("session_id")
                or request_metadata.get("litellm_call_id")
                or request_metadata.get("request_id")
            )

        body: Dict[str, object] = {
            "content": input_text,
            "content_type": "prompt",
            "action": "process_prompt",
            "mode": "enforce",
            "early_exit": True,
        }
        if session_id:
            body["session_id"] = str(session_id)
        return body

    @staticmethod
    def _safe_response_text(response) -> str:
        try:
            return response.text
        except Exception:
            return "<unreadable body>"

    async def call_highflame_guard(
        self,
        request: HighflameGuardRequest,
        event_type: GuardrailEventHooks,
    ) -> HighflameGuardResponse:
        """
        Call Highflame Shield's ``POST /v1/guard`` endpoint and synthesize a
        ``HighflameGuardResponse`` from Shield's response so that downstream
        LiteLLM code typed against the existing per-guard TypedDicts
        continues to work without changes.

        Errors:
        * 5xx from Shield → treat as service-unavailable. Return a
          synthesized ``allow`` response (passthrough) and warn.
        * 4xx from Shield → log error with body, return a synthesized
          ``allow`` response (passthrough) so a misconfigured guardrail
          does not crash the upstream LiteLLM request.
        * Network / unexpected exception → same passthrough behavior.
        """
        start_time = datetime.now()
        if request.get("metadata") is None and self.metadata is not None:
            request = {**request, "metadata": self.metadata}

        headers = self._build_shield_headers()
        url = f"{self.api_base.rstrip('/')}/v1/guard"
        shield_body = self._build_shield_request_body(request)

        status: GuardrailStatus = "guardrail_failed_to_respond"
        highflame_response: Optional[HighflameGuardResponse] = None
        exception_str = ""

        try:
            verbose_proxy_logger.debug("Highflame Guardrail: Calling URL: %s", url)
            response = await self.async_handler.post(
                url=url,
                headers=headers,
                json=shield_body,
            )
            status_code = response.status_code

            if 500 <= status_code < 600:
                body_text = self._safe_response_text(response)
                verbose_proxy_logger.warning(
                    "Highflame Guardrail: Shield returned %s — treating as service-unavailable, allowing request through. Body: %s",
                    status_code,
                    body_text,
                )
                exception_str = f"Shield {status_code}: {body_text}"
                highflame_response = self._synthesize_guard_response(
                    decision="allow", reason=""
                )
                return highflame_response

            if 400 <= status_code < 500:
                body_text = self._safe_response_text(response)
                verbose_proxy_logger.error(
                    "Highflame Guardrail: Shield returned %s — likely misconfiguration. Allowing request through. Body: %s",
                    status_code,
                    body_text,
                )
                exception_str = f"Shield {status_code}: {body_text}"
                highflame_response = self._synthesize_guard_response(
                    decision="allow", reason=""
                )
                return highflame_response

            response_data = response.json()
            verbose_proxy_logger.debug(
                "Highflame Guardrail: Shield response: %s", response_data
            )

            decision = str(response_data.get("decision", "allow")).lower()
            reason = str(response_data.get("reason", "") or "")
            verbose_proxy_logger.debug(
                "Highflame Guardrail: Shield decision=%s request_id=%s audit_id=%s",
                decision,
                response_data.get("request_id"),
                response_data.get("audit_id"),
            )

            highflame_response = self._synthesize_guard_response(
                decision=decision, reason=reason
            )
            status = "success"
            return highflame_response
        except Exception as e:
            verbose_proxy_logger.error("Highflame Guardrail: API call failed: %s", e)
            status = "guardrail_failed_to_respond"
            exception_str = str(e)
            highflame_response = self._synthesize_guard_response(
                decision="allow", reason=""
            )
            return highflame_response
        finally:
            guardrail_json_response: Union[Exception, str, dict, List[dict]] = {}
            if status == "success" and highflame_response is not None:
                guardrail_json_response = dict(highflame_response)
            elif highflame_response is not None and exception_str:
                # Failed-to-respond branch (5xx / 4xx / exception): include
                # both the error string and the synthesized passthrough
                # response so audit logs capture the full picture.
                guardrail_json_response = {
                    "error": exception_str,
                    "passthrough_response": dict(highflame_response),
                }
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
