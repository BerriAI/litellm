"""
XecGuard guardrail integration for LiteLLM.

Calls the CyCraft XecGuard API (https://api-xecguard.cycraft.ai)
to scan the full conversation history against configured policies
(prompt-injection, PII, harmful-content, custom rules) and, when
grounding documents are supplied via request metadata, also validates
the assistant response against those reference documents via the
/grounding endpoint.

Design notes (intentional divergences from the framework defaults):
  * The full conversation history (system + user + assistant) is always
    forwarded to XecGuard regardless of ``scan_type``. This bypasses the
    framework's optional ``skip_system_message_in_guardrail`` behaviour
    on purpose - policy enforcement depends on system-prompt visibility.
  * ``apply_guardrail`` is defined directly on this class so the
    ``during_call`` dispatch (proxy/utils.py checks for the method on
    ``type(callback).__dict__``) reaches our implementation.
  * ``async_logging_hook`` is overridden because the framework calls it
    directly for ``logging_only`` mode - it does NOT bridge to
    ``apply_guardrail``. Our override runs the scan non-blockingly and
    swallows every exception.
"""

import asyncio
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Literal, Optional

from fastapi.exceptions import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.core_helpers import redact_nested_match_and_regex_keys
from litellm.litellm_core_utils.sensitive_data_masker import mask_credentials_in_payload
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    GenericGuardrailAPIInputs,
    GuardrailStatus,
    StandardLoggingGuardrailInformation,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )


def _sanitize_scan_result_for_logging(scan_result: dict) -> dict:
    without_secrets: Final = {key: value for key, value in scan_result.items() if key != "secret_fields"}
    redacted: Final = redact_nested_match_and_regex_keys(without_secrets)
    masked: Final = mask_credentials_in_payload(redacted if isinstance(redacted, dict) else without_secrets)
    return masked if isinstance(masked, dict) else without_secrets


_DEFAULT_API_BASE: Final = "https://api-xecguard.cycraft.ai"
_SCAN_ENDPOINT: Final = "/xecguard/v1/scan"
_GROUNDING_ENDPOINT: Final = "/xecguard/v1/grounding"
_DEFAULT_MODEL: Final = "xecguard_v2"
_DEFAULT_GROUNDING_STRICTNESS: Final = "BALANCED"
_METADATA_GROUNDING_KEY: Final = "xecguard_grounding_documents"
_RATIONALE_TRUNCATE_CHARS: Final = 200
_DEFAULT_POLICIES: Final = [
    "Default_Policy_SystemPromptEnforcement",
    "Default_Policy_HarmfulContentProtection",
    "Default_Policy_GeneralPromptAttackProtection",
]


class XecGuardMissingCredentials(Exception):
    pass


class XecGuardGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        xecguard_model: str | None = None,
        policy_names: list[str] | None = None,
        block_on_error: bool | None = None,
        grounding_strictness: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key or os.environ.get("XECGUARD_API_KEY")
        if not self.api_key:
            raise XecGuardMissingCredentials(
                "XecGuard API key is required. "
                "Set XECGUARD_API_KEY in the "
                "environment or pass api_key in "
                "the guardrail config."
            )

        self.api_base = (api_base or os.environ.get("XECGUARD_API_BASE") or _DEFAULT_API_BASE).rstrip("/")

        self.xecguard_model = xecguard_model or _DEFAULT_MODEL
        self.policy_names = policy_names

        if block_on_error is None:
            env: Final = os.environ.get("XECGUARD_BLOCK_ON_ERROR", "true")
            self.block_on_error = env.lower() in (
                "true",
                "1",
                "yes",
            )
        else:
            self.block_on_error = block_on_error

        self.grounding_strictness = grounding_strictness or _DEFAULT_GROUNDING_STRICTNESS

        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))

        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.xecguard import (
            XecGuardConfigModel,
        )

        return XecGuardConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
        ]

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        messages: Final = self._build_full_history(
            request_data=request_data,
            inputs=inputs,
            input_type=input_type,
        )
        if not messages:
            return inputs

        scan_type: Final = "input" if input_type == "request" else "response"
        scan_result: Final = await self._call_scan(messages=messages, scan_type=scan_type)
        if scan_result is None:
            return inputs

        if scan_result.get("decision") == "UNSAFE":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": self._format_scan_block_message(scan_result),
                    "guardrail_name": self.guardrail_name or "xecguard",
                    "xecguard_response": scan_result,
                },
            )

        if input_type == "response":
            documents: Final = self._extract_grounding_documents(request_data)
            if documents:
                grounding_result: Final = await self._call_grounding(
                    messages=messages,
                    documents=documents,
                )
                if grounding_result is not None and grounding_result.get("decision") == "UNSAFE":
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": self._format_grounding_block_message(grounding_result),
                            "guardrail_name": self.guardrail_name or "xecguard",
                            "xecguard_response": grounding_result,
                        },
                    )

        return inputs

    async def async_logging_hook(
        self,
        kwargs: dict,
        result: Any,
        call_type: str,
    ) -> tuple[dict, Any]:
        """Observe-only scan for logging_only mode.

        Never blocks, never raises - all errors are swallowed. Records a
        StandardLoggingGuardrailInformation entry so the scan decision
        reaches downstream loggers (Langfuse, DataDog, etc.).
        """
        if (
            isinstance(kwargs, dict)
            and "litellm_params" in kwargs
            and "metadata" in kwargs["litellm_params"]
            and "standard_logging_guardrail_information" in kwargs["litellm_params"]["metadata"]
            and kwargs["litellm_params"]["metadata"]["standard_logging_guardrail_information"]
        ):
            return kwargs, result

        start_time: Final = datetime.now()
        try:
            assistant_text: Final = self._extract_assistant_text_from_response(result)
            request_data: Final = {**kwargs}
            if assistant_text is not None:
                request_data["response"] = result
                messages = self._build_full_history(
                    request_data=request_data,
                    inputs={},
                    input_type="response",
                )
                scan_type = "response"
            else:
                messages = self._build_full_history(
                    request_data=request_data,
                    inputs={},
                    input_type="request",
                )
                scan_type = "input"

            if not messages:
                return kwargs, result

            scan_result: Final = await self._call_scan(
                messages=messages,
                scan_type=scan_type,
                suppress_errors=True,
            )
            if scan_result is None:
                return kwargs, result

            guardrail_status: Final[GuardrailStatus] = (
                "guardrail_intervened" if scan_result.get("decision") == "UNSAFE" else "success"
            )
            end_time: Final = datetime.now()
            slg: Final = StandardLoggingGuardrailInformation(
                guardrail_name=self.guardrail_name or "xecguard",
                guardrail_mode=GuardrailEventHooks.logging_only,
                guardrail_response=_sanitize_scan_result_for_logging(scan_result),
                guardrail_status=guardrail_status,
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                duration=(end_time - start_time).total_seconds(),
                masked_entity_count=None,
            )
            existing: Final = kwargs["standard_logging_object"].get("guardrail_information")
            if isinstance(existing, list):
                existing.append(slg)
            else:
                kwargs["standard_logging_object"]["guardrail_information"] = [slg]

        except Exception as exc:
            verbose_proxy_logger.debug(
                "XecGuard logging_only swallowed exception: %s",
                str(exc),
            )
        return kwargs, result

    def logging_hook(
        self,
        kwargs: dict,
        result: Any,
        call_type: str,
    ) -> tuple[dict, Any]:
        """Sync counterpart to ``async_logging_hook``.

        Runs the async version on an available loop, swallowing every
        exception. Mirrors the pattern used by the Presidio guardrail
        for sync logging callbacks.
        """
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            if loop.is_running():
                return kwargs, result
            loop.run_until_complete(self.async_logging_hook(kwargs=kwargs, result=result, call_type=call_type))
        except Exception as exc:
            verbose_proxy_logger.debug(
                "XecGuard sync logging_hook swallowed exception: %s",
                str(exc),
            )
        return kwargs, result

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _call_scan(
        self,
        messages: list[dict],
        scan_type: str,
        suppress_errors: bool = False,
    ) -> dict | None:
        payload: Final[dict[str, Any]] = {
            "model": self.xecguard_model,
            "scan_type": scan_type,
            "messages": messages,
            "policy_names": (self.policy_names if self.policy_names else _DEFAULT_POLICIES),
        }
        return await self._post(
            path=_SCAN_ENDPOINT,
            payload=payload,
            suppress_errors=suppress_errors,
        )

    async def _call_grounding(
        self,
        messages: list[dict],
        documents: list[dict],
    ) -> dict | None:
        prompt: Final = self._extract_last_text_by_role(messages, "user")
        response_text: Final = self._extract_last_text_by_role(messages, "assistant")
        if prompt is None or response_text is None:
            return None
        payload: Final = {
            "model": self.xecguard_model,
            "prompt": prompt,
            "response": response_text,
            "documents": documents,
            "strictness": self.grounding_strictness,
        }
        return await self._post(path=_GROUNDING_ENDPOINT, payload=payload)

    async def _post(
        self,
        path: str,
        payload: dict,
        suppress_errors: bool = False,
    ) -> dict | None:
        endpoint: Final = f"{self.api_base}{path}"
        verbose_proxy_logger.debug(
            "XecGuard: POST %s payload_keys=%s",
            endpoint,
            list(payload.keys()),
        )
        try:
            response: Final = await self.async_handler.post(
                url=endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            verbose_proxy_logger.error("XecGuard API error: %s", str(exc))
            if suppress_errors:
                return None
            if self.block_on_error:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": (f"XecGuard API unreachable (block_on_error=True): {exc}"),
                        "guardrail_name": self.guardrail_name or "xecguard",
                    },
                ) from exc
            return None

    # ------------------------------------------------------------------
    # Message-assembly helpers (respect the full-history requirement)
    # ------------------------------------------------------------------

    def _build_full_history(
        self,
        request_data: dict,
        inputs: Any,
        input_type: str,
    ) -> list[dict]:
        """Assemble the full message list that will be sent to XecGuard.

        Always reads from ``request_data['messages']`` so the framework's
        optional ``skip_system_message_in_guardrail`` filter cannot strip
        system prompts. Synthesises a trailing user/assistant message when
        the request data is incomplete.
        """
        raw_messages: Final = request_data.get("messages") or []
        messages: Final[list[dict]] = [self._normalize_message(m) for m in raw_messages if isinstance(m, dict)]

        if input_type == "request":
            if not messages:
                return []
            if messages[-1].get("role") != "user":
                synthesized: Final = self._synthesize_user_from_inputs(inputs)
                if synthesized is None:
                    return []
                messages.append(synthesized)
            return messages

        # input_type == "response"
        assistant_text: Final = self._extract_assistant_text_from_response(request_data.get("response"))
        if assistant_text is None:
            return []
        messages.append({"role": "assistant", "content": assistant_text})
        return messages

    @staticmethod
    def _normalize_message(message: dict) -> dict:
        """Flatten multimodal content to a plain string for XecGuard."""
        role: Final = message.get("role") or "user"
        content: Final = message.get("content")
        if isinstance(content, str):
            return {"role": role, "content": content}
        if isinstance(content, list):
            parts: Final[list[str]] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return {"role": role, "content": "\n".join(parts)}
        return {"role": role, "content": ""}

    @staticmethod
    def _synthesize_user_from_inputs(inputs: Any) -> dict | None:
        if not isinstance(inputs, dict):
            return None
        texts: Final = inputs.get("texts")
        if not texts:
            return None
        joined: Final = "\n".join(t for t in texts if isinstance(t, str) and t)
        if not joined:
            return None
        return {"role": "user", "content": joined}

    @staticmethod
    def _extract_last_text_by_role(messages: list[dict], role: str) -> str | None:
        for message in reversed(messages):
            if message.get("role") == role:
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
                return None
        return None

    @staticmethod
    def _extract_assistant_text_from_response(response: Any) -> str | None:
        if response is None:
            return None
        choices = None
        if hasattr(response, "choices"):
            choices = response.choices
        elif isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return None
        text_parts: Final[list[str]] = []
        for choice in choices:
            content = XecGuardGuardrail._extract_choice_content(choice)
            text = XecGuardGuardrail._content_to_text(content)
            if text:
                text_parts.append(text)
        return "\n".join(text_parts) or None

    @staticmethod
    def _extract_choice_content(choice: Any) -> Any:
        if hasattr(choice, "message"):
            message = choice.message
        elif isinstance(choice, dict):
            message = choice.get("message")
        else:
            return None
        if message is None:
            return None
        if hasattr(message, "content"):
            return message.content
        if isinstance(message, dict):
            return message.get("content")
        return None

    @staticmethod
    def _content_to_text(content: Any) -> str | None:
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            parts: Final = [
                item.get("text")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
            ]
            joined: Final = "\n".join(p for p in parts if p)
            return joined or None
        return None

    # ------------------------------------------------------------------
    # Grounding document extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_grounding_documents(request_data: dict) -> list[dict]:
        metadata: Final = request_data.get("metadata") or request_data.get("litellm_metadata")
        if not isinstance(metadata, dict):
            return []
        raw_docs: Final = metadata.get(_METADATA_GROUNDING_KEY)
        if not isinstance(raw_docs, list) or not raw_docs:
            return []
        valid_docs: Final[list[dict]] = []
        for doc in raw_docs:
            if (
                isinstance(doc, dict)
                and isinstance(doc.get("document_id"), str)
                and isinstance(doc.get("context"), str)
            ):
                valid_docs.append(
                    {
                        "document_id": doc["document_id"],
                        "context": doc["context"],
                    }
                )
            else:
                verbose_proxy_logger.debug(
                    "XecGuard: dropping malformed grounding document: %r",
                    doc,
                )
        return valid_docs

    # ------------------------------------------------------------------
    # Error-message formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_scan_block_message(result: dict) -> str:
        trace_id: Final = result.get("trace_id", "")
        violations = result.get("xecguard_result")
        if not isinstance(violations, list):
            violations = []
        seen: Final[list[str]] = []
        for v in violations:
            if not isinstance(v, dict):
                continue
            name = v.get("violated_policy_name")
            if isinstance(name, str) and name and name not in seen:
                seen.append(name)
        policies: Final = ",".join(seen) if seen else "unknown"
        rationale = ""
        for v in violations:
            if isinstance(v, dict):
                candidate = v.get("rationale")
                if isinstance(candidate, str) and candidate:
                    rationale = candidate[:_RATIONALE_TRUNCATE_CHARS]
                    break
        return f"Blocked by XecGuard: policies=[{policies}] trace_id={trace_id} rationale={rationale}"

    @staticmethod
    def _format_grounding_block_message(result: dict) -> str:
        trace_id: Final = result.get("trace_id", "")
        detail: Final = result.get("xecguard_result")
        rules: list[str] = []
        rationale = ""
        if isinstance(detail, dict):
            raw_rules: Final = detail.get("violated_rules_list")
            if isinstance(raw_rules, list):
                rules = [r for r in raw_rules if isinstance(r, str)]
            candidate: Final = detail.get("rationale")
            if isinstance(candidate, str):
                rationale = candidate[:_RATIONALE_TRUNCATE_CHARS]
        rules_str: Final = ",".join(rules) if rules else "unknown"
        return f"Blocked by XecGuard grounding: rules=[{rules_str}] trace_id={trace_id} rationale={rationale}"
