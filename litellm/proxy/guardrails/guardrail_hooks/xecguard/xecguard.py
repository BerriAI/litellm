"""
CyCraft XecGuard guardrail integration for LiteLLM proxy.

Covers:
  - /xecguard/v1/scan   – input & response scanning
  - /xecguard/v1/grounding – RAG context-grounding verification

All three LiteLLM hook points are wired:
  pre_call   -> scan INPUT before the LLM call
  during_call -> scan INPUT in parallel with the LLM call
  post_call   -> scan OUTPUT (+ optional grounding) after the LLM call
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from fastapi import HTTPException

import litellm
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
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.guardrails.guardrail_hooks.xecguard import (
    XecGuardUIConfigModel,
)
from litellm.types.utils import (
    Choices,
    ModelResponse,
    ModelResponseStream,
    StandardLoggingGuardrailInformation,
)

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.types.utils import CallTypesLiteral

# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────
GUARDRAIL_NAME = "xecguard"

DEFAULT_POLICY_NAMES: List[str] = [
    "Default_Policy_SystemPromptEnforcement",
    "Default_Policy_GeneralPromptAttackProtection",
    "Default_Policy_ContentBiasProtection",
    "Default_Policy_HarmfulContentProtection",
    "Default_Policy_PIISensitiveDataProtection",
    "Default_Policy_SkillsProtection",
]

SCAN_ENDPOINT = "/xecguard/v1/scan"
GROUNDING_ENDPOINT = "/xecguard/v1/grounding"


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _extract_text(content: Any) -> str:
    """Flatten message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content) if content else ""


def _litellm_messages_to_xecguard(
    messages: List[AllMessageValues],
) -> List[Dict[str, str]]:
    """Convert LiteLLM messages to XecGuard's ``{role, content}`` list."""
    converted: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = _extract_text(msg.get("content", ""))
        if content:
            converted.append({"role": role, "content": content})
    return converted


def _last_role(messages: List[Dict[str, str]]) -> str:
    if messages:
        return messages[-1].get("role", "user")
    return "user"


def _pre_register_guardrail_info(
    data: dict,
    guardrail_name: Optional[str],
    event_type: GuardrailEventHooks,
    start_time: float,
) -> StandardLoggingGuardrailInformation:
    """
    Eagerly create and register a guardrail-info placeholder in request
    metadata **before** the async scan call.

    In ``during_call`` mode the guardrail scan runs in parallel with the
    LLM call via ``asyncio.gather``.  The LLM success handler may build
    the ``StandardLoggingPayload`` (which reads
    ``metadata["standard_logging_guardrail_information"]``) before the
    scan finishes.  By pre-registering a placeholder with an optimistic
    ``guardrail_status="success"``, the payload always includes guardrail
    information.  The caller updates the **same dict object** in-place
    once the scan completes or fails.

    Returns the placeholder dict stored in metadata so the caller can
    mutate it.
    """
    placeholder = StandardLoggingGuardrailInformation(
        guardrail_name=guardrail_name,
        guardrail_provider=GUARDRAIL_NAME,
        guardrail_mode=event_type,
        guardrail_response=None,
        guardrail_status="success",
        start_time=start_time,
        end_time=None,
        duration=None,
    )

    key = "standard_logging_guardrail_information"
    metadata = data.get("metadata")
    if metadata is None:
        data["metadata"] = {}
        metadata = data["metadata"]

    existing = metadata.get(key)
    if existing is None:
        metadata[key] = [placeholder]
    elif isinstance(existing, list):
        existing.append(placeholder)

    return placeholder


# ────────────────────────────────────────────────────────────────────
# Main guardrail class
# ────────────────────────────────────────────────────────────────────


class XecGuardGuardrail(CustomGuardrail):
    """CyCraft XecGuard guardrail for LiteLLM proxy."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "xecguard_v2",
        policy_names: Optional[List[str]] = None,
        grounding_enabled: bool = False,
        grounding_strictness: Literal["BALANCED", "STRICT"] = "BALANCED",
        grounding_documents: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback,
        )

        resolved_key = api_key or os.getenv("XECGUARD_SERVICE_TOKEN", "")
        if not resolved_key:
            raise ValueError(
                "XecGuard: no API key provided. Pass `api_key` or set "
                "the XECGUARD_SERVICE_TOKEN environment variable."
            )
        self.api_key: str = resolved_key

        self.api_base: str = (
            api_base
            if api_base
            else os.getenv("XECGUARD_API_BASE", "https://api-xecguard.cycraft.ai")
        ).rstrip("/")

        self.model = model
        self.policy_names: List[str] = policy_names or list(DEFAULT_POLICY_NAMES)

        self.grounding_enabled = grounding_enabled
        self.grounding_strictness = grounding_strictness
        self.grounding_documents: List[Dict[str, str]] = grounding_documents or []

        self.guardrail_provider = GUARDRAIL_NAME

        super().__init__(**kwargs)

        verbose_proxy_logger.debug(
            "XecGuard guardrail initialised – api_base=%s, policies=%s, grounding=%s",
            self.api_base,
            self.policy_names,
            self.grounding_enabled,
        )

    # ----------------------------------------------------------------
    # Config model
    # ----------------------------------------------------------------

    @staticmethod
    def get_config_model() -> type:
        """Return the UI config model (excludes Context Grounding fields)."""
        return XecGuardUIConfigModel

    # ----------------------------------------------------------------
    # Low-level API helpers
    # ----------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def _call_scan(
        self,
        scan_type: Literal["input", "response"],
        messages: List[Dict[str, str]],
        policy_names: Optional[List[str]] = None,
        request_data: Optional[dict] = None,
    ) -> dict:
        """Call ``POST /xecguard/v1/scan`` and return the JSON body."""
        url = f"{self.api_base}{SCAN_ENDPOINT}"

        effective_policies = list(policy_names or self.policy_names)

        if request_data:
            dyn = self.get_guardrail_dynamic_request_body_params(
                request_data=request_data,
            )
            if dyn.get("policy_names"):
                effective_policies = dyn["policy_names"]
            if dyn.get("scan_type"):
                scan_type = dyn["scan_type"]

        body = {
            "model": self.model,
            "scan_type": scan_type,
            "messages": messages,
            "policy_names": effective_policies,
        }

        verbose_proxy_logger.debug(
            "XecGuard /scan request: %s",
            json.dumps(body, ensure_ascii=False)[:2000],
        )

        resp = await self.async_handler.post(
            url=url,
            headers=self._headers(),
            json=body,
            timeout=30,
        )

        if resp.status_code == 413:
            raise HTTPException(
                status_code=413,
                detail="XecGuard: request content exceeds the maximum allowed length (128k tokens).",
            )

        if resp.status_code != 200:
            detail = resp.text[:500] if resp.text else "Unknown error"
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"XecGuard scan failed ({resp.status_code}): {detail}",
            )

        return resp.json()

    async def _call_grounding(
        self,
        prompt: str,
        response_text: str,
        documents: Optional[List[Dict[str, str]]] = None,
        strictness: Optional[Literal["BALANCED", "STRICT"]] = None,
        request_data: Optional[dict] = None,
    ) -> dict:
        """Call ``POST /xecguard/v1/grounding`` and return the JSON body."""
        url = f"{self.api_base}{GROUNDING_ENDPOINT}"

        docs = documents or []
        level = strictness or self.grounding_strictness

        if request_data:
            dyn = self.get_guardrail_dynamic_request_body_params(
                request_data=request_data,
            )
            if dyn.get("grounding_documents"):
                docs = dyn["grounding_documents"]
            if dyn.get("grounding_strictness"):
                level = dyn["grounding_strictness"]

        body = {
            "model": self.model,
            "prompt": prompt,
            "response": response_text,
            "documents": docs,
            "strictness": level,
        }

        verbose_proxy_logger.debug(
            "XecGuard /grounding request: %s",
            json.dumps(body, ensure_ascii=False)[:2000],
        )

        resp = await self.async_handler.post(
            url=url,
            headers=self._headers(),
            json=body,
            timeout=30,
        )

        if resp.status_code == 413:
            raise HTTPException(
                status_code=413,
                detail="XecGuard: grounding request content exceeds the 128k token limit.",
            )

        if resp.status_code != 200:
            detail = resp.text[:500] if resp.text else "Unknown error"
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"XecGuard grounding failed ({resp.status_code}): {detail}",
            )

        return resp.json()

    # ----------------------------------------------------------------
    # Decision helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _raise_if_unsafe_scan(result: dict) -> None:
        """Raise HTTPException when XecGuard /scan returns UNSAFE."""
        decision = result.get("decision", "SAFE")
        if decision != "UNSAFE":
            return

        trace_id = result.get("trace_id", "n/a")
        xecguard_result = result.get("xecguard_result", [])

        violation_summaries: List[str] = []
        for v in xecguard_result:
            vtype = v.get("type", "UNKNOWN")
            policy = v.get("violated_policy_name", "")
            rationale = v.get("rationale", "")
            violation_summaries.append(f"[{vtype}] {policy}: {rationale}")
        detail_str = (
            "; ".join(violation_summaries)
            if violation_summaries
            else "Policy violation detected"
        )

        raise HTTPException(
            status_code=400,
            detail={
                "error": f"XecGuard scan blocked — {detail_str} (trace_id={trace_id})",
                "xecguard_response": xecguard_result,
            },
        )

    @staticmethod
    def _raise_if_unsafe_grounding(result: dict) -> None:
        """Raise HTTPException when XecGuard /grounding returns UNSAFE."""
        decision = result.get("decision", "SAFE")
        if decision != "UNSAFE":
            return

        trace_id = result.get("trace_id", "n/a")
        xr = result.get("xecguard_result")
        if xr is None:
            xr = {}

        rationale = ""
        violated_rules: List[str] = []
        if isinstance(xr, dict):
            rationale = xr.get("rationale", "")
            violated_rules = xr.get("violated_rules_list", [])

        detail_str = rationale or "Response is not grounded in the provided context"
        if violated_rules:
            detail_str += f" (rules: {', '.join(violated_rules)})"

        raise HTTPException(
            status_code=400,
            detail={
                "error": f"XecGuard grounding failed — {detail_str} (trace_id={trace_id})",
                "xecguard_response": xr,
            },
        )

    # ----------------------------------------------------------------
    # Convenience extractors
    # ----------------------------------------------------------------

    @staticmethod
    def _last_user_prompt(messages: List[AllMessageValues]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return _extract_text(m.get("content", ""))
        return ""

    @staticmethod
    def _response_text(response: Any) -> str:
        if isinstance(response, litellm.ModelResponse):
            parts: List[str] = []
            for choice in response.choices:
                if isinstance(choice, Choices) and choice.message.content:
                    parts.append(choice.message.content)
            return "\n".join(parts)
        return ""

    # ----------------------------------------------------------------
    # Hook: pre_call  (scan INPUT before the LLM call)
    # ----------------------------------------------------------------

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: "DualCache",
        data: dict,
        call_type: "CallTypesLiteral",
    ) -> Optional[Union[Exception, str, dict]]:
        event_type = GuardrailEventHooks.pre_call
        if not self.should_run_guardrail(data=data, event_type=event_type):
            return data

        messages: Optional[List[AllMessageValues]] = data.get("messages")
        if not messages:
            return data

        xg_messages = _litellm_messages_to_xecguard(messages)
        if not xg_messages:
            return data

        last = _last_role(xg_messages)
        scan_type: Literal["input", "response"] = (
            "input" if last == "user" else "response"
        )

        result = await self._call_scan(
            scan_type=scan_type,
            messages=xg_messages,
            request_data=data,
        )
        self._raise_if_unsafe_scan(result)

        return data

    # ----------------------------------------------------------------
    # Hook: during_call  (scan INPUT in parallel with LLM)
    # ----------------------------------------------------------------

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: "CallTypesLiteral",
    ) -> Optional[dict]:
        event_type = GuardrailEventHooks.during_call
        if not self.should_run_guardrail(data=data, event_type=event_type):
            return None

        messages: Optional[List[AllMessageValues]] = data.get("messages")
        if not messages:
            return None

        xg_messages = _litellm_messages_to_xecguard(messages)
        if not xg_messages:
            return None

        last = _last_role(xg_messages)
        scan_type: Literal["input", "response"] = (
            "input" if last == "user" else "response"
        )

        # Pre-register guardrail info in metadata BEFORE the scan call.
        # In during_call mode the scan runs in parallel with the LLM call
        # via asyncio.gather.  The LLM success handler may build the
        # StandardLoggingPayload before the scan finishes; pre-registering
        # ensures passed requests are counted in the Guardrails Monitor.
        start_time = time.time()
        placeholder = _pre_register_guardrail_info(
            data=data,
            guardrail_name=self.guardrail_name,
            event_type=event_type,
            start_time=start_time,
        )

        try:
            result = await self._call_scan(
                scan_type=scan_type,
                messages=xg_messages,
                request_data=data,
            )
            end_time = time.time()
            placeholder["guardrail_response"] = result
            placeholder["end_time"] = end_time
            placeholder["duration"] = end_time - start_time

            self._raise_if_unsafe_scan(result)
        except Exception as e:
            end_time = time.time()
            placeholder["end_time"] = end_time
            placeholder["duration"] = end_time - start_time
            placeholder["guardrail_response"] = str(e)
            if self._is_guardrail_intervention(e):
                placeholder["guardrail_status"] = "guardrail_intervened"
            else:
                placeholder["guardrail_status"] = "guardrail_failed_to_respond"
            raise

        return None

    # ----------------------------------------------------------------
    # Hook: post_call  (scan OUTPUT + optional grounding)
    # ----------------------------------------------------------------

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> None:
        if not self.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.post_call
        ):
            return

        messages: Optional[List[AllMessageValues]] = data.get("messages")
        if not messages:
            return

        resp_text = self._response_text(response)
        if not resp_text:
            verbose_proxy_logger.debug(
                "XecGuard post_call: no text in response, skipping"
            )
            return

        # OUTPUT scan
        xg_messages = _litellm_messages_to_xecguard(messages)
        xg_messages.append({"role": "assistant", "content": resp_text})

        scan_task = self._call_scan(
            scan_type="response",
            messages=xg_messages,
            request_data=data,
        )

        # Grounding (optional)
        grounding_task = None
        if self.grounding_enabled:
            user_prompt = self._last_user_prompt(messages)
            dyn = self.get_guardrail_dynamic_request_body_params(request_data=data)
            docs = dyn.get("grounding_documents") or self.grounding_documents
            if docs:
                grounding_task = self._call_grounding(
                    prompt=user_prompt,
                    response_text=resp_text,
                    documents=docs,
                    request_data=data,
                )

        if grounding_task:
            scan_result, grounding_result = await asyncio.gather(
                scan_task, grounding_task
            )
            self._raise_if_unsafe_scan(scan_result)
            self._raise_if_unsafe_grounding(grounding_result)
        else:
            scan_result = await scan_task
            self._raise_if_unsafe_scan(scan_result)

    # ----------------------------------------------------------------
    # Hook: streaming post_call
    # ----------------------------------------------------------------

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """Collect the full stream, scan it, then re-emit."""
        if not self.should_run_guardrail(
            data=request_data, event_type=GuardrailEventHooks.post_call
        ):
            async for chunk in response:
                yield chunk
            return

        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder
        from litellm.types.utils import TextCompletionResponse

        all_chunks: List[ModelResponseStream] = []
        async for chunk in response:
            all_chunks.append(chunk)

        assembled: Optional[Union[ModelResponse, TextCompletionResponse]] = (
            stream_chunk_builder(chunks=all_chunks)
        )

        if isinstance(assembled, ModelResponse):
            resp_text = self._response_text(assembled)
            messages: Optional[List[AllMessageValues]] = request_data.get("messages")

            if resp_text and messages:
                xg_messages = _litellm_messages_to_xecguard(messages)
                xg_messages.append({"role": "assistant", "content": resp_text})

                scan_task = self._call_scan(
                    scan_type="response",
                    messages=xg_messages,
                    request_data=request_data,
                )

                grounding_task = None
                if self.grounding_enabled:
                    user_prompt = self._last_user_prompt(messages)
                    dyn = self.get_guardrail_dynamic_request_body_params(
                        request_data=request_data
                    )
                    docs = dyn.get("grounding_documents") or self.grounding_documents
                    if docs:
                        grounding_task = self._call_grounding(
                            prompt=user_prompt,
                            response_text=resp_text,
                            documents=docs,
                            request_data=request_data,
                        )

                if grounding_task:
                    scan_result, grounding_result = await asyncio.gather(
                        scan_task, grounding_task
                    )
                    self._raise_if_unsafe_scan(scan_result)
                    self._raise_if_unsafe_grounding(grounding_result)
                else:
                    scan_result = await scan_task
                    self._raise_if_unsafe_scan(scan_result)

            mock_iter = MockResponseIterator(model_response=assembled)
            async for chunk in mock_iter:
                yield chunk
        else:
            for chunk in all_chunks:
                yield chunk
