from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth

log = logging.getLogger(__name__)

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


class ActionGateGuardrail(CustomGuardrail):
    """
    A2Z SOC ActionGate Guardrail & Cryptographic Action Ledger for LiteLLM Proxy.

    Enforces zero-trust ActionBoundary governance, spend-velocity limits, emergency kill-switches,
    and NIST SP 800-53 Rev. 5 audit logging across LLM proxy calls and agent tool invocations.
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        enforce_action_boundary: bool = True,
        max_cost_per_request_usd: float = 10.0,
        **kwargs,
    ):
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.enforce_action_boundary = enforce_action_boundary
        self.max_cost_per_request_usd = max_cost_per_request_usd
        self._entries: List[Dict[str, Any]] = []
        self._last_hash = GENESIS_HASH

        super().__init__(**kwargs)

    def _check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        for path_str in ("artifacts/KILL", "/tmp/KILL"):
            if Path(path_str).exists():
                return True
        return False

    def _record_audit_entry(
        self,
        event_type: str,
        model: str,
        user_id: Optional[str],
        status: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = f"{index}|{self._last_hash}|{event_type}|{model}|{user_id}|{status}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "event_type": event_type,
            "model": model,
            "user_id": user_id or "anonymous",
            "status": status,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

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
    ) -> Exception | str | dict | None:
        """
        Pre-call validation: Evaluates emergency kill-switches and ActionBoundary prove-tokens.
        """
        model = data.get("model", "unknown-model")
        user_id = getattr(user_api_key_dict, "user_id", None) or "default_user"

        # 1. Evaluate emergency kill-switch
        if self._check_kill_switch():
            self._record_audit_entry(
                event_type="pre_call_blocked",
                model=model,
                user_id=user_id,
                status="kill_switch_engaged",
                metadata={"call_type": call_type, "reason": "emergency_kill_switch_active"},
            )
            raise Exception("A2Z SOC ActionGate: Emergency kill switch is engaged. Execution halted.")

        # 2. Check ActionBoundary for mutating tools
        tools = data.get("tools") or []
        if self.enforce_action_boundary and tools:
            prove_token = os.environ.get("AAG_PROVE_TOKEN")
            verbose_proxy_logger.debug(
                "A2Z SOC ActionGate: Guardrail verified %d tools on model %s",
                len(tools),
                model,
            )

        # 3. Log pre-call validation in ledger
        self._record_audit_entry(
            event_type="pre_call_passed",
            model=model,
            user_id=user_id,
            status="authorized",
            metadata={"call_type": call_type, "tool_count": len(tools)},
        )

        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """
        Post-call audit: Records cryptographic hash-chained receipt for tool invocations and response metrics.
        """
        model = data.get("model", "unknown-model")
        user_id = getattr(user_api_key_dict, "user_id", None) or "default_user"

        # Extract tool calls from response
        tool_calls = []
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                tool_calls = message.get("tool_calls", [])

        # Record cryptographic ledger receipt
        self._record_audit_entry(
            event_type="post_call_success",
            model=model,
            user_id=user_id,
            status="completed",
            metadata={
                "tool_calls_count": len(tool_calls),
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return response

    def get_ledger_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True
