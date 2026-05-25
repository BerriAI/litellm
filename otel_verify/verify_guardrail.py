"""A self-contained guardrail that mimics a Bedrock pre-call guardrail.

Drives the OTEL guardrail-span code paths added in #28362 (serialize
guardrail_response as JSON) and #28364 (emit guardrail span on the failure
path + surface guardrail_status / guardrail_action / violation_categories),
without needing AWS.

Trigger words in the prompt:
  - "blockme": records an intervention then raises -> failure path
  - "scanme" : records a successful scan then allows -> success path
  - otherwise: no-op (records nothing)
"""

import time

from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

# Shape mirrors a real Bedrock ApplyGuardrail response so the OTEL integration's
# violation-category extraction has realistic input.
BEDROCK_RESPONSE = {
    "action": "GUARDRAIL_INTERVENED",
    "assessments": [
        {
            "topicPolicy": {
                "topics": [{"name": "Fiduciary Advice", "action": "BLOCKED"}]
            },
            "contentPolicy": {
                "filters": [{"type": "VIOLENCE", "action": "BLOCKED"}]
            },
        }
    ],
}


class VerifyGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _content(data: dict) -> str:
        parts = []
        for m in data.get("messages") or []:
            c = m.get("content")
            if isinstance(c, str):
                parts.append(c)
        return " ".join(parts).lower()

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ):
        content = self._content(data)
        now = time.time()

        if "blockme" in content:
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response=BEDROCK_RESPONSE,
                request_data=data,
                guardrail_status="guardrail_intervened",
                guardrail_provider="bedrock",
                start_time=now,
                end_time=now + 0.01,
                tracing_detail={
                    "guardrail_action": "GUARDRAIL_INTERVENED",
                    "violation_categories": ["Fiduciary Advice", "VIOLENCE"],
                },
            )
            raise HTTPException(
                status_code=400,
                detail={"error": "Blocked by verify-guardrail"},
            )

        if "scanme" in content:
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response=BEDROCK_RESPONSE,
                request_data=data,
                guardrail_status="success",
                guardrail_provider="bedrock",
                start_time=now,
                end_time=now + 0.01,
                tracing_detail={"guardrail_action": "NONE"},
            )

        return data
