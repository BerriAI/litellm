"""
Xcity wallet billing callback (XCT-322 / S3).

Debits the xct-wallet KWH ledger on every successful model call. KWH is a
display layer over the wallet's integer `credit` unit (1 KWH = 100 credits =
$0.10). The per-call charge carries a flat markup over upstream cost:

    billed_credits = round(response_cost_usd * 1000 * markup)   # markup=1.8

The wallet debit is idempotent on the LiteLLM call id, so retries / duplicate
callback fires never double-charge. This is post-hoc accounting that runs
alongside LiteLLM's own key budgets — it never blocks the (already completed)
request, and any failure is swallowed so billing can never break inference.

Enable by setting WALLET_BASE_URL + WALLET_SERVICE_TOKEN and registering in the
proxy config:

    litellm_settings:
      callbacks: litellm.integrations.xcity_wallet.xcity_wallet_billing_instance
"""

import os
from typing import Any, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
from litellm.utils import get_end_user_id_for_cost_tracking

# 1 USD = 1000 credits (wallet internal peg); KWH is credits/100.
CREDITS_PER_USD = 1000
DEFAULT_MARKUP = 1.8

# Wallet debit product must be one of the wallet's allowed enum values.
_PRODUCT = "tokenhub"
_METER = "model_call"


class XcityWalletBilling(CustomLogger):
    def __init__(self) -> None:
        self.base_url = (os.getenv("WALLET_BASE_URL") or "").rstrip("/")
        self.service_token = os.getenv("WALLET_SERVICE_TOKEN") or ""
        try:
            self.markup = float(os.getenv("KWH_MARKUP") or DEFAULT_MARKUP)
        except (TypeError, ValueError):
            self.markup = DEFAULT_MARKUP
        self.enabled = bool(self.base_url and self.service_token)
        if not self.enabled:
            verbose_logger.info(
                "[xcity_wallet] disabled — WALLET_BASE_URL / WALLET_SERVICE_TOKEN unset"
            )

    def _resolve_user(self, kwargs: dict) -> Optional[str]:
        """The xct user to bill: end_user (claws / X-Fastclaw-End-User isolation)
        falls back to the key owner's user_id. Both are the GoTrue user UUID."""
        litellm_params = kwargs.get("litellm_params", {}) or {}
        end_user_id = get_end_user_id_for_cost_tracking(litellm_params)
        if end_user_id:
            return end_user_id
        metadata = get_litellm_metadata_from_kwargs(kwargs=kwargs) or {}
        return metadata.get("user_api_key_user_id")

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        if not self.enabled:
            return
        try:
            sl = kwargs.get("standard_logging_object") or {}

            response_cost = sl.get("response_cost")
            if response_cost is None:
                response_cost = kwargs.get("response_cost")
            # Cache hits / zero-cost calls: nothing to bill.
            if not response_cost or response_cost <= 0:
                return

            xct_user = self._resolve_user(kwargs)
            if not xct_user:
                verbose_logger.warning(
                    "[xcity_wallet] no end_user/user_id on call — skipping debit"
                )
                return

            request_id = sl.get("id") or kwargs.get("litellm_call_id")
            if not request_id:
                return

            billed = round(response_cost * CREDITS_PER_USD * self.markup)
            if billed <= 0:
                return

            payload: dict = {
                "user_id": xct_user,
                "request_id": str(request_id),
                "amount_credits": billed,
                "product": _PRODUCT,
                "meter": _METER,
                "upstream_cost_usd": float(response_cost),
            }
            model = sl.get("model") or kwargs.get("model")
            if model:
                payload["upstream_model"] = str(model)[:128]
            for k, src in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
                v = sl.get(src)
                if isinstance(v, int) and v >= 0:
                    payload[k] = v

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/wallet/debit",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.service_token}"},
                )
            # 402 = insufficient credits: expected under post-paid accounting
            # (LiteLLM's key budget is the real-time gate); log, don't raise.
            if resp.status_code not in (200, 402):
                verbose_logger.warning(
                    f"[xcity_wallet] debit {resp.status_code} for user={xct_user} req={request_id}"
                )
        except Exception as e:  # never break inference on a billing error
            verbose_logger.warning(f"[xcity_wallet] debit failed: {e}")


xcity_wallet_billing_instance = XcityWalletBilling()
