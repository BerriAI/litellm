"""
Community maintained integration for Stripe

This will send stripe MeterEvent on every LLM API Call

Use this for customer billing on Stripe

https://docs.stripe.com/billing/subscriptions/usage-based
"""

import math
import os

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger


class StripeLogger(CustomLogger):

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        user_id = (
            kwargs["litellm_params"]
            .get("metadata", {})
            .get("user_api_key_user_id", None)
        )
        stripe_customer_id = (
            kwargs["litellm_params"]
            .get("metadata", {})
            .get("user_api_key_metadata", {})
            .get("stripe_customer_id", None)
        )

        model_and_provider = (
            kwargs["litellm_params"]
            .get("metadata", {})
            .get("model_group", kwargs["model"])
        )
        model = model_and_provider.split("/")[-1]

        costs = kwargs["response_cost"]
        prompt_tokens = response_obj.get("usage", {}).get("prompt_tokens", 0)
        completion_tokens = response_obj.get("usage", {}).get("completion_tokens", 0)

        if stripe_customer_id and model:
            stripe_helper = StripeHelper()

            stripe_helper.create_meter_event(
                f"{model}-costs",
                math.ceil(
                    costs * 10_000_000_000
                ),  # track costs as unit of dollars with 10 decimals
                stripe_customer_id,
            )
            stripe_helper.create_meter_event(
                "tokens",
                prompt_tokens + completion_tokens,
                stripe_customer_id,
            )
            stripe_helper.create_meter_event(
                f"{model}-input-tokens",
                prompt_tokens,
                stripe_customer_id,
            )
            stripe_helper.create_meter_event(
                f"{model}-output-tokens",
                completion_tokens,
                stripe_customer_id,
            )
        else:
            verbose_proxy_logger.debug(
                f"No stripe_customer_id found in metadata for user with id: {user_id} or no model found in response"
            )


class StripeHelper:
    def __init__(self):
        import stripe

        stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")

    def create_meter_event(self, event_name, token_amount, customer_id):
        import stripe

        try:
            stripe.billing.MeterEvent.create(
                event_name=event_name,
                payload={"value": token_amount, "stripe_customer_id": customer_id},
            )
        except Exception as e:
            verbose_proxy_logger.debug(f"Error creating meter event {event_name}: {e}")
