"""Live e2e: POST /guardrails/apply_guardrail is the customer-facing apply surface.

Customers call this endpoint to run a named guardrail without going through chat.
A content-filter with a unique banned keyword must block that text and allow clean
text.
"""

from __future__ import annotations

import pytest

from e2e_config import MASTER_KEY, unique_marker
from e2e_http import Success, UnauthorizedError, UnknownApiError
from guardrails_client import GuardrailsClient
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e


class TestApplyGuardrailEndpoint:
    @pytest.mark.covers(
        "guardrail.litellm_content_filter.apply_endpoint.blocks",
        "guardrail.litellm_content_filter.apply_endpoint.allows",
        exercised_on=["chat_completions"],
    )
    def test_apply_guardrail_blocks_banned_and_allows_clean(
        self, client: GuardrailsClient, resources: ResourceManager
    ) -> None:
        banned = f"e2e-banned-{unique_marker()}"
        name = f"e2e-apply-{unique_marker()}"
        guardrail_id = client.create_content_filter_guardrail(name, banned)
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        blocked = client.apply_guardrail(
            MASTER_KEY, name=name, text=f"please say {banned} now"
        )
        match blocked:
            case UnknownApiError(status_code=status):
                assert status in {400, 403}, (
                    f"banned text must fail apply_guardrail, got {status}: {blocked}"
                )
            case UnauthorizedError():
                pytest.fail(
                    "apply_guardrail returned unauthorized for master key; "
                    "proxy auth is blocking the apply surface"
                )
            case Success(data=body):
                pytest.fail(
                    f"banned text must not pass apply_guardrail; got {body}"
                )
            case _:
                pytest.fail(f"unexpected apply_guardrail block outcome: {blocked}")

        allowed = client.apply_guardrail(
            MASTER_KEY, name=name, text="hello, this is clean input"
        )
        match allowed:
            case Success(data=body):
                assert body.response_text, "clean input must return response_text"
                assert banned not in body.response_text
            case _:
                pytest.fail(f"clean input must succeed on apply_guardrail: {allowed}")
