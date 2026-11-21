import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import (
    Logging,
    get_standard_logging_object_payload,  # pyright: ignore[reportUnknownVariableType]  # its kwargs parameter is an unparameterized dict
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth
from litellm.types.utils import Choices, Message, ModelResponse, StandardLoggingPayload, Usage

FIXTURES: Final = Path(__file__).parents[3] / "litellm/rust/crates/standard-logging/tests/fixtures"
MODEL: Final = "claude-unlisted"
START: Final = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
FIRST_TOKEN: Final = datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
END: Final = datetime(2026, 1, 1, 12, 0, 2, 500000, tzinfo=timezone.utc)
MESSAGES: Final = ({"role": "user", "content": "Hey"},)
KEY: Final = UserAPIKeyAuth(
    api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
    key_alias="ci-key",
    spend=1.5,
    max_budget=10.0,
    budget_reset_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
    metadata={"cost_center": "research"},
    user_id="user-1",
    user_email="user@example.com",
    user_spend=2.5,
    user_max_budget=20.0,
    team_id="team-1",
    team_alias="platform",
    team_spend=3.5,
    team_max_budget=30.0,
    org_id="org-1",
    organization_alias="acme",
    project_id="project-1",
    project_alias="search",
    end_user_id="customer-1",
    request_route="/v1/messages",
    customer_hierarchy=CustomerHierarchyAuth(
        customer_org_id="customer-org-1",
        customer_org_alias="customer-org",
        customer_team_id="customer-team-1",
        customer_team_alias="customer-team",
        customer_member_id="customer-member-1",
        customer_member_alias="customer-member",
        scopes=(),
    ),
)


def _kwargs() -> dict[str, object]:
    return {
        "model": MODEL,
        "call_type": "anthropic_messages",
        "litellm_call_id": "call-1",
        "custom_llm_provider": "anthropic",
        "response_cost": 0.000125,
        "cache_hit": False,
        "messages": list(MESSAGES),
        "optional_params": {"max_tokens": 64, "temperature": 0.5},
        "completion_start_time": FIRST_TOKEN,
        "litellm_params": {
            "api_base": "https://api.anthropic.com/",
            "litellm_trace_id": "trace-1",
            "metadata": {
                **LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(KEY),
                "model_group": "claude",
                "model_info": {"id": "deployment-1"},
                "tags": ["batch-job"],
                "requester_ip_address": "10.0.0.1",
                "user_agent": "curl/8.7.1",
                "spend_logs_metadata": {"project": "alpha"},
            },
            "proxy_server_request": {"headers": {"user-agent": "curl/8.7.1", "x-team": "platform"}, "body": {}},
        },
    }


def _logging() -> Logging:
    return Logging(
        model=MODEL,
        messages=list(MESSAGES),
        stream=False,
        call_type="anthropic_messages",
        start_time=START,
        litellm_call_id="call-1",
        function_id="function-1",
    )


def _success() -> StandardLoggingPayload | None:
    response: Final = ModelResponse(
        id="msg_1",
        model=MODEL,
        created=1767268800,
        choices=[Choices(index=0, finish_reason="stop", message=Message(role="assistant", content="Hi"))],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    response.hidden_params = {
        "model_id": "deployment-1",
        "cache_key": "cache-key-1",
        "api_base": "https://api.anthropic.com/",
        "response_cost": 0.000125,
        "litellm_overhead_time_ms": 3.5,
        "litellm_model_name": MODEL,
        "additional_headers": {
            "x-ratelimit-remaining-requests": "99",
            "llm_provider-request-id": "req_1",
        },
    }
    return get_standard_logging_object_payload(
        kwargs=_kwargs(),
        init_response_obj=response,
        start_time=START,
        end_time=END,
        logging_obj=_logging(),
        status="success",
    )


def _failure() -> StandardLoggingPayload | None:
    error: Final = litellm.RateLimitError(message="slow down", llm_provider="anthropic", model=MODEL)
    return get_standard_logging_object_payload(
        kwargs={**_kwargs(), "response_cost": 0, "exception": error},
        init_response_obj={},
        start_time=START,
        end_time=END,
        logging_obj=_logging(),
        status="failure",
        error_str=str(error),
        original_exception=error,
    )


@pytest.mark.parametrize(("fixture", "build"), [("success.json", _success), ("failure.json", _failure)])
def test_the_builder_logs_the_document_the_rust_payload_is_pinned_to(
    fixture: str, build: Callable[[], StandardLoggingPayload | None]
) -> None:
    path: Final = FIXTURES / fixture
    rendered: Final = json.dumps(build(), indent=2, sort_keys=True) + "\n"
    assert rendered == path.read_text(), f"regenerate {path} from {build.__name__}()"
