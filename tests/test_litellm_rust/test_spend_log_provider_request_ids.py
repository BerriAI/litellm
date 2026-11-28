import datetime
import json
from collections.abc import Mapping
from datetime import timezone
from typing import Final

import pytest

import litellm
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

pytestmark = pytest.mark.requires_rust_extension


def _kwargs(**extra: object) -> dict[str, object]:
    return {
        "model": "gpt-4o",
        "custom_llm_provider": "azure",
        "litellm_call_id": "prid-call",
        "litellm_params": {"metadata": {"user_api_key": "test-key"}},
        **extra,
    }


def _stored_ids(kwargs: Mapping[str, object]) -> object:
    payload: Final = get_logging_payload(
        kwargs=dict(kwargs),
        response_obj=litellm.ModelResponse(id="chatcmpl-prid", choices=[], usage=litellm.Usage()),
        start_time=datetime.datetime.now(timezone.utc),
        end_time=datetime.datetime.now(timezone.utc),
    )
    return json.loads(payload["metadata"])["provider_request_ids"]


def test_request_ids_are_stamped_from_the_raw_response_headers() -> None:
    kwargs: Final = _kwargs(
        response_headers={"apim-request-id": "azure-corr-1", "x-request-id": "openai-req-1", "retry-after": "7"}
    )
    assert _stored_ids(kwargs) == {"apim-request-id": "azure-corr-1", "x-request-id": "openai-req-1"}


def test_request_ids_are_stamped_from_the_llm_provider_copies_of_a_streaming_call() -> None:
    kwargs: Final = _kwargs(
        standard_logging_object={
            "hidden_params": {"additional_headers": {"llm_provider-apim-request-id": "azure-corr-2"}},
            "metadata": {},
            "model_map_information": None,
        }
    )
    assert _stored_ids(kwargs) == {"apim-request-id": "azure-corr-2"}


def test_a_mixed_case_response_header_is_stored_under_its_lower_case_name() -> None:
    assert _stored_ids(_kwargs(response_headers={"Apim-Request-Id": "azure-corr-3"})) == {
        "apim-request-id": "azure-corr-3"
    }


def test_no_request_id_header_stores_none() -> None:
    assert _stored_ids(_kwargs(response_headers={"retry-after": "7"})) is None


def test_an_oversized_request_id_is_not_stored() -> None:
    assert _stored_ids(_kwargs(response_headers={"apim-request-id": "x" * 257})) is None


def test_request_ids_cannot_be_set_through_request_metadata() -> None:
    kwargs: Final = _kwargs()
    kwargs["litellm_params"]["metadata"]["provider_request_ids"] = {"apim-request-id": "forged"}  # pyright: ignore[reportIndexIssue]
    assert _stored_ids(kwargs) is None
