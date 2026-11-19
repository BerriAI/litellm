import pytest

from litellm.litellm_core_utils.clientside_credential_handler import carries_caller_credentials


@pytest.mark.parametrize("carrier", ["headers", "extra_headers", "default_headers"])
@pytest.mark.parametrize("header", ["Authorization", "x-api-key", "Api-Key"])
def test_a_credential_header_in_any_header_carrier_counts_as_a_caller_credential(carrier: str, header: str) -> None:
    assert carries_caller_credentials({carrier: {header: "caller-secret"}}) is True


@pytest.mark.parametrize("carrier", ["headers", "extra_headers", "default_headers"])
def test_a_plain_header_is_not_a_caller_credential(carrier: str) -> None:
    assert carries_caller_credentials({carrier: {"x-app": "my-app", "traceparent": "00-abc-def-01"}}) is False
