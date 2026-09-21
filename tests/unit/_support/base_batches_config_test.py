from unittest.mock import MagicMock

import httpx
import pytest

from litellm.types.utils import LiteLLMBatch, LlmProviders

VALID_BATCH_STATUSES = {
    "validating",
    "failed",
    "in_progress",
    "finalizing",
    "completed",
    "expired",
    "cancelling",
    "cancelled",
}


def make_raw_response(body: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status_code, json=body)


class BatchesConfigContractTests:
    expected_provider: LlmProviders
    supports_create: bool = False
    supports_retrieve_response: bool = True
    expected_retrieve_batch_id: str
    expected_retrieve_status: str

    def make_config(self):
        raise NotImplementedError("override make_config()")

    def sample_retrieve_response_body(self) -> dict:
        raise NotImplementedError("override sample_retrieve_response_body()")

    def test_contract__custom_llm_provider(self):
        assert self.make_config().custom_llm_provider == self.expected_provider

    def test_contract__get_error_class_is_exception_with_status(self):
        err = self.make_config().get_error_class(error_message="boom", status_code=429, headers={})
        assert isinstance(err, Exception)
        assert getattr(err, "status_code", None) == 429

    def test_contract__create_unsupported_raises(self):
        if self.supports_create:
            pytest.skip("provider supports batch create; see provider-specific tests")
        with pytest.raises(NotImplementedError):
            self.make_config().transform_create_batch_request(
                model="m",
                create_batch_data={
                    "input_file_id": "f",
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
                optional_params={},
                litellm_params={},
            )

    def test_contract__retrieve_response_is_valid_litellm_batch(self):
        if not self.supports_retrieve_response:
            pytest.skip("provider handles retrieve outside the transformation layer")
        out = self.make_config().transform_retrieve_batch_response(
            model=None,
            raw_response=make_raw_response(self.sample_retrieve_response_body()),
            logging_obj=MagicMock(),
            litellm_params={},
        )
        assert isinstance(out, LiteLLMBatch)
        assert out.object == "batch"
        assert out.status in VALID_BATCH_STATUSES
        assert isinstance(out.created_at, int)
        assert out.id == self.expected_retrieve_batch_id
        assert out.status == self.expected_retrieve_status
