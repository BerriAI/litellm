"""
Reusable contract test suite for BaseBatchesConfig implementations.

Any provider whose batch transformation subclasses
`litellm.llms.base_llm.batches.transformation.BaseBatchesConfig` gets a shared
consistency net by subclassing `BatchesConfigContractTests` in its own
`test_transformation.py` (as a `Test*`-named class) and overriding the hooks
below. pytest then runs every contract test against that provider, guaranteeing
all provider batch transformations honour the same BaseBatchesConfig contract -
e.g. every `transform_retrieve_batch_response` returns a real `LiteLLMBatch`
with `object == "batch"`, a valid status, and an int `created_at`.

This module is intentionally NOT named `test_*`: it holds no standalone tests
and must not be collected on its own. It mirrors the established repo pattern in
`tests/llm_translation/base_*_unit_tests.py`.

Providers that do NOT implement BaseBatchesConfig (e.g. vertex_ai, whose
transformation is a standalone class with a different shape) cannot use this and
keep fully standalone tests.
"""

import os
import sys
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.types.utils import LiteLLMBatch, LlmProviders

# The OpenAI BatchJobStatus literal set - every provider must map into this.
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
    """Build an httpx.Response whose .json() yields `body` - the input shape the
    transform_*_response methods consume."""
    return httpx.Response(status_code=status_code, json=body)


class BatchesConfigContractTests:
    """Contract every BaseBatchesConfig implementation must satisfy.

    Subclass this with a `Test`-prefixed class and override the hooks. Do NOT
    add a `Test` prefix here - this base must not be collected directly.
    """

    # ----------------------------------------------------------------------- #
    # Hooks - providers MUST override these.
    # ----------------------------------------------------------------------- #

    def make_config(self):
        """Return a fresh instance of the provider's BaseBatchesConfig."""
        raise NotImplementedError("override make_config()")

    # The LlmProviders value this config reports.
    expected_provider: LlmProviders = None  # type: ignore[assignment]

    # Does this provider implement batch CREATE via the transformation?
    # (anthropic raises NotImplementedError; bedrock/others may support it.)
    supports_create: bool = False

    # Does this provider parse retrieve responses in the transformation layer?
    supports_retrieve_response: bool = True

    def sample_retrieve_response_body(self) -> dict:
        """A representative raw provider retrieve-batch response body."""
        raise NotImplementedError("override sample_retrieve_response_body()")

    # Expected mapped values for the sample above.
    expected_retrieve_batch_id: str = None  # type: ignore[assignment]
    expected_retrieve_status: str = None  # type: ignore[assignment]

    # ----------------------------------------------------------------------- #
    # Contract tests - run for every provider subclass.
    # ----------------------------------------------------------------------- #

    def test_contract__custom_llm_provider(self):
        assert self.make_config().custom_llm_provider == self.expected_provider

    def test_contract__get_error_class_is_exception_with_status(self):
        err = self.make_config().get_error_class(
            error_message="boom", status_code=429, headers={}
        )
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
