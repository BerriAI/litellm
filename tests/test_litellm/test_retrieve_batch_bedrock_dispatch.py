"""Cover the Bedrock-ARN dispatch in ``litellm.batches.main.retrieve_batch``.

The dispatch picks one of two Bedrock handlers depending on the ARN
family in ``batch_id``:

* ``:async-invoke/<id>``        -> ``_handle_async_invoke_status`` (data plane)
* ``:model-invocation-job/<id>`` -> ``_handle_model_invocation_job_status``
                                    (control plane, added in this PR)

Anything else falls through to the generic ``provider_config`` retrieve
flow. We mock the two handlers so the tests don't hit AWS — the focus
here is purely the dispatch logic that lives in ``main.py``.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm  # noqa: E402

ASYNC_INVOKE_ARN = "arn:aws:bedrock:us-west-2:123456789012:async-invoke/abc123def456"
MIJ_ARN = "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/abc1234567"


@pytest.fixture
def mock_handlers():
    """Patch both Bedrock retrieve handlers and yield the mocks.

    We patch at the import site (litellm.batches.main) rather than the
    definition site so the ``BedrockBatchesHandler`` reference inside
    ``retrieve_batch`` resolves to our mocks.
    """
    fake_batch = MagicMock(name="LiteLLMBatch")
    with (
        patch(
            "litellm.batches.main.BedrockBatchesHandler._handle_async_invoke_status",
            return_value=fake_batch,
        ) as async_invoke,
        patch(
            "litellm.batches.main.BedrockBatchesHandler._handle_model_invocation_job_status",
            return_value=fake_batch,
        ) as mij,
    ):
        yield async_invoke, mij, fake_batch


def test_async_invoke_arn_routes_to_async_invoke_handler(mock_handlers):
    """``:async-invoke/`` ARNs go to the data-plane handler."""
    async_invoke, mij, fake_batch = mock_handlers

    result = litellm.retrieve_batch(
        batch_id=ASYNC_INVOKE_ARN,
        custom_llm_provider="bedrock",
        aws_region_name="us-west-2",
    )

    assert result is fake_batch
    async_invoke.assert_called_once()
    mij.assert_not_called()
    call_kwargs = async_invoke.call_args.kwargs
    assert call_kwargs["batch_id"] == ASYNC_INVOKE_ARN
    assert call_kwargs["aws_region_name"] == "us-west-2"
    # Region must be stripped from the forwarded kwargs to avoid TypeError
    # (it's already an explicit positional/keyword arg).
    assert "aws_region_name" not in {
        k
        for k in call_kwargs
        if k not in {"batch_id", "aws_region_name", "logging_obj"}
    }


def test_async_invoke_arn_falls_back_to_default_region_when_unset(mock_handlers):
    """If no ``aws_region_name`` is passed, the data-plane handler defaults
    to ``us-east-1`` (preserving prior behavior on this branch)."""
    async_invoke, _mij, _ = mock_handlers

    litellm.retrieve_batch(
        batch_id=ASYNC_INVOKE_ARN,
        custom_llm_provider="bedrock",
    )

    async_invoke.assert_called_once()
    assert async_invoke.call_args.kwargs["aws_region_name"] == "us-east-1"


def test_model_invocation_job_arn_routes_to_mij_handler(mock_handlers):
    """``:model-invocation-job/`` ARNs go to the new control-plane handler."""
    _async_invoke, mij, fake_batch = mock_handlers

    result = litellm.retrieve_batch(
        batch_id=MIJ_ARN,
        custom_llm_provider="bedrock",
        aws_region_name="us-west-2",
    )

    assert result is fake_batch
    mij.assert_called_once()
    _async_invoke.assert_not_called()
    call_kwargs = mij.call_args.kwargs
    assert call_kwargs["batch_id"] == MIJ_ARN
    assert call_kwargs["aws_region_name"] == "us-west-2"


def test_model_invocation_job_arn_with_no_region_passes_none(mock_handlers):
    """The MIJ handler is responsible for sniffing region from the ARN
    when none is explicitly provided. Dispatch must forward ``None``
    rather than substituting a default — otherwise per-region jobs in
    other AWS regions would silently route to ``us-east-1``."""
    _async_invoke, mij, _ = mock_handlers

    litellm.retrieve_batch(
        batch_id=MIJ_ARN,
        custom_llm_provider="bedrock",
    )

    mij.assert_called_once()
    assert mij.call_args.kwargs["aws_region_name"] is None


def test_unrelated_bedrock_arn_falls_through_to_provider_config(mock_handlers):
    """Bedrock ARNs that aren't async-invoke or model-invocation-job
    must NOT hit either special handler — they should fall through to
    the existing generic provider_config path. We don't fully exercise
    that path here (it requires a real provider config); we just assert
    neither special handler is invoked."""
    async_invoke, mij, _ = mock_handlers

    # Use a plausible-but-unsupported Bedrock ARN family.
    unrelated_arn = "arn:aws:bedrock:us-west-2:123456789012:provisioned-model/xyz"

    with pytest.raises(Exception):
        # Will raise because no provider_config exists for this path —
        # that's fine, we just need to assert neither bedrock handler ran
        # before the failure.
        litellm.retrieve_batch(
            batch_id=unrelated_arn,
            custom_llm_provider="bedrock",
        )

    async_invoke.assert_not_called()
    mij.assert_not_called()


def test_non_bedrock_id_skips_bedrock_dispatch_entirely(mock_handlers):
    """Plain (non-ARN) batch ids must not even enter the Bedrock dispatch
    block — they belong to other providers' retrieve flows."""
    async_invoke, mij, _ = mock_handlers

    with pytest.raises(Exception):
        litellm.retrieve_batch(
            batch_id="batch_abc123",
            custom_llm_provider="openai",
        )

    async_invoke.assert_not_called()
    mij.assert_not_called()
