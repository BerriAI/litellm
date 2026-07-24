"""
Unit tests for litellm/proxy/guardrails/guardrail_helpers.py
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.guardrails.guardrail_helpers import should_proceed_based_on_metadata
from litellm.types.guardrails import GuardrailItem


def _config_map():
    return {
        "prompt_injection": GuardrailItem(
            guardrail_name="prompt_injection",
            callbacks=["lakera_prompt_injection"],
            default_on=True,
        ),
        "pii_masking": GuardrailItem(
            guardrail_name="pii_masking",
            callbacks=["presidio"],
            default_on=False,
        ),
    }


@pytest.mark.asyncio
async def test_default_on_guardrail_runs_when_other_guardrail_named():
    """
    Regression for #33944: naming a different guardrail must not silently skip a
    default_on guardrail the caller never tried to disable
    """
    with patch("litellm.guardrail_name_config_map", _config_map()):
        assert (
            await should_proceed_based_on_metadata(
                data={"metadata": {"guardrails": {"pii_masking": True}}},
                guardrail_name="lakera_prompt_injection",
            )
            is True
        )


@pytest.mark.asyncio
async def test_default_on_guardrail_runs_when_guardrails_dict_empty():
    """
    Regression for #33944: an empty metadata.guardrails dict must not skip a
    default_on guardrail
    """
    with patch("litellm.guardrail_name_config_map", _config_map()):
        assert (
            await should_proceed_based_on_metadata(
                data={"metadata": {"guardrails": {}}},
                guardrail_name="lakera_prompt_injection",
            )
            is True
        )


@pytest.mark.asyncio
async def test_guardrail_skipped_when_explicitly_disabled():
    with patch("litellm.guardrail_name_config_map", _config_map()):
        assert (
            await should_proceed_based_on_metadata(
                data={"metadata": {"guardrails": {"prompt_injection": False}}},
                guardrail_name="lakera_prompt_injection",
            )
            is False
        )


@pytest.mark.asyncio
async def test_guardrail_runs_when_explicitly_enabled():
    with patch("litellm.guardrail_name_config_map", _config_map()):
        assert (
            await should_proceed_based_on_metadata(
                data={"metadata": {"guardrails": {"prompt_injection": True}}},
                guardrail_name="lakera_prompt_injection",
            )
            is True
        )


@pytest.mark.asyncio
async def test_non_default_on_guardrail_stays_opt_in():
    """
    A guardrail that is not default_on must not start running just because it was
    left unnamed in a per-request override
    """
    with patch("litellm.guardrail_name_config_map", _config_map()):
        assert (
            await should_proceed_based_on_metadata(
                data={"metadata": {"guardrails": {"prompt_injection": True}}},
                guardrail_name="presidio",
            )
            is False
        )


@pytest.mark.asyncio
async def test_no_guardrails_metadata_runs_all():
    with patch("litellm.guardrail_name_config_map", _config_map()):
        assert (
            await should_proceed_based_on_metadata(
                data={"metadata": {}},
                guardrail_name="lakera_prompt_injection",
            )
            is True
        )


@pytest.mark.asyncio
async def test_list_form_guardrails_ignored_by_v1_helper():
    with patch("litellm.guardrail_name_config_map", _config_map()):
        assert (
            await should_proceed_based_on_metadata(
                data={"metadata": {"guardrails": ["pii_masking"]}},
                guardrail_name="lakera_prompt_injection",
            )
            is True
        )
