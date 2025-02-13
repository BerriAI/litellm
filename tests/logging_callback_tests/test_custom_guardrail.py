"""
Test custom guardrail + unit tests for guardrails
"""

import io
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import gzip
import json
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.custom_guardrail import CustomGuardrail


from typing import Any, Dict, List, Literal, Optional, Union

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_helpers import should_proceed_based_on_metadata
from litellm.types.guardrails import GuardrailEventHooks
from litellm.proxy.guardrails.guardrail_endpoints import _get_guardrails_list_response
from litellm.types.guardrails import GuardrailInfoResponse, ListGuardrailsResponse


def test_get_guardrail_from_metadata():
    guardrail = CustomGuardrail(guardrail_name="test-guardrail")

    # Test with empty metadata
    assert guardrail.get_guardrail_from_metadata({}) == []

    # Test with guardrails in metadata
    data = {"metadata": {"guardrails": ["guardrail1", "guardrail2"]}}
    assert guardrail.get_guardrail_from_metadata(data) == ["guardrail1", "guardrail2"]

    # Test with dict guardrails
    data = {
        "metadata": {
            "guardrails": [{"test-guardrail": {"extra_body": {"key": "value"}}}]
        }
    }
    assert guardrail.get_guardrail_from_metadata(data) == [
        {"test-guardrail": {"extra_body": {"key": "value"}}}
    ]


def test_guardrail_is_in_requested_guardrails():
    guardrail = CustomGuardrail(guardrail_name="test-guardrail")

    # Test with string list
    assert (
        guardrail._guardrail_is_in_requested_guardrails(["test-guardrail", "other"])
        == True
    )
    assert guardrail._guardrail_is_in_requested_guardrails(["other"]) == False

    # Test with dict list
    assert (
        guardrail._guardrail_is_in_requested_guardrails(
            [{"test-guardrail": {"extra_body": {"extra_key": "extra_value"}}}]
        )
        == True
    )
    assert (
        guardrail._guardrail_is_in_requested_guardrails(
            [
                {
                    "other-guardrail": {"extra_body": {"extra_key": "extra_value"}},
                    "test-guardrail": {"extra_body": {"extra_key": "extra_value"}},
                }
            ]
        )
        == True
    )
    assert (
        guardrail._guardrail_is_in_requested_guardrails(
            [{"other-guardrail": {"extra_body": {"extra_key": "extra_value"}}}]
        )
        == False
    )


def test_should_run_guardrail():
    guardrail = CustomGuardrail(
        guardrail_name="test-guardrail", event_hook=GuardrailEventHooks.pre_call
    )

    # Test matching event hook and guardrail
    assert (
        guardrail.should_run_guardrail(
            {"metadata": {"guardrails": ["test-guardrail"]}},
            GuardrailEventHooks.pre_call,
        )
        == True
    )

    # Test non-matching event hook
    assert (
        guardrail.should_run_guardrail(
            {"metadata": {"guardrails": ["test-guardrail"]}},
            GuardrailEventHooks.during_call,
        )
        == False
    )

    # Test guardrail not in requested list
    assert (
        guardrail.should_run_guardrail(
            {"metadata": {"guardrails": ["other-guardrail"]}},
            GuardrailEventHooks.pre_call,
        )
        == False
    )


def test_get_guardrail_dynamic_request_body_params():
    guardrail = CustomGuardrail(guardrail_name="test-guardrail")

    # Test with no extra_body
    data = {"metadata": {"guardrails": [{"test-guardrail": {}}]}}
    assert guardrail.get_guardrail_dynamic_request_body_params(data) == {}

    # Test with extra_body
    data = {
        "metadata": {
            "guardrails": [{"test-guardrail": {"extra_body": {"key": "value"}}}]
        }
    }
    assert guardrail.get_guardrail_dynamic_request_body_params(data) == {"key": "value"}

    # Test with non-matching guardrail
    data = {
        "metadata": {
            "guardrails": [{"other-guardrail": {"extra_body": {"key": "value"}}}]
        }
    }
    assert guardrail.get_guardrail_dynamic_request_body_params(data) == {}


def test_get_guardrails_list_response():
    # Test case 1: Valid guardrails config
    sample_config = [
        {
            "guardrail_name": "test-guard",
            "litellm_params": {
                "guardrail": "test-guard",
                "mode": "pre_call",
                "api_key": "test-api-key",
                "api_base": "test-api-base",
            },
            "guardrail_info": {
                "params": [
                    {
                        "name": "toxicity_score",
                        "type": "float",
                        "description": "Score between 0-1",
                    }
                ]
            },
        }
    ]

    response = _get_guardrails_list_response(sample_config)
    assert isinstance(response, ListGuardrailsResponse)
    assert len(response.guardrails) == 1
    assert response.guardrails[0].guardrail_name == "test-guard"
    assert response.guardrails[0].guardrail_info == {
        "params": [
            {
                "name": "toxicity_score",
                "type": "float",
                "description": "Score between 0-1",
            }
        ]
    }

    # Test case 2: Empty guardrails config
    empty_response = _get_guardrails_list_response([])
    assert isinstance(empty_response, ListGuardrailsResponse)
    assert len(empty_response.guardrails) == 0

    # Test case 3: Missing optional fields
    minimal_config = [
        {
            "guardrail_name": "minimal-guard",
            "litellm_params": {"guardrail": "minimal-guard", "mode": "pre_call"},
        }
    ]
    minimal_response = _get_guardrails_list_response(minimal_config)
    assert isinstance(minimal_response, ListGuardrailsResponse)
    assert len(minimal_response.guardrails) == 1
    assert minimal_response.guardrails[0].guardrail_name == "minimal-guard"
    assert minimal_response.guardrails[0].guardrail_info is None


def test_default_on_guardrail():
    # Test guardrail with default_on=True
    guardrail = CustomGuardrail(
        guardrail_name="test-guardrail",
        event_hook=GuardrailEventHooks.pre_call,
        default_on=True,
    )

    # Should run when event_type matches, even without explicit request
    assert (
        guardrail.should_run_guardrail(
            {"metadata": {}},  # Empty metadata, no explicit guardrail request
            GuardrailEventHooks.pre_call,
        )
        == True
    )

    # Should not run when event_type doesn't match
    assert (
        guardrail.should_run_guardrail({"metadata": {}}, GuardrailEventHooks.post_call)
        == False
    )

    # Should run even when different guardrail explicitly requested
    # run test-guardrail-5 and test-guardrail
    assert (
        guardrail.should_run_guardrail(
            {"metadata": {"guardrails": ["test-guardrail-5"]}},
            GuardrailEventHooks.pre_call,
        )
        == True
    )

    assert (
        guardrail.should_run_guardrail(
            {"metadata": {"guardrails": []}},
            GuardrailEventHooks.pre_call,
        )
        == True
    )
