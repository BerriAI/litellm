import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

from pydantic.main import Model

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

from typing import Literal

import pytest
import litellm
import asyncio
import logging
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_metadata,
    StandardLoggingMetadata,
)


def all_fields_present(standard_logging_metadata: StandardLoggingMetadata):
    for field in StandardLoggingMetadata.__annotations__.keys():
        assert field in standard_logging_metadata


@pytest.mark.parametrize(
    "metadata_key, metadata_value",
    [
        ("user_api_key_alias", "test_alias"),
        ("user_api_key_hash", "test_hash"),
        ("user_api_key_team_id", "test_team_id"),
        ("user_api_key_user_id", "test_user_id"),
        ("user_api_key_team_alias", "test_team_alias"),
        ("spend_logs_metadata", {"key": "value"}),
        ("requester_ip_address", "127.0.0.1"),
        ("requester_metadata", {"user_agent": "test_agent"}),
    ],
)
def test_get_standard_logging_metadata(metadata_key, metadata_value):
    """
    Test that the get_standard_logging_metadata function correctly sets the metadata fields.

    All fields in StandardLoggingMetadata should ALWAYS be present.
    """
    metadata = {metadata_key: metadata_value}
    standard_logging_metadata = get_standard_logging_metadata(metadata)

    print("standard_logging_metadata", standard_logging_metadata)

    # Assert that all fields in StandardLoggingMetadata are present
    all_fields_present(standard_logging_metadata)

    # Assert that the specific metadata field is set correctly
    assert standard_logging_metadata[metadata_key] == metadata_value


def test_get_standard_logging_metadata_user_api_key_hash():
    valid_hash = "a" * 64  # 64 character string
    metadata = {"user_api_key": valid_hash}
    result = get_standard_logging_metadata(metadata)
    assert result["user_api_key_hash"] == valid_hash


def test_get_standard_logging_metadata_invalid_user_api_key():
    invalid_hash = "not_a_valid_hash"
    metadata = {"user_api_key": invalid_hash}
    result = get_standard_logging_metadata(metadata)
    all_fields_present(result)
    assert result["user_api_key_hash"] is None


def test_get_standard_logging_metadata_invalid_keys():
    metadata = {
        "user_api_key_alias": "test_alias",
        "invalid_key": "should_be_ignored",
        "another_invalid_key": 123,
    }
    result = get_standard_logging_metadata(metadata)
    all_fields_present(result)
    assert result["user_api_key_alias"] == "test_alias"
    assert "invalid_key" not in result
    assert "another_invalid_key" not in result
