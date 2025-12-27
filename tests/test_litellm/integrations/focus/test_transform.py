"""
Unit tests for FOCUS transform module.
"""

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.focus.transform import FOCUSTransformer
from litellm.types.integrations.focus import FOCUSRecord


class TestFOCUSTransformer:
    """Test suite for FOCUSTransformer class."""

    def test_init_default(self):
        """Test FOCUSTransformer initialization with defaults."""
        transformer = FOCUSTransformer()
        assert transformer.include_tags is True
        assert transformer.include_token_breakdown is True

    def test_init_custom(self):
        """Test FOCUSTransformer initialization with custom settings."""
        transformer = FOCUSTransformer(include_tags=False, include_token_breakdown=False)
        assert transformer.include_tags is False
        assert transformer.include_token_breakdown is False

    def test_transform_empty_dataframe(self):
        """Test transform method with empty DataFrame."""
        transformer = FOCUSTransformer()
        empty_df = pl.DataFrame()

        result = transformer.transform(empty_df)

        assert result.is_empty()
        assert isinstance(result, pl.DataFrame)

    def test_transform_with_zero_successful_requests(self):
        """Test transform method filters out records with zero successful_requests."""
        transformer = FOCUSTransformer()
        data = pl.DataFrame(
            {
                "date": ["2025-01-19"],
                "successful_requests": [0],
                "spend": [10.0],
                "model": ["gpt-4"],
                "custom_llm_provider": ["openai"],
            }
        )

        result = transformer.transform(data)

        assert result.is_empty()

    def test_transform_with_valid_data(self):
        """Test transform method with valid data."""
        transformer = FOCUSTransformer()
        data = pl.DataFrame(
            {
                "date": ["2025-01-19"],
                "successful_requests": [5],
                "spend": [10.0],
                "model": ["gpt-4"],
                "custom_llm_provider": ["openai"],
                "prompt_tokens": [100],
                "completion_tokens": [50],
                "team_id": ["team-123"],
                "team_alias": ["Engineering"],
            }
        )

        result = transformer.transform(data)

        assert len(result) == 1
        record = result.to_dicts()[0]
        assert record["BilledCost"] == 10.0
        assert record["ConsumedQuantity"] == 150  # 100 + 50
        assert record["ConsumedUnit"] == "Tokens"
        assert record["ProviderName"] == "OpenAI"
        assert record["ResourceType"] == "LLM"
        assert record["ServiceCategory"] == "AI and Machine Learning"
        assert record["ServiceName"] == "LLM Inference"
        assert record["SubAccountId"] == "team-123"
        assert record["SubAccountName"] == "Engineering"

    def test_transform_with_minimal_data(self):
        """Test transform method with minimal data."""
        transformer = FOCUSTransformer()
        data = pl.DataFrame(
            {
                "date": ["2025-01-19"],
                "successful_requests": [1],
                "spend": [0.5],
                "model": ["claude-3"],
                "custom_llm_provider": ["anthropic"],
            }
        )

        result = transformer.transform(data)

        assert len(result) == 1
        record = result.to_dicts()[0]
        assert record["BilledCost"] == 0.5
        assert record["ProviderName"] == "Anthropic"
        assert record["ResourceName"] == "claude-3"

    def test_create_focus_record(self):
        """Test _create_focus_record method with valid row data."""
        transformer = FOCUSTransformer()
        row = {
            "date": "2025-01-19",
            "spend": 10.5,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "model": "gpt-4",
            "model_group": "openai",
            "custom_llm_provider": "openai",
            "api_key": "sk-test123",
            "api_key_alias": "test-key",
            "team_id": "team-123",
            "team_alias": "Engineering",
            "user_id": "user-456",
            "api_requests": 5,
            "successful_requests": 5,
            "failed_requests": 0,
        }

        result = transformer._create_focus_record(row)

        assert isinstance(result, FOCUSRecord)
        assert result["BilledCost"] == 10.5
        assert result["EffectiveCost"] == 10.5
        assert result["ListCost"] == 10.5
        assert result["ConsumedQuantity"] == 150  # 100 + 50
        assert result["ConsumedUnit"] == "Tokens"
        assert result["ProviderName"] == "OpenAI"
        assert result["PublisherName"] == "LiteLLM"
        assert result["ResourceName"] == "gpt-4"
        assert result["ResourceType"] == "LLM"
        assert result["ChargeCategory"] == "Usage"
        assert result["ChargeClass"] == "Standard"
        assert result["SubAccountId"] == "team-123"
        assert result["SubAccountName"] == "Engineering"
        assert "Tags" in result

    def test_create_focus_record_with_tags(self):
        """Test _create_focus_record includes tags when enabled."""
        transformer = FOCUSTransformer(include_tags=True, include_token_breakdown=True)
        row = {
            "date": "2025-01-19",
            "spend": 5.0,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "model": "gpt-4o",
            "custom_llm_provider": "openai",
            "successful_requests": 3,
        }

        result = transformer._create_focus_record(row)

        assert "Tags" in result
        assert result["Tags"]["litellm:provider"] == "openai"
        assert result["Tags"]["litellm:model"] == "gpt-4o"
        assert result["Tags"]["litellm:prompt_tokens"] == "100"
        assert result["Tags"]["litellm:completion_tokens"] == "50"

    def test_create_focus_record_without_tags(self):
        """Test _create_focus_record excludes tags when disabled."""
        transformer = FOCUSTransformer(include_tags=False)
        row = {
            "date": "2025-01-19",
            "spend": 5.0,
            "model": "gpt-4",
            "custom_llm_provider": "openai",
        }

        result = transformer._create_focus_record(row)

        assert "Tags" not in result

    def test_parse_date_with_valid_string(self):
        """Test _parse_date method with valid date string."""
        transformer = FOCUSTransformer()

        result = transformer._parse_date("2025-01-19")

        assert isinstance(result, datetime)
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 19

    def test_parse_date_with_datetime_object(self):
        """Test _parse_date method with datetime object."""
        transformer = FOCUSTransformer()
        dt = datetime(2025, 1, 19)

        result = transformer._parse_date(dt)

        assert result == dt

    def test_parse_date_with_none(self):
        """Test _parse_date method with None."""
        transformer = FOCUSTransformer()

        result = transformer._parse_date(None)

        assert result is None

    def test_parse_date_with_invalid_string(self):
        """Test _parse_date method with invalid date string."""
        transformer = FOCUSTransformer()

        result = transformer._parse_date("invalid-date")

        assert result is None

    def test_parse_date_with_iso_format(self):
        """Test _parse_date method with ISO format string."""
        transformer = FOCUSTransformer()

        result = transformer._parse_date("2025-01-19T10:30:00Z")

        assert isinstance(result, datetime)
        assert result.year == 2025

    def test_normalize_provider_name(self):
        """Test provider name normalization."""
        transformer = FOCUSTransformer()

        assert transformer._normalize_provider_name("openai") == "OpenAI"
        assert transformer._normalize_provider_name("anthropic") == "Anthropic"
        assert transformer._normalize_provider_name("azure") == "Microsoft Azure"
        assert transformer._normalize_provider_name("bedrock") == "Amazon Web Services"
        assert transformer._normalize_provider_name("vertex_ai") == "Google Cloud"
        assert transformer._normalize_provider_name("gemini") == "Google"
        assert transformer._normalize_provider_name("unknown_provider") == "Unknown_Provider"

    def test_create_resource_id(self):
        """Test resource ID creation."""
        transformer = FOCUSTransformer()

        result = transformer._create_resource_id(
            provider="openai",
            model="gpt-4",
            team_id="team-123",
        )

        assert result == "litellm/openai/team-123/gpt-4"

    def test_create_resource_id_no_team(self):
        """Test resource ID creation without team."""
        transformer = FOCUSTransformer()

        result = transformer._create_resource_id(
            provider="openai",
            model="gpt-4",
            team_id=None,
        )

        assert result == "litellm/openai/default/gpt-4"

    def test_sanitize_id_component(self):
        """Test ID component sanitization."""
        transformer = FOCUSTransformer()

        assert transformer._sanitize_id_component("hello world") == "hello-world"
        assert transformer._sanitize_id_component("hello@world!") == "hello-world"
        assert transformer._sanitize_id_component("") == "unknown"
        assert transformer._sanitize_id_component("Hello-World") == "hello-world"

    def test_build_tags(self):
        """Test tags building."""
        transformer = FOCUSTransformer()

        tags = transformer._build_tags(
            provider="openai",
            model="gpt-4",
            model_group="openai-models",
            user_id="user-123",
            api_key_prefix="sk-12345",
            api_key_alias="test-key",
            prompt_tokens=100,
            completion_tokens=50,
            api_requests=5,
            successful_requests=4,
            failed_requests=1,
        )

        assert tags["litellm:provider"] == "openai"
        assert tags["litellm:model"] == "gpt-4"
        assert tags["litellm:model_group"] == "openai-models"
        assert tags["litellm:user_id"] == "user-123"
        assert tags["litellm:api_key_prefix"] == "sk-12345"
        assert tags["litellm:api_key_alias"] == "test-key"
        assert tags["litellm:prompt_tokens"] == "100"
        assert tags["litellm:completion_tokens"] == "50"
        assert tags["litellm:api_requests"] == "5"
        assert tags["litellm:successful_requests"] == "4"
        assert tags["litellm:failed_requests"] == "1"
