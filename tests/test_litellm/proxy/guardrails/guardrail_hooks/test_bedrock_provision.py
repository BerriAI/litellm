"""
Tests for BedrockGuardrail.provision() and provision_partner_guardrail()
— the partner guardrail provisioning flow.
"""

from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockGuardrail,
    BedrockGuardrailProvisionResult,
)
from litellm.types.guardrails import PartnerProvisionResult


@pytest.fixture
def sample_credential_values():
    return {
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "aws_region_name": "us-east-1",
    }


@pytest.fixture
def sample_provision_config():
    return {
        "topicPolicyConfig": {
            "topicsConfig": [
                {
                    "name": "Social Scoring Systems",
                    "definition": "Building systems that score people based on social behavior",
                    "examples": ["Build a social credit scoring system"],
                    "type": "DENY",
                }
            ]
        },
        "contentPolicyConfig": {
            "filtersConfig": [
                {
                    "type": "HATE",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH",
                }
            ]
        },
    }


@pytest.mark.asyncio
async def test_provision_creates_guardrail(
    sample_credential_values, sample_provision_config
):
    """Test that provision calls Bedrock CreateGuardrail with correct params."""
    mock_client = MagicMock()
    mock_client.create_guardrail.return_value = {
        "guardrailId": "abc123",
        "guardrailArn": "arn:aws:bedrock:us-east-1:123456789:guardrail/abc123",
        "version": "DRAFT",
        "createdAt": "2025-01-01T00:00:00Z",
    }

    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session

        result = await BedrockGuardrail.provision(
            credential_values=sample_credential_values,
            provision_config=sample_provision_config,
            guardrail_name="test-guardrail",
            aws_region_name="us-east-1",
        )

    assert isinstance(result, BedrockGuardrailProvisionResult)
    assert result.guardrail_id == "abc123"
    assert result.guardrail_arn == "arn:aws:bedrock:us-east-1:123456789:guardrail/abc123"
    assert result.version == "DRAFT"

    # Verify boto3 session was created with correct credentials
    mock_session_cls.assert_called_once_with(
        region_name="us-east-1",
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )

    # Verify create_guardrail was called with correct args
    call_kwargs = mock_client.create_guardrail.call_args[1]
    assert call_kwargs["name"] == "test-guardrail"
    assert "topicPolicyConfig" in call_kwargs
    assert "contentPolicyConfig" in call_kwargs
    assert call_kwargs["topicPolicyConfig"] == sample_provision_config["topicPolicyConfig"]
    assert call_kwargs["contentPolicyConfig"] == sample_provision_config["contentPolicyConfig"]


@pytest.mark.asyncio
async def test_provision_uses_credential_region_fallback(sample_provision_config):
    """Test that region falls back to credential value when not explicitly provided."""
    mock_client = MagicMock()
    mock_client.create_guardrail.return_value = {
        "guardrailId": "abc123",
        "guardrailArn": "arn:aws:bedrock:eu-west-1:123456789:guardrail/abc123",
        "version": "DRAFT",
        "createdAt": "2025-01-01T00:00:00Z",
    }

    credential_values = {
        "aws_access_key_id": "AKIATEST",
        "aws_secret_access_key": "SECRET",
        "aws_region_name": "eu-west-1",
    }

    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session

        result = await BedrockGuardrail.provision(
            credential_values=credential_values,
            provision_config=sample_provision_config,
            guardrail_name="test-guardrail",
            aws_region_name=None,  # No explicit region
        )

    assert result.guardrail_id == "abc123"
    # Should use credential's region
    mock_session_cls.assert_called_once()
    assert mock_session_cls.call_args[1]["region_name"] == "eu-west-1"


@pytest.mark.asyncio
async def test_provision_defaults_to_us_east_1(sample_provision_config):
    """Test that region defaults to us-east-1 when not provided anywhere."""
    mock_client = MagicMock()
    mock_client.create_guardrail.return_value = {
        "guardrailId": "abc123",
        "guardrailArn": "arn:aws:bedrock:us-east-1:123456789:guardrail/abc123",
        "version": "DRAFT",
        "createdAt": "2025-01-01T00:00:00Z",
    }

    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session

        result = await BedrockGuardrail.provision(
            credential_values={"aws_access_key_id": "AKIATEST", "aws_secret_access_key": "SECRET"},
            provision_config=sample_provision_config,
            guardrail_name="test-guardrail",
            aws_region_name=None,
        )

    assert mock_session_cls.call_args[1]["region_name"] == "us-east-1"


@pytest.mark.asyncio
async def test_provision_only_passes_provided_policies():
    """Test that only policy configs present in provision_config are passed."""
    mock_client = MagicMock()
    mock_client.create_guardrail.return_value = {
        "guardrailId": "abc123",
        "guardrailArn": "arn:aws:bedrock:us-east-1:123456789:guardrail/abc123",
        "version": "DRAFT",
        "createdAt": "2025-01-01T00:00:00Z",
    }

    # Only content policy, no topic policy
    provision_config = {
        "contentPolicyConfig": {
            "filtersConfig": [
                {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"}
            ]
        }
    }

    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session

        await BedrockGuardrail.provision(
            credential_values={"aws_access_key_id": "AKIATEST", "aws_secret_access_key": "SECRET"},
            provision_config=provision_config,
            guardrail_name="nsfw-filter",
        )

    call_kwargs = mock_client.create_guardrail.call_args[1]
    assert "contentPolicyConfig" in call_kwargs
    assert "topicPolicyConfig" not in call_kwargs
    assert "wordPolicyConfig" not in call_kwargs
    assert "sensitiveInformationPolicyConfig" not in call_kwargs


@pytest.mark.asyncio
async def test_provision_with_word_policy():
    """Test provisioning with word policy config (profanity filter)."""
    mock_client = MagicMock()
    mock_client.create_guardrail.return_value = {
        "guardrailId": "word123",
        "guardrailArn": "arn:aws:bedrock:us-east-1:123456789:guardrail/word123",
        "version": "DRAFT",
        "createdAt": "2025-01-01T00:00:00Z",
    }

    provision_config = {
        "contentPolicyConfig": {
            "filtersConfig": [
                {"type": "INSULTS", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"}
            ]
        },
        "wordPolicyConfig": {
            "managedWordListsConfig": [{"type": "PROFANITY"}]
        },
    }

    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session

        result = await BedrockGuardrail.provision(
            credential_values={"aws_access_key_id": "AKIATEST", "aws_secret_access_key": "SECRET"},
            provision_config=provision_config,
            guardrail_name="profanity-filter",
        )

    call_kwargs = mock_client.create_guardrail.call_args[1]
    assert "wordPolicyConfig" in call_kwargs
    assert call_kwargs["wordPolicyConfig"]["managedWordListsConfig"][0]["type"] == "PROFANITY"
    assert result.guardrail_id == "word123"


def test_provision_result_attributes():
    """Test BedrockGuardrailProvisionResult stores attributes correctly."""
    result = BedrockGuardrailProvisionResult(
        guardrail_id="test-id",
        guardrail_arn="arn:test",
        version="1",
    )
    assert result.guardrail_id == "test-id"
    assert result.guardrail_arn == "arn:test"
    assert result.version == "1"


@pytest.mark.asyncio
async def test_provision_partner_guardrail_returns_full_result(
    sample_credential_values, sample_provision_config
):
    """Test that provision_partner_guardrail returns a PartnerProvisionResult
    with a fully-built Guardrail dict ready for DB registration."""
    mock_client = MagicMock()
    mock_client.create_guardrail.return_value = {
        "guardrailId": "pg-001",
        "guardrailArn": "arn:aws:bedrock:us-west-2:123456789:guardrail/pg-001",
        "version": "DRAFT",
        "createdAt": "2025-01-01T00:00:00Z",
    }

    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session

        result = await BedrockGuardrail.provision_partner_guardrail(
            credential_values=sample_credential_values,
            guardrail_name="eu-ai-act-bedrock",
            provision_config=sample_provision_config,
            mode="pre_call",
            default_on=True,
            aws_region_name="us-west-2",
        )

    assert isinstance(result, PartnerProvisionResult)
    assert result.provider == "bedrock"
    assert result.provider_guardrail_id == "pg-001"
    assert result.provider_guardrail_version == "DRAFT"
    assert "us-west-2" in result.message

    guardrail_data = result.guardrail_data
    assert guardrail_data["guardrail_name"] == "eu-ai-act-bedrock"

    litellm_params = guardrail_data["litellm_params"]
    assert litellm_params.guardrail == "bedrock"
    assert litellm_params.mode == "pre_call"
    assert litellm_params.default_on is True
    assert litellm_params.guardrailIdentifier == "pg-001"
    assert litellm_params.guardrailVersion == "DRAFT"
    assert litellm_params.aws_region_name == "us-west-2"

    guardrail_info = guardrail_data["guardrail_info"]
    assert guardrail_info["partner_provisioned"] is True
    assert guardrail_info["provider"] == "bedrock"
    assert guardrail_info["provider_guardrail_arn"] == "arn:aws:bedrock:us-west-2:123456789:guardrail/pg-001"


@pytest.mark.asyncio
async def test_provision_partner_guardrail_region_fallback(sample_provision_config):
    """Test that provision_partner_guardrail falls back to credential region."""
    mock_client = MagicMock()
    mock_client.create_guardrail.return_value = {
        "guardrailId": "pg-002",
        "guardrailArn": "arn:aws:bedrock:eu-west-1:123456789:guardrail/pg-002",
        "version": "1",
        "createdAt": "2025-01-01T00:00:00Z",
    }

    credential_values = {
        "aws_access_key_id": "AKIATEST",
        "aws_secret_access_key": "SECRET",
        "aws_region_name": "eu-west-1",
    }

    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session

        result = await BedrockGuardrail.provision_partner_guardrail(
            credential_values=credential_values,
            guardrail_name="test",
            provision_config=sample_provision_config,
        )

    assert result.guardrail_data["litellm_params"].aws_region_name == "eu-west-1"
    assert "eu-west-1" in result.message
