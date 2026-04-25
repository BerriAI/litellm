"""
Tests for project_alias and project_id tracking through callback kwargs / metadata.

Verifies that project_alias flows from UserAPIKeyAuth through the metadata pipeline
to StandardLoggingMetadata, mirroring how team_alias already works.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.proxy._types import LiteLLM_VerificationTokenView, UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.types.utils import StandardLoggingUserAPIKeyMetadata


class TestProjectAliasOnTypes:
    """project_alias field exists on the relevant types."""

    def test_verification_token_view_has_project_alias(self):
        token_view = LiteLLM_VerificationTokenView(
            token="test-token",
            project_id="proj-123",
            project_alias="My Project",
        )
        assert token_view.project_alias == "My Project"

    def test_verification_token_view_project_alias_defaults_none(self):
        token_view = LiteLLM_VerificationTokenView(token="test-token")
        assert token_view.project_alias is None

    def test_user_api_key_auth_inherits_project_alias(self):
        """UserAPIKeyAuth extends LiteLLM_VerificationTokenView, so it gets project_alias."""
        auth = UserAPIKeyAuth(
            api_key="sk-test",
            project_id="proj-1",
            project_alias="billing-service",
        )
        assert auth.project_alias == "billing-service"

    def test_standard_logging_metadata_has_project_alias_field(self):
        metadata = StandardLoggingUserAPIKeyMetadata(
            user_api_key_hash="hash",
            user_api_key_alias=None,
            user_api_key_spend=None,
            user_api_key_max_budget=None,
            user_api_key_budget_reset_at=None,
            user_api_key_org_id=None,
            user_api_key_team_id=None,
            user_api_key_project_id="proj-1",
            user_api_key_project_alias="billing-service",
            user_api_key_user_id=None,
            user_api_key_user_email=None,
            user_api_key_team_alias=None,
            user_api_key_end_user_id=None,
            user_api_key_request_route=None,
            user_api_key_auth_metadata=None,
        )
        assert metadata["user_api_key_project_alias"] == "billing-service"


class TestProjectAliasThroughMetadataPipeline:
    """project_alias flows through the full metadata pipeline."""

    def test_get_sanitized_user_information_includes_project_alias(self):
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-hashed",
            project_id="proj-123",
            project_alias="My Cool Project",
            team_id="team-1",
            team_alias="my-team",
        )

        result = LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
            user_api_key_dict=user_api_key_dict
        )

        assert result["user_api_key_project_id"] == "proj-123"
        assert result["user_api_key_project_alias"] == "My Cool Project"

    def test_get_sanitized_user_information_project_alias_none_when_no_project(self):
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-hashed")

        result = LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
            user_api_key_dict=user_api_key_dict
        )

        assert result["user_api_key_project_id"] is None
        assert result["user_api_key_project_alias"] is None

    def test_project_alias_flows_to_standard_logging_metadata(self):
        """get_standard_logging_metadata picks up project_alias from input metadata."""
        metadata = {
            "user_api_key_project_id": "proj-123",
            "user_api_key_project_alias": "My Cool Project",
            "user_api_key_team_id": "team-1",
            "user_api_key_team_alias": "my-team",
        }

        result = StandardLoggingPayloadSetup.get_standard_logging_metadata(metadata)
        assert result["user_api_key_project_alias"] == "My Cool Project"

    def test_project_alias_defaults_to_none_in_logging_metadata(self):
        result = StandardLoggingPayloadSetup.get_standard_logging_metadata({})
        assert result["user_api_key_project_alias"] is None

    def test_end_to_end_project_alias_flow(self):
        """Full flow: UserAPIKeyAuth -> get_sanitized -> get_standard_logging_metadata."""
        auth = UserAPIKeyAuth(
            api_key="sk-test",
            project_id="proj-abc",
            project_alias="analytics-pipeline",
            team_id="team-1",
            team_alias="data-team",
        )

        # Step 1: Auth → sanitized metadata
        sanitized = LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
            user_api_key_dict=auth
        )

        # Step 2: Sanitized metadata → standard logging metadata
        logging_metadata = StandardLoggingPayloadSetup.get_standard_logging_metadata(
            dict(sanitized)
        )

        assert logging_metadata["user_api_key_project_id"] == "proj-abc"
        assert logging_metadata["user_api_key_project_alias"] == "analytics-pipeline"
        assert logging_metadata["user_api_key_team_id"] == "team-1"
        assert logging_metadata["user_api_key_team_alias"] == "data-team"
