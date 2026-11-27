from typing import Final

import pytest

from litellm.litellm_core_utils.get_provider_specific_headers import (
    ProviderSpecificHeaderUtils,
    override_headers,
)
from litellm.types.utils import ProviderSpecificHeader


class TestProviderSpecificHeaderUtils:
    def test_get_provider_specific_headers_matching_provider(self):
        """Test that the method returns extra_headers when custom_llm_provider matches."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "openai",
            "extra_headers": {
                "Authorization": "Bearer token123",
                "Custom-Header": "value",
            },
        }
        custom_llm_provider = "openai"

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, custom_llm_provider
        )

        expected = {"Authorization": "Bearer token123", "Custom-Header": "value"}
        assert result == expected

    def test_get_provider_specific_headers_no_match_or_none(self):
        """Test that the method returns empty dict when provider doesn't match or is None."""
        # Test case 1: Provider doesn't match
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic",
            "extra_headers": {"Authorization": "Bearer token123"},
        }
        custom_llm_provider = "openai"

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, custom_llm_provider
        )
        assert result == {}

        # Test case 2: provider_specific_header is None
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            None, "openai"
        )
        assert result == {}

    def test_get_provider_specific_headers_multi_provider_anthropic_to_bedrock(self):
        """Test that anthropic headers work with bedrock provider (multi-provider support)."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic,bedrock,bedrock_converse,vertex_ai",
            "extra_headers": {"anthropic-beta": "context-1m-2025-08-07"},
        }

        # Test bedrock provider
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "bedrock"
        )
        assert result == {"anthropic-beta": "context-1m-2025-08-07"}

        # Test anthropic provider
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "anthropic"
        )
        assert result == {"anthropic-beta": "context-1m-2025-08-07"}

        # Test bedrock_converse provider
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "bedrock_converse"
        )
        assert result == {"anthropic-beta": "context-1m-2025-08-07"}

        # Test vertex_ai provider
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "vertex_ai"
        )
        assert result == {"anthropic-beta": "context-1m-2025-08-07"}

    def test_get_provider_specific_headers_multi_provider_no_match(self):
        """Test that non-listed providers return empty dict with multi-provider list."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic,bedrock,vertex_ai",
            "extra_headers": {"anthropic-beta": "test"},
        }

        # Test provider not in list
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "openai"
        )
        assert result == {}

    def test_get_provider_specific_headers_with_spaces(self):
        """Test that comma-separated list with spaces is handled correctly."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic, bedrock, vertex_ai",
            "extra_headers": {"anthropic-beta": "test"},
        }

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "bedrock"
        )
        assert result == {"anthropic-beta": "test"}

    def test_get_provider_specific_headers_none_custom_llm_provider(self):
        """Test that None custom_llm_provider returns empty dict."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic",
            "extra_headers": {"anthropic-beta": "test"},
        }

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, None
        )
        assert result == {}

    def test_get_provider_specific_headers_scopes_each_entry_independently(self):
        """Entries in a list each carry their own provider scope."""
        scoped_headers: list[ProviderSpecificHeader] = [
            {
                "custom_llm_provider": "anthropic,bedrock,vertex_ai",
                "extra_headers": {"anthropic-beta": "context-1m-2025-08-07"},
            },
            {
                "custom_llm_provider": "anthropic",
                "extra_headers": {"authorization": "Bearer sk-ant-oat01-fake-token"},
            },
        ]

        assert ProviderSpecificHeaderUtils.get_provider_specific_headers(
            scoped_headers, "anthropic"
        ) == {
            "anthropic-beta": "context-1m-2025-08-07",
            "authorization": "Bearer sk-ant-oat01-fake-token",
        }
        assert ProviderSpecificHeaderUtils.get_provider_specific_headers(
            scoped_headers, "bedrock"
        ) == {"anthropic-beta": "context-1m-2025-08-07"}
        assert (
            ProviderSpecificHeaderUtils.get_provider_specific_headers(
                scoped_headers, "openai"
            )
            == {}
        )

    def test_get_provider_specific_headers_empty_list(self):
        """An empty list of scoped entries contributes nothing."""
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers([], "anthropic")
        assert result == {}

    def test_override_headers_replaces_any_casing_but_keeps_a_caller_beta(self):
        merged: Final = override_headers(
            {"Anthropic-Beta": "caller-beta", "Anthropic-User-Profile-Id": "uprof_other"},
            {"anthropic-beta": "client-header-beta", "anthropic-user-profile-id": "uprof_pinned"},
        )

        assert merged == {
            "Anthropic-Beta": "caller-beta",
            "anthropic-beta": "client-header-beta",
            "anthropic-user-profile-id": "uprof_pinned",
        }

_PROFILE_ON_AZURE_AI: Final[ProviderSpecificHeader] = {
    "custom_llm_provider": "azure_ai",
    "extra_headers": {"anthropic-user-profile-id": "uprof_client"},
    "user_profile_id_models_only": True,
}


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-sonnet-5-5", {"anthropic-user-profile-id": "uprof_client"}),
        ("claude-opus-5-5", {"anthropic-user-profile-id": "uprof_client"}),
        ("mistral-large-3", {}),
        ("claude-unlisted-deployment", {}),
        (None, {}),
    ],
    ids=["flagged_claude", "another_flagged_claude", "mapped_without_flag", "unmapped_claude_name", "unknown_model"],
)
def test_a_profile_id_entry_reaches_only_models_the_cost_map_flags(model: str | None, expected: dict[str, str]) -> None:
    assert (
        ProviderSpecificHeaderUtils.get_provider_specific_headers(_PROFILE_ON_AZURE_AI, "azure_ai", model=model)
        == expected
    )


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-sonnet-5-5", {}),
        ("claude-opus-5-5", {}),
        ("mistral-large-3", {"authorization": "Bearer caller"}),
        ("claude-unlisted-deployment", {"authorization": "Bearer caller"}),
        (None, {"authorization": "Bearer caller"}),
    ],
    ids=["flagged_claude", "another_flagged_claude", "mapped_without_flag", "unmapped_claude_name", "unknown_model"],
)
def test_a_non_profile_id_entry_skips_models_the_cost_map_flags(model: str | None, expected: dict[str, str]) -> None:
    entry: Final[ProviderSpecificHeader] = {
        "custom_llm_provider": "azure_ai",
        "extra_headers": {"authorization": "Bearer caller"},
        "non_user_profile_id_models_only": True,
    }
    assert ProviderSpecificHeaderUtils.get_provider_specific_headers(entry, "azure_ai", model=model) == expected


@pytest.mark.parametrize(
    "base_model,expected",
    [
        ("azure_ai/claude-sonnet-5-5", {"anthropic-user-profile-id": "uprof_client"}),
        (None, {"authorization": "Bearer caller"}),
    ],
    ids=["base_model_flagged", "no_base_model"],
)
def test_an_unlisted_deployment_takes_the_profile_id_when_its_base_model_is_flagged(
    base_model: str | None, expected: dict[str, str]
) -> None:
    caller_credential: Final[ProviderSpecificHeader] = {
        "custom_llm_provider": "azure_ai",
        "extra_headers": {"authorization": "Bearer caller"},
        "non_user_profile_id_models_only": True,
    }
    assert (
        ProviderSpecificHeaderUtils.get_provider_specific_headers(
            [_PROFILE_ON_AZURE_AI, caller_credential],
            "azure_ai",
            model="claude-unlisted-deployment",
            base_model=base_model,
        )
        == expected
    )


@pytest.mark.parametrize(
    "yields_to_configured,configured_headers,expected",
    [
        (True, {"Anthropic-User-Profile-Id": "uprof_deployment"}, {}),
        (True, {"x-tenant": "a"}, {"anthropic-user-profile-id": "uprof_client"}),
        (True, None, {"anthropic-user-profile-id": "uprof_client"}),
        (False, {"anthropic-user-profile-id": "uprof_deployment"}, {"anthropic-user-profile-id": "uprof_client"}),
    ],
    ids=["yields_to_a_configured_header", "other_configured_header", "nothing_configured", "pinned_entry_overrides"],
)
def test_a_yielding_entry_never_replaces_a_configured_header(
    yields_to_configured: bool, configured_headers: dict[str, str] | None, expected: dict[str, str]
) -> None:
    scoped_header: Final[ProviderSpecificHeader] = {**_PROFILE_ON_AZURE_AI, "yields_to_configured": yields_to_configured}

    assert (
        ProviderSpecificHeaderUtils.get_provider_specific_headers(
            scoped_header, "azure_ai", model="claude-sonnet-5-5", configured_headers=configured_headers
        )
        == expected
    )


@pytest.mark.parametrize(
    "configured_headers,expected",
    [
        ({"X-Api-Key": "sk-ant-deployment"}, {"anthropic-user-profile-id": "uprof_pinned"}),
        ({"Authorization": "Bearer deployment"}, {"anthropic-user-profile-id": "uprof_pinned"}),
        ({"api-key": "foundry-deployment"}, {}),
        ({"x-tenant": "a"}, {}),
        (None, {}),
    ],
    ids=["configured_x_api_key", "configured_authorization", "unlisted_api_key", "no_credential", "nothing"],
)
def test_a_configured_credential_entry_applies_only_beside_a_listed_configured_credential(
    configured_headers: dict[str, str] | None, expected: dict[str, str]
) -> None:
    scoped_header: Final[ProviderSpecificHeader] = {
        "custom_llm_provider": "anthropic",
        "extra_headers": {"anthropic-user-profile-id": "uprof_pinned"},
        "configured_credential_headers": ("authorization", "X-Api-Key"),
    }

    assert (
        ProviderSpecificHeaderUtils.get_provider_specific_headers(
            scoped_header, "anthropic", model="claude-sonnet-5-5", configured_headers=configured_headers
        )
        == expected
    )


@pytest.mark.parametrize(
    "configured_headers,expected",
    [
        ({"Authorization": "Bearer sk-ant-oat01-deployment"}, {"anthropic-user-profile-id": "uprof_pinned"}),
        ({"authorization": "Bearer sk-ant-deployment"}, {}),
        ({"x-api-key": "Bearer sk-ant-oat01-deployment"}, {}),
    ],
    ids=["configured_oauth_token", "configured_api_key_bearer", "prefix_on_an_unlisted_header"],
)
def test_a_configured_credential_prefix_applies_only_beside_a_matching_value(
    configured_headers: dict[str, str], expected: dict[str, str]
) -> None:
    scoped_header: Final[ProviderSpecificHeader] = {
        "custom_llm_provider": "anthropic",
        "extra_headers": {"anthropic-user-profile-id": "uprof_pinned"},
        "configured_credential_headers": ("authorization",),
        "configured_credential_prefix": "Bearer sk-ant-oat",
    }

    assert (
        ProviderSpecificHeaderUtils.get_provider_specific_headers(
            scoped_header, "anthropic", model="claude-sonnet-5-5", configured_headers=configured_headers
        )
        == expected
    )


@pytest.mark.parametrize("without_configured_credential", [False, True])
@pytest.mark.parametrize(
    "configured_headers,configured_token_sent",
    [
        ({"authorization": "Bearer sk-ant-oat01-caller"}, False),
        ({"Authorization": "Bearer sk-ant-oat01-deployment"}, True),
        ({"authorization": "Bearer sk-ant-oat01-caller", "Authorization": "Bearer sk-ant-oat01-deployment"}, False),
        ({"Authorization": "Bearer sk-ant-oat01-deployment", "authorization": "Bearer sk-ant-oat01-caller"}, True),
        ({"authorization": "Bearer sk-ant-api03-caller", "Authorization": "Bearer sk-ant-oat01-deployment"}, True),
        ({}, False),
    ],
    ids=[
        "caller_token_survives_the_merge",
        "configured_token_survives_the_merge",
        "caller_token_first_of_both",
        "configured_token_first_of_both",
        "configured_token_behind_a_non_oauth_value",
        "no_token",
    ],
)
def test_an_entry_waits_on_whether_a_configured_token_that_is_not_the_callers_own_is_sent(
    configured_headers: dict[str, str], configured_token_sent: bool, without_configured_credential: bool
) -> None:
    scoped_header: Final[ProviderSpecificHeader] = {
        "custom_llm_provider": "anthropic",
        "extra_headers": {"anthropic-user-profile-id": "uprof_pinned"},
        "configured_credential_headers": ("authorization",),
        "configured_credential_prefix": "Bearer sk-ant-oat",
        "caller_credentials": ("Bearer sk-ant-oat01-caller",),
        "without_configured_credential": without_configured_credential,
    }

    assert ProviderSpecificHeaderUtils.get_provider_specific_headers(
        scoped_header, "anthropic", model="claude-sonnet-5-5", configured_headers=configured_headers
    ) == ({"anthropic-user-profile-id": "uprof_pinned"} if configured_token_sent != without_configured_credential else {})


@pytest.mark.parametrize(
    "configured_headers,expected",
    [
        ({"authorization": "Bearer sk-ant-oat01-deployment"}, {"authorization": "Bearer sk-ant-oat01-caller"}),
        (
            {"Authorization": "Bearer sk-ant-oat01-deployment"},
            {"authorization": "Bearer sk-ant-oat01-caller", "anthropic-user-profile-id": "uprof_pinned"},
        ),
        (
            {"authorization": "Bearer sk-ant-api03-caller", "Authorization": "Bearer sk-ant-oat01-deployment"},
            {"authorization": "Bearer sk-ant-oat01-caller"},
        ),
    ],
    ids=[
        "caller_replaces_the_configured_token",
        "configured_token_kept_under_another_name",
        "caller_token_replaces_a_value_ahead_of_the_configured_token",
    ],
)
def test_a_pin_applies_only_while_the_configured_credential_survives_the_merge(
    configured_headers: dict[str, str], expected: dict[str, str]
) -> None:
    scoped_headers: Final[tuple[ProviderSpecificHeader, ...]] = (
        {"custom_llm_provider": "anthropic", "extra_headers": {"authorization": "Bearer sk-ant-oat01-caller"}},
        {
            "custom_llm_provider": "anthropic",
            "extra_headers": {"anthropic-user-profile-id": "uprof_pinned"},
            "configured_credential_headers": ("authorization",),
            "configured_credential_prefix": "Bearer sk-ant-oat",
        },
    )

    assert (
        ProviderSpecificHeaderUtils.get_provider_specific_headers(
            scoped_headers, "anthropic", model="claude-sonnet-5-5", configured_headers=configured_headers
        )
        == expected
    )


@pytest.mark.parametrize(
    "headers,overrides,expected",
    [
        (
            {"Anthropic-User-Profile-Id": "uprof_caller", "x-tenant": "a"},
            {"anthropic-user-profile-id": "uprof_pinned"},
            {"x-tenant": "a", "anthropic-user-profile-id": "uprof_pinned"},
        ),
        (
            {"Authorization": "Bearer deployment", "X-Custom": "deployment"},
            {"authorization": "Bearer client", "x-custom": "client"},
            {
                "Authorization": "Bearer deployment",
                "X-Custom": "deployment",
                "authorization": "Bearer client",
                "x-custom": "client",
            },
        ),
    ],
    ids=["profile_id_in_any_case", "other_headers_keep_exact_case_merge"],
)
def test_only_the_profile_id_overrides_a_header_spelled_in_another_case(
    headers: dict[str, str], overrides: dict[str, str], expected: dict[str, str]
) -> None:
    assert override_headers(headers, overrides) == expected
