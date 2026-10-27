"""The beta allowlist row is picked per deployment: an ``anthropic`` deployment on a Foundry host gets ``azure_ai``."""

import json
import os
from typing import Final

import pytest

import litellm
from litellm import anthropic_beta_headers_manager
from litellm.anthropic_beta_headers_manager import (
    is_foundry_api_base,
    resolve_beta_provider,
    update_headers_with_filtered_beta,
    update_request_with_filtered_beta,
)

FOUNDRY_API_BASE: Final = "https://my-resource.services.ai.azure.com/anthropic"
DIRECT_API_BASE: Final = "https://api.anthropic.com"
DIRECT_ONLY_BETA: Final = "advisor-tool-2026-03-01"
PRESERVED_THINKING_BETAS: Final = (
    "thinking-binding-controls-2026-08-01",
    "mid-conversation-system-clear-at-2026-08-21",
    "mid-conversation-tool-changes-2026-07-01",
    "mid-conversation-output-config-2026-07-01",
    "thinking-display-updates-2026-08-18",
)


@pytest.fixture(autouse=True)
def _bundled_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    anthropic_beta_headers_manager._BETA_HEADERS_CONFIG = None


def _bundled_rows() -> dict:
    with open(os.path.join(os.path.dirname(litellm.__file__), "anthropic_beta_headers_config.json")) as f:
        return json.load(f)


@pytest.mark.parametrize(
    ("api_base", "expected"),
    (
        (FOUNDRY_API_BASE, True),
        ("https://my-resource.openai.azure.com", True),
        ("my-resource.services.ai.azure.com/anthropic", True),
        (DIRECT_API_BASE, False),
        ("https://services.ai.azure.com.example.com", False),
        ("", False),
        (None, False),
    ),
)
def test_is_foundry_api_base_matches_only_foundry_hosts(api_base: str | None, expected: bool) -> None:
    assert is_foundry_api_base(api_base) is expected


@pytest.mark.parametrize(
    ("provider", "api_base", "expected"),
    (
        ("anthropic", FOUNDRY_API_BASE, "azure_ai"),
        ("anthropic", DIRECT_API_BASE, "anthropic"),
        ("anthropic", None, "anthropic"),
        ("vertex_ai", FOUNDRY_API_BASE, "vertex_ai"),
        ("bedrock_converse", None, "bedrock_converse"),
    ),
)
def test_resolve_beta_provider_picks_the_row_per_deployment(provider: str, api_base: str | None, expected: str) -> None:
    assert resolve_beta_provider(provider, api_base) == expected


def test_foundry_deployment_headers_follow_the_azure_ai_row() -> None:
    headers: Final = {"anthropic-beta": f"{DIRECT_ONLY_BETA},thinking-binding-controls-2026-08-01"}
    result: Final = update_headers_with_filtered_beta(headers=headers, provider="anthropic", api_base=FOUNDRY_API_BASE)
    assert result == {"anthropic-beta": "thinking-binding-controls-2026-08-01"}


def test_direct_deployment_headers_keep_every_anthropic_row_beta() -> None:
    headers: Final = {"anthropic-beta": f"{DIRECT_ONLY_BETA},thinking-binding-controls-2026-08-01"}
    result: Final = update_headers_with_filtered_beta(headers=headers, provider="anthropic", api_base=DIRECT_API_BASE)
    assert result == {"anthropic-beta": f"{DIRECT_ONLY_BETA},thinking-binding-controls-2026-08-01"}


def test_foundry_deployment_request_filters_header_and_body_betas_together() -> None:
    headers, data = update_request_with_filtered_beta(
        headers={"anthropic-beta": DIRECT_ONLY_BETA},
        request_data={"anthropic_beta": [DIRECT_ONLY_BETA, "thinking-binding-controls-2026-08-01"], "max_tokens": 8},
        provider="anthropic",
        api_base=FOUNDRY_API_BASE,
    )
    assert headers == {}
    assert data == {"anthropic_beta": ["thinking-binding-controls-2026-08-01"], "max_tokens": 8}


@pytest.mark.parametrize("provider", ("anthropic", "azure_ai", "vertex_ai"))
@pytest.mark.parametrize("beta", PRESERVED_THINKING_BETAS)
def test_live_verified_platforms_forward_every_preserved_thinking_beta(provider: str, beta: str) -> None:
    assert _bundled_rows()[provider][beta] == beta


@pytest.mark.parametrize("provider", ("bedrock", "bedrock_converse", "bedrock_mantle", "databricks"))
def test_unverified_platforms_hold_back_the_system_clear_beta(provider: str) -> None:
    assert _bundled_rows()[provider]["mid-conversation-system-clear-at-2026-08-21"] is None
    assert (
        update_headers_with_filtered_beta(
            headers={"anthropic-beta": "mid-conversation-system-clear-at-2026-08-21"}, provider=provider
        )
        == {}
    )
