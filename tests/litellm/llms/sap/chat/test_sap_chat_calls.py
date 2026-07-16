"""
Live / blackbox tests for the SAP GenAI Hub orchestration provider.

These tests make real network calls to SAP AI Core.  They are skipped
automatically when AICORE_SERVICE_KEY is not set, so CI never runs them.

Run locally:
    export AICORE_SERVICE_KEY='{ ... json ... }'
    pytest tests/litellm/llms/sap/chat/test_sap_chat_calls.py -v

All tests use the default SAP model (gpt-4o via the orchestration service)
unless a specific model is needed for the feature under test.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("AICORE_SERVICE_KEY"),
    reason="Live SAP credentials not available (AICORE_SERVICE_KEY not set)",
)

_SAP_MODEL = "sap/gpt-4o"
_MESSAGES = [{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_live_response(response) -> None:
    """Shared sanity checks for a non-streaming ModelResponse."""
    assert response is not None
    assert response.choices, "response.choices is empty"
    content = response.choices[0].message.content or ""
    assert content.strip(), "response content is empty"


# ---------------------------------------------------------------------------
# Step 1 — URL resolution chain (multi-orch)
# ---------------------------------------------------------------------------


class TestURLResolution:
    """Verify the three-step URL resolution chain against a live AI Core tenant.

    Each test exercises exactly one step of the chain so a failure is
    immediately traceable to a single resolution path.
    """

    def test_url_via_optional_params(self):
        """Step 1: deployment_url in optional_params is used directly; no discovery."""
        import litellm

        # Retrieve the real deployment URL via discovery first so we have a
        # concrete value to pass back in.  This exercises discovery (step 3)
        # only as a setup prerequisite; the assertion target is step 1.
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        cfg = GenAIHubOrchestrationConfig()
        cfg.run_env_setup(os.environ["AICORE_SERVICE_KEY"])
        known_url = cfg.deployment_url  # triggers discovery once

        response = litellm.completion(
            model=_SAP_MODEL,
            messages=_MESSAGES,
            api_key=os.environ["AICORE_SERVICE_KEY"],
            deployment_url=known_url,  # passed as optional_param
        )
        _assert_live_response(response)

    def test_url_via_env_var(self):
        """Step 2: AICORE_ORCHESTRATION_DEPLOYMENT_URL env var is used; no discovery."""
        import litellm
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        # Resolve URL once to populate the env var for this test.
        cfg = GenAIHubOrchestrationConfig()
        cfg.run_env_setup(os.environ["AICORE_SERVICE_KEY"])
        known_url = cfg.deployment_url

        original = os.environ.get("AICORE_ORCHESTRATION_DEPLOYMENT_URL")
        try:
            os.environ["AICORE_ORCHESTRATION_DEPLOYMENT_URL"] = known_url
            response = litellm.completion(
                model=_SAP_MODEL,
                messages=_MESSAGES,
                api_key=os.environ["AICORE_SERVICE_KEY"],
                # no deployment_url optional param → should pick up env var
            )
        finally:
            if original is None:
                os.environ.pop("AICORE_ORCHESTRATION_DEPLOYMENT_URL", None)
            else:
                os.environ["AICORE_ORCHESTRATION_DEPLOYMENT_URL"] = original

        _assert_live_response(response)

    def test_url_via_discovery(self):
        """Step 3: no optional_param, no env var → auto-discovery runs."""
        import litellm

        # Ensure the env var is absent so we definitely fall through to step 3.
        original = os.environ.pop("AICORE_ORCHESTRATION_DEPLOYMENT_URL", None)
        try:
            response = litellm.completion(
                model=_SAP_MODEL,
                messages=_MESSAGES,
                api_key=os.environ["AICORE_SERVICE_KEY"],
            )
        finally:
            if original is not None:
                os.environ["AICORE_ORCHESTRATION_DEPLOYMENT_URL"] = original

        _assert_live_response(response)
