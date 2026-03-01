"""
Tests for Azure model identification in get_router_model_info.

Ensures Azure deployments without explicit 'base_model' no longer produce
ERROR-level log spam. The deployment name is used as a fallback, and only
a DEBUG-level message is emitted.

Fixes https://github.com/BerriAI/litellm/issues/13718
"""

import logging

import pytest

from litellm import Router


def _make_deployment_dict(model_name, azure_model, base_model=None):
    """Build a raw deployment dict for get_router_model_info."""
    dep = {
        "model_name": model_name,
        "litellm_params": {
            "model": azure_model,
            "api_key": "fake-key",
            "api_base": "https://test.openai.azure.com/",
            "api_version": "2025-03-01-preview",
        },
        "model_info": {"id": "test-id-" + model_name},
    }
    if base_model is not None:
        dep["model_info"]["base_model"] = base_model
    return dep


@pytest.fixture
def _azure_router():
    return Router(
        model_list=[
            _make_deployment_dict("gpt-5", "azure/gpt-5"),
        ]
    )


@pytest.fixture
def _azure_router_with_base_model():
    return Router(
        model_list=[
            _make_deployment_dict(
                "my-deploy", "azure/my-custom-deploy-name", base_model="azure/gpt-5"
            ),
        ]
    )


class TestAzureModelIdentificationNoBaseModel:
    """Azure deployments without base_model should NOT produce ERROR logs."""

    def test_no_error_log_for_standard_azure_model(self, _azure_router, caplog):
        """Standard Azure model names (gpt-5, gpt-4o) resolve without errors."""
        dep = _make_deployment_dict("gpt-5", "azure/gpt-5")

        with caplog.at_level(logging.DEBUG):
            _azure_router.get_router_model_info(
                deployment=dep,
                received_model_name="gpt-5",
            )

        error_msgs = [r for r in caplog.records if r.levelno >= logging.ERROR]
        azure_errors = [
            r for r in error_msgs if "Could not identify azure model" in r.getMessage()
        ]
        assert len(azure_errors) == 0, (
            "Standard Azure models should NOT produce ERROR logs"
        )

    def test_debug_log_emitted_for_azure_without_base_model(
        self, _azure_router, caplog
    ):
        """A DEBUG log should be emitted when base_model is not set."""
        dep = _make_deployment_dict("gpt-5", "azure/gpt-5")

        with caplog.at_level(logging.DEBUG, logger="LiteLLM Router"):
            _azure_router.get_router_model_info(
                deployment=dep,
                received_model_name="gpt-5",
            )

        debug_msgs = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG and "base_model" in r.getMessage()
        ]
        assert len(debug_msgs) >= 1, (
            "Should emit DEBUG log about missing base_model"
        )

    def test_model_info_returned_for_standard_azure(self, _azure_router):
        """get_router_model_info returns valid info for standard Azure models."""
        dep = _make_deployment_dict("gpt-5", "azure/gpt-5")
        info = _azure_router.get_router_model_info(
            deployment=dep,
            received_model_name="gpt-5",
        )
        assert info is not None
        assert info.get("max_tokens") is not None or info.get("max_output_tokens") is not None


class TestAzureModelIdentificationWithBaseModel:
    """Azure deployments WITH base_model should work silently."""

    def test_no_warning_with_base_model(self, _azure_router_with_base_model, caplog):
        """No base_model warnings when base_model is explicitly set."""
        dep = _make_deployment_dict(
            "my-deploy", "azure/my-custom-deploy-name", base_model="azure/gpt-5"
        )

        with caplog.at_level(logging.DEBUG):
            _azure_router_with_base_model.get_router_model_info(
                deployment=dep,
                received_model_name="my-deploy",
            )

        azure_msgs = [
            r
            for r in caplog.records
            if "base_model" in r.getMessage()
            and "azure" in r.getMessage().lower()
        ]
        assert len(azure_msgs) == 0, (
            "No azure base_model warnings when base_model is set"
        )


class TestAzureModelIdentificationMultipleDeployments:
    """Multiple Azure deployments should not produce error floods."""

    def test_multiple_deployments_no_error_flood(self, caplog):
        """Many Azure deployments without base_model produce zero ERROR logs."""
        models = ["gpt-5", "gpt-5-mini", "gpt-4o", "gpt-4o-mini"]
        model_list = [_make_deployment_dict(m, f"azure/{m}") for m in models]
        router = Router(model_list=model_list)

        with caplog.at_level(logging.DEBUG):
            for m in models:
                dep = _make_deployment_dict(m, f"azure/{m}")
                router.get_router_model_info(
                    deployment=dep,
                    received_model_name=m,
                )

        error_msgs = [r for r in caplog.records if r.levelno >= logging.ERROR]
        azure_errors = [
            r for r in error_msgs if "azure" in r.getMessage().lower()
        ]
        assert len(azure_errors) == 0, (
            f"Expected 0 ERROR logs for standard Azure models, got {len(azure_errors)}"
        )


class TestAzureModelDocLink:
    """Verify the documentation link in the debug message is correct."""

    def test_doc_link_points_to_custom_pricing(self, _azure_router, caplog):
        """The debug message should reference the correct docs page."""
        dep = _make_deployment_dict("gpt-5", "azure/gpt-5")

        with caplog.at_level(logging.DEBUG, logger="LiteLLM Router"):
            _azure_router.get_router_model_info(
                deployment=dep,
                received_model_name="gpt-5",
            )

        debug_msgs = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG and "base_model" in r.getMessage()
        ]
        if debug_msgs:
            assert "custom_pricing" in debug_msgs[0].getMessage(), (
                "Doc link should point to custom_pricing page"
            )
