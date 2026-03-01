"""
Tests for Google/Gemini model variant suffix resolution.

When Gemini CLI sends a request for a model variant like
``gemini-3.1-pro-preview-customtools`` but the proxy only has
``gemini-3.1-pro-preview`` configured, the router should resolve the
variant to the base model's deployment and forward the variant name
to the upstream API.

Refs: https://github.com/BerriAI/litellm/issues/21697
"""
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

import pytest

import litellm
from litellm import Router
from litellm.router_utils.google_model_variant_utils import (
    GOOGLE_MODEL_VARIANT_SUFFIXES,
    resolve_google_model_variant,
)


@pytest.fixture
def gemini_router():
    """Router with a single gemini-3.1-pro-preview deployment."""
    return Router(
        model_list=[
            {
                "model_name": "gemini-3.1-pro-preview",
                "litellm_params": {
                    "model": "vertex_ai/gemini-3.1-pro-preview",
                    "vertex_project": "test-project",
                    "vertex_location": "global",
                    "api_key": "fake-key",
                },
            }
        ]
    )


class TestResolveGoogleModelVariant:
    """Unit tests for ``resolve_google_model_variant``."""

    def test_customtools_suffix_resolves_to_base(self, gemini_router):
        result = resolve_google_model_variant(
            model="gemini-3.1-pro-preview-customtools",
            model_names=gemini_router.model_names,
            get_model_from_alias=lambda m: gemini_router._get_model_from_alias(model=m),
        )
        assert result is not None
        base_model, suffix = result
        assert base_model == "gemini-3.1-pro-preview"
        assert suffix == "-customtools"

    def test_base_model_returns_none(self, gemini_router):
        result = resolve_google_model_variant(
            model="gemini-3.1-pro-preview",
            model_names=gemini_router.model_names,
            get_model_from_alias=lambda m: gemini_router._get_model_from_alias(model=m),
        )
        assert result is None

    def test_unknown_suffix_returns_none(self, gemini_router):
        result = resolve_google_model_variant(
            model="gemini-3.1-pro-preview-unknownsuffix",
            model_names=gemini_router.model_names,
            get_model_from_alias=lambda m: gemini_router._get_model_from_alias(model=m),
        )
        assert result is None

    def test_unrelated_model_returns_none(self, gemini_router):
        result = resolve_google_model_variant(
            model="gpt-4o",
            model_names=gemini_router.model_names,
            get_model_from_alias=lambda m: gemini_router._get_model_from_alias(model=m),
        )
        assert result is None

    def test_customtools_with_missing_base_returns_none(self):
        """Suffix matches but base model not configured."""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {
                        "model": "gpt-4o",
                        "api_key": "fake-key",
                    },
                }
            ]
        )
        result = resolve_google_model_variant(
            model="gemini-3.1-pro-preview-customtools",
            model_names=router.model_names,
            get_model_from_alias=lambda m: router._get_model_from_alias(model=m),
        )
        assert result is None

    def test_resolves_via_model_group_alias(self):
        """Variant resolves when base model is a model_group_alias."""
        router = Router(
            model_list=[
                {
                    "model_name": "my-gemini",
                    "litellm_params": {
                        "model": "vertex_ai/gemini-3.1-pro-preview",
                        "api_key": "fake-key",
                    },
                }
            ],
            model_group_alias={
                "gemini-3.1-pro-preview": "my-gemini",
            },
        )
        result = resolve_google_model_variant(
            model="gemini-3.1-pro-preview-customtools",
            model_names=router.model_names,
            get_model_from_alias=lambda m: router._get_model_from_alias(model=m),
        )
        assert result is not None
        base_model, suffix = result
        assert base_model == "my-gemini"
        assert suffix == "-customtools"


class TestCommonChecksVariantRouting:
    """Integration tests for variant resolution in ``_common_checks_available_deployment``."""

    def test_variant_finds_deployment(self, gemini_router):
        model, deployments = gemini_router._common_checks_available_deployment(
            model="gemini-3.1-pro-preview-customtools",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert model == "gemini-3.1-pro-preview-customtools"
        assert isinstance(deployments, list)
        assert len(deployments) > 0

    def test_variant_deployment_has_correct_litellm_model(self, gemini_router):
        _, deployments = gemini_router._common_checks_available_deployment(
            model="gemini-3.1-pro-preview-customtools",
            messages=[{"role": "user", "content": "hello"}],
        )
        dep = deployments[0]
        assert (
            dep["litellm_params"]["model"]
            == "vertex_ai/gemini-3.1-pro-preview-customtools"
        )

    def test_variant_does_not_mutate_original_deployment(self, gemini_router):
        original_model_list = gemini_router.model_list
        original_litellm_model = original_model_list[0]["litellm_params"]["model"]

        gemini_router._common_checks_available_deployment(
            model="gemini-3.1-pro-preview-customtools",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert (
            original_model_list[0]["litellm_params"]["model"]
            == original_litellm_model
        )

    def test_base_model_still_works(self, gemini_router):
        model, deployments = gemini_router._common_checks_available_deployment(
            model="gemini-3.1-pro-preview",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert model == "gemini-3.1-pro-preview"
        assert isinstance(deployments, list)
        assert len(deployments) > 0
        dep = deployments[0]
        assert dep["litellm_params"]["model"] == "vertex_ai/gemini-3.1-pro-preview"

    def test_unknown_model_still_raises(self, gemini_router):
        with pytest.raises(litellm.BadRequestError):
            gemini_router._common_checks_available_deployment(
                model="totally-unknown-model",
                messages=[{"role": "user", "content": "hello"}],
            )

    def test_multiple_deployments_all_get_variant_suffix(self):
        """When base model has multiple deployments, all get the variant suffix."""
        router = Router(
            model_list=[
                {
                    "model_name": "gemini-3.1-pro-preview",
                    "litellm_params": {
                        "model": "vertex_ai/gemini-3.1-pro-preview",
                        "vertex_project": "project-a",
                        "vertex_location": "us-central1",
                        "api_key": "fake-key-a",
                    },
                },
                {
                    "model_name": "gemini-3.1-pro-preview",
                    "litellm_params": {
                        "model": "vertex_ai/gemini-3.1-pro-preview",
                        "vertex_project": "project-b",
                        "vertex_location": "europe-west4",
                        "api_key": "fake-key-b",
                    },
                },
            ]
        )
        model, deployments = router._common_checks_available_deployment(
            model="gemini-3.1-pro-preview-customtools",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert model == "gemini-3.1-pro-preview-customtools"
        assert len(deployments) == 2
        for dep in deployments:
            assert (
                dep["litellm_params"]["model"]
                == "vertex_ai/gemini-3.1-pro-preview-customtools"
            )

    def test_variant_preserves_vertex_credentials(self, gemini_router):
        """Variant deployment retains vertex_project and vertex_location from base."""
        _, deployments = gemini_router._common_checks_available_deployment(
            model="gemini-3.1-pro-preview-customtools",
            messages=[{"role": "user", "content": "hello"}],
        )
        dep = deployments[0]
        assert dep["litellm_params"]["vertex_project"] == "test-project"
        assert dep["litellm_params"]["vertex_location"] == "global"

    def test_gemini_provider_prefix_variant(self):
        """Works with gemini/ provider prefix too."""
        router = Router(
            model_list=[
                {
                    "model_name": "gemini-3.1-pro-preview",
                    "litellm_params": {
                        "model": "gemini/gemini-3.1-pro-preview",
                        "api_key": "fake-key",
                    },
                }
            ]
        )
        _, deployments = router._common_checks_available_deployment(
            model="gemini-3.1-pro-preview-customtools",
            messages=[{"role": "user", "content": "hello"}],
        )
        dep = deployments[0]
        assert (
            dep["litellm_params"]["model"]
            == "gemini/gemini-3.1-pro-preview-customtools"
        )

    def test_variant_resolves_even_with_default_fallbacks(self):
        """Variant resolution must run before and independently of default fallbacks."""
        router = Router(
            model_list=[
                {
                    "model_name": "gemini-3.1-pro-preview",
                    "litellm_params": {
                        "model": "vertex_ai/gemini-3.1-pro-preview",
                        "api_key": "fake-key",
                    },
                },
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {
                        "model": "gpt-4o",
                        "api_key": "fake-key",
                    },
                },
            ],
            default_fallbacks=["gpt-4o"],
        )
        model, deployments = router._common_checks_available_deployment(
            model="gemini-3.1-pro-preview-customtools",
            messages=[{"role": "user", "content": "hello"}],
        )
        # Should resolve to the variant, NOT fall back to gpt-4o
        assert model == "gemini-3.1-pro-preview-customtools"
        assert len(deployments) > 0
        dep = deployments[0]
        assert (
            dep["litellm_params"]["model"]
            == "vertex_ai/gemini-3.1-pro-preview-customtools"
        )


class TestGoogleModelVariantSuffixesConstant:
    """Ensure the constant is well-formed."""

    def test_customtools_in_suffixes(self):
        assert "-customtools" in GOOGLE_MODEL_VARIANT_SUFFIXES

    def test_all_suffixes_start_with_dash(self):
        for suffix in GOOGLE_MODEL_VARIANT_SUFFIXES:
            assert suffix.startswith("-"), f"Suffix {suffix!r} must start with '-'"
