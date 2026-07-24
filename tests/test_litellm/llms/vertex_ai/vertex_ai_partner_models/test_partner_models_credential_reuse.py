"""
Test that VertexBase subclasses (PartnerModels, Gemma, ModelGarden) reuse
cached credentials instead of creating a new VertexLLM instance on every request.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.vertex_ai.vertex_ai_partner_models.main import (
    VertexAIPartnerModels,
)
from litellm.llms.vertex_ai.vertex_gemma_models.main import VertexAIGemmaModels
from litellm.llms.vertex_ai.vertex_model_garden.main import VertexAIModelGardenModels


def _mock_vertexai():
    """Return a MagicMock that satisfies the vertexai import guards."""
    m = MagicMock()
    m.preview = MagicMock()
    m.preview.language_models = MagicMock()
    return m


class TestVertexBaseSubclassInit:
    """All VertexBase subclasses must call super().__init__() so that
    the credential cache is initialized."""

    @pytest.mark.parametrize(
        "cls",
        [VertexAIPartnerModels, VertexAIGemmaModels, VertexAIModelGardenModels],
        ids=["PartnerModels", "Gemma", "ModelGarden"],
    )
    def test_init_calls_super(self, cls):
        instance = cls()
        assert hasattr(instance, "_credentials_project_mapping")
        assert isinstance(instance._credentials_project_mapping, dict)
        assert hasattr(instance, "access_token")
        assert hasattr(instance, "project_id")


class TestPartnerModelsCredentialReuse:
    def test_completion_uses_self_ensure_access_token(self):
        """completion() should call self._ensure_access_token, not create a
        throwaway VertexLLM instance."""
        partner = VertexAIPartnerModels()

        with (
            patch.dict(sys.modules, {"vertexai": _mock_vertexai()}),
            patch.object(
                partner,
                "_ensure_access_token",
                return_value=("cached-token", "test-project"),
            ) as mock_ensure,
            patch(
                "litellm.llms.vertex_ai.vertex_ai_partner_models.main.base_llm_http_handler"
            ) as mock_handler,
        ):
            mock_handler.completion.return_value = "response"

            partner.completion(
                model="meta/llama-3.1-405b-instruct-maas",
                messages=[{"role": "user", "content": "hello"}],
                model_response=MagicMock(),
                print_verbose=lambda *a, **kw: None,
                encoding=MagicMock(),
                logging_obj=MagicMock(),
                api_base=None,
                optional_params={},
                custom_prompt_dict={},
                headers=None,
                timeout=30.0,
                litellm_params={},
                vertex_project="test-project",
                vertex_location="us-central1",
                vertex_credentials='{"type": "service_account"}',
            )

            mock_ensure.assert_called_once_with(
                credentials='{"type": "service_account"}',
                project_id="test-project",
                custom_llm_provider="vertex_ai",
            )

    def test_credential_cache_shared_across_calls(self):
        """Two successive completion() calls should hit load_auth only once."""
        partner = VertexAIPartnerModels()

        mock_creds = MagicMock()
        mock_creds.token = "my-token"
        mock_creds.expired = False
        mock_creds.project_id = "proj"
        mock_creds.quota_project_id = "proj"

        with (
            patch.dict(sys.modules, {"vertexai": _mock_vertexai()}),
            patch.object(
                partner, "load_auth", return_value=(mock_creds, "proj")
            ) as mock_load,
            patch(
                "litellm.llms.vertex_ai.vertex_ai_partner_models.main.base_llm_http_handler"
            ) as mock_handler,
        ):
            mock_handler.completion.return_value = "resp"

            common_kwargs = dict(
                model="meta/llama-3.1-405b-instruct-maas",
                messages=[{"role": "user", "content": "hi"}],
                model_response=MagicMock(),
                print_verbose=lambda *a, **kw: None,
                encoding=MagicMock(),
                logging_obj=MagicMock(),
                api_base=None,
                optional_params={},
                custom_prompt_dict={},
                headers=None,
                timeout=30.0,
                litellm_params={},
                vertex_project="proj",
                vertex_location="us-central1",
                vertex_credentials='{"type": "service_account"}',
            )

            partner.completion(**common_kwargs)
            partner.completion(**common_kwargs)

            assert mock_load.call_count == 1


class TestGemmaModelsCredentialReuse:
    def test_completion_uses_self_ensure_access_token(self):
        """completion() should call self._ensure_access_token, not create a
        throwaway VertexLLM instance."""
        gemma = VertexAIGemmaModels()

        mock_gemma_config = MagicMock()
        mock_gemma_config.return_value.completion.return_value = "response"

        with (
            patch.dict(sys.modules, {"vertexai": _mock_vertexai()}),
            patch.object(
                gemma,
                "_ensure_access_token",
                return_value=("cached-token", "test-project"),
            ) as mock_ensure,
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.transformation.VertexGemmaConfig",
                mock_gemma_config,
            ),
        ):
            gemma.completion(
                model="gemma/gemma-3-12b-it-1234567890",
                messages=[{"role": "user", "content": "hello"}],
                model_response=MagicMock(),
                print_verbose=lambda *a, **kw: None,
                encoding=MagicMock(),
                logging_obj=MagicMock(),
                api_base="https://123.us-central1-1.prediction.vertexai.goog/v1/projects/proj/locations/us-central1/endpoints/456:predict",
                optional_params={},
                custom_prompt_dict={},
                headers=None,
                timeout=30.0,
                litellm_params={},
                vertex_project="test-project",
                vertex_location="us-central1",
                vertex_credentials='{"type": "service_account"}',
            )

            mock_ensure.assert_called_once_with(
                credentials='{"type": "service_account"}',
                project_id="test-project",
                custom_llm_provider="vertex_ai",
            )


class TestModelGardenCredentialReuse:
    def test_completion_uses_self_ensure_access_token(self):
        """completion() should call self._ensure_access_token, not create a
        throwaway VertexLLM instance."""
        garden = VertexAIModelGardenModels()

        mock_handler = MagicMock()
        mock_handler.return_value.completion.return_value = "response"

        with (
            patch.dict(sys.modules, {"vertexai": _mock_vertexai()}),
            patch.object(
                garden,
                "_ensure_access_token",
                return_value=("cached-token", "test-project"),
            ) as mock_ensure,
            patch(
                "litellm.llms.openai_like.chat.handler.OpenAILikeChatHandler",
                mock_handler,
            ),
        ):
            garden.completion(
                model="openai/5464397967697903616",
                messages=[{"role": "user", "content": "hello"}],
                model_response=MagicMock(),
                print_verbose=lambda *a, **kw: None,
                encoding=MagicMock(),
                logging_obj=MagicMock(),
                api_base=None,
                optional_params={},
                custom_prompt_dict={},
                headers=None,
                timeout=30.0,
                litellm_params={},
                vertex_project="test-project",
                vertex_location="us-central1",
                vertex_credentials='{"type": "service_account"}',
            )

            mock_ensure.assert_called_once_with(
                credentials='{"type": "service_account"}',
                project_id="test-project",
                custom_llm_provider="vertex_ai",
            )
