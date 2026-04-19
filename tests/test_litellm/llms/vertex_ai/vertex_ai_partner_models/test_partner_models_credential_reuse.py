"""
Test that VertexAIPartnerModels reuses cached credentials from VertexBase
instead of creating a new VertexLLM instance on every request.
"""

import sys
from unittest.mock import MagicMock, patch

from litellm.llms.vertex_ai.vertex_ai_partner_models.main import (
    VertexAIPartnerModels,
)


class TestPartnerModelsCredentialReuse:
    def test_init_calls_super(self):
        """VertexAIPartnerModels.__init__ must call super().__init__() so that
        the VertexBase credential cache is initialized."""
        partner = VertexAIPartnerModels()
        # These attributes are set by VertexBase.__init__
        assert hasattr(partner, "_credentials_project_mapping")
        assert isinstance(partner._credentials_project_mapping, dict)
        assert hasattr(partner, "access_token")
        assert hasattr(partner, "project_id")

    def test_completion_uses_self_ensure_access_token(self):
        """completion() should call self._ensure_access_token, not create a
        throwaway VertexLLM instance. This ensures the credential cache on the
        singleton is reused across calls."""
        partner = VertexAIPartnerModels()

        # Mock vertexai import and the completion handler
        mock_vertexai = MagicMock()
        mock_vertexai.preview = MagicMock()
        mock_vertexai.preview.language_models = MagicMock()

        with (
            patch.dict(sys.modules, {"vertexai": mock_vertexai}),
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

            # _ensure_access_token should have been called on self
            mock_ensure.assert_called_once_with(
                credentials='{"type": "service_account"}',
                project_id="test-project",
                custom_llm_provider="vertex_ai",
            )

    def test_credential_cache_shared_across_calls(self):
        """Two successive completion() calls should hit load_auth only once,
        proving the credential cache on the VertexAIPartnerModels instance works."""
        partner = VertexAIPartnerModels()

        mock_creds = MagicMock()
        mock_creds.token = "my-token"
        mock_creds.expired = False
        mock_creds.project_id = "proj"
        mock_creds.quota_project_id = "proj"

        mock_vertexai = MagicMock()
        mock_vertexai.preview = MagicMock()
        mock_vertexai.preview.language_models = MagicMock()

        with (
            patch.dict(sys.modules, {"vertexai": mock_vertexai}),
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

            # load_auth should only be called once — second call uses cache
            assert mock_load.call_count == 1
