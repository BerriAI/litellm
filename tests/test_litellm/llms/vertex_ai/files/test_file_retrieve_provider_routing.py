"""
Tests for Fix 1: file_retrieve Literal type was missing 'vertex_ai' and 'gemini',
causing a type mismatch when afile_retrieve delegated to the sync function.
"""

import pytest
from unittest.mock import MagicMock, patch

from litellm.files.main import file_retrieve


class TestFileRetrieveProviderRouting:
    """
    Verify that file_retrieve accepts 'vertex_ai' and 'gemini' providers and
    routes them through ProviderConfigManager / base_llm_http_handler.
    """

    def _make_mock_file_object(self):
        mock = MagicMock()
        mock.model_dump.return_value = {
            "id": "gs://my-bucket/file.jsonl",
            "object": "file",
            "bytes": 1024,
            "created_at": 0,
            "filename": "file.jsonl",
            "purpose": "batch",
            "status": "processed",
        }
        return mock

    def test_should_route_vertex_ai_through_provider_config(self):
        """
        Regression: file_retrieve Literal type was missing 'vertex_ai',
        so passing custom_llm_provider='vertex_ai' would fail type-checking
        and potentially cause a routing failure at runtime.
        """
        mock_file = self._make_mock_file_object()

        with patch(
            "litellm.files.main.base_llm_http_handler.retrieve_file",
            return_value=mock_file,
        ) as mock_retrieve:
            result = file_retrieve(
                file_id="gs://my-bucket/file.jsonl",
                custom_llm_provider="vertex_ai",
            )

        mock_retrieve.assert_called_once()
        assert result is not None

    def test_should_route_gemini_through_provider_config(self):
        """
        Regression: file_retrieve Literal type was also missing 'gemini'.
        """
        mock_file = self._make_mock_file_object()

        with patch(
            "litellm.files.main.base_llm_http_handler.retrieve_file",
            return_value=mock_file,
        ) as mock_retrieve:
            result = file_retrieve(
                file_id="some-gemini-file-id",
                custom_llm_provider="gemini",
            )

        mock_retrieve.assert_called_once()
        assert result is not None

    def test_should_pass_file_id_to_handler_for_vertex_ai(self):
        """Verify the file_id is forwarded correctly to the underlying handler."""
        mock_file = self._make_mock_file_object()
        expected_file_id = "gs://my-bucket/path/to/file.jsonl"

        with patch(
            "litellm.files.main.base_llm_http_handler.retrieve_file",
            return_value=mock_file,
        ) as mock_retrieve:
            file_retrieve(
                file_id=expected_file_id,
                custom_llm_provider="vertex_ai",
            )

        call_kwargs = mock_retrieve.call_args.kwargs
        assert call_kwargs.get("file_id") == expected_file_id

    def test_should_not_raise_bad_request_for_vertex_ai(self):
        """
        Before the fix, vertex_ai fell through to the else-branch which raised
        BadRequestError. Verify it no longer does.
        """
        import litellm

        mock_file = self._make_mock_file_object()

        with patch(
            "litellm.files.main.base_llm_http_handler.retrieve_file",
            return_value=mock_file,
        ):
            try:
                file_retrieve(
                    file_id="gs://my-bucket/file.jsonl",
                    custom_llm_provider="vertex_ai",
                )
            except litellm.exceptions.BadRequestError as e:
                pytest.fail(
                    f"file_retrieve raised BadRequestError for vertex_ai: {e}"
                )

    def test_should_not_raise_bad_request_for_gemini(self):
        """Same as above but for 'gemini'."""
        import litellm

        mock_file = self._make_mock_file_object()

        with patch(
            "litellm.files.main.base_llm_http_handler.retrieve_file",
            return_value=mock_file,
        ):
            try:
                file_retrieve(
                    file_id="some-file-id",
                    custom_llm_provider="gemini",
                )
            except litellm.exceptions.BadRequestError as e:
                pytest.fail(
                    f"file_retrieve raised BadRequestError for gemini: {e}"
                )
