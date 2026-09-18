import pytest
import litellm
from typing import List, Union
from unittest.mock import Mock

from litellm.llms.exa_ai.search.transformation import ExaAISearchConfig
from tests.search_tests.base_search_unit_tests import BaseSearchTest


class TestExaAISearch(BaseSearchTest):
    """
    Tests for Exa AI Search functionality.
    """

    def get_search_provider(self) -> str:
        """
        Return search_provider for Exa AI Search.
        """
        return "exa_ai"


class TestExaAISearchTransformation:
    def test_should_include_output_when_exa_returns_output(self):
        config = ExaAISearchConfig()
        raw_response = Mock()
        raw_response.json.return_value = {
            "output": {
                "content": "Nvidia announced the Vera Rubin platform.",
                "grounding": [
                    {
                        "field": "content",
                        "citations": [
                            {
                                "url": "https://nvidianews.nvidia.com/news/test",
                                "title": "NVIDIA Newsroom",
                            }
                        ],
                        "confidence": "high",
                    }
                ],
            },
            "results": [
                {
                    "title": "NVIDIA Newsroom",
                    "url": "https://nvidianews.nvidia.com/news/test",
                    "text": "NVIDIA announced Vera Rubin.",
                    "publishedDate": "2026-06-22T00:00:00.000Z",
                }
            ],
        }

        response = config.transform_search_response(
            raw_response=raw_response,
            logging_obj=None,
        )

        assert response.output == raw_response.json.return_value["output"]
        assert (
            response.model_dump()["output"]
            == raw_response.json.return_value["output"]
        )
        assert response.results[0].title == "NVIDIA Newsroom"

    def test_should_not_require_output_when_exa_omits_output(self):
        config = ExaAISearchConfig()
        raw_response = Mock()
        raw_response.json.return_value = {
            "results": [
                {
                    "title": "NVIDIA Newsroom",
                    "url": "https://nvidianews.nvidia.com/news/test",
                    "text": "NVIDIA announced Vera Rubin.",
                }
            ],
        }

        response = config.transform_search_response(
            raw_response=raw_response,
            logging_obj=None,
        )

        assert "output" not in response.model_dump()
        assert response.object == "search"
        assert response.results[0].title == "NVIDIA Newsroom"
