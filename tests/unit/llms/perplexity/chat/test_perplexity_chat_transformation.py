"""
Test file for Perplexity chat transformation functionality.

Tests the response transformation to extract citation tokens and search queries
from Perplexity API responses.
"""

import datetime
from unittest.mock import Mock

import httpx
import pytest

# Add the project root to Python path

from litellm import ModelResponse
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.perplexity.chat.transformation import PerplexityChatConfig
from litellm.types.utils import Usage


class TestPerplexityChatTransformation:
    """Test suite for Perplexity chat transformation functionality."""

    def test_enhance_usage_with_citation_tokens(self):
        """Test extraction of citation tokens from API response."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

        # Mock raw response with citations
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "citations": [
                "This is a citation with some text content",
                "Another citation with more text here",
                "Third citation with additional information",
            ],
        }

        # Enhance the usage with Perplexity fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)

        # Check that citation tokens were added
        assert hasattr(model_response.usage, "citation_tokens")
        citation_tokens = getattr(model_response.usage, "citation_tokens")

        # Should have extracted citation tokens (estimated based on character count)
        assert citation_tokens > 0
        assert isinstance(citation_tokens, int)

    def test_enhance_usage_with_search_queries_from_usage(self):
        """Test extraction of search queries from usage field in API response."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

        # Mock raw response with search queries in usage
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "num_search_queries": 3,
            },
        }

        # Enhance the usage with Perplexity fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)

        # Check that search queries were added to prompt_tokens_details
        assert hasattr(model_response.usage, "prompt_tokens_details")
        assert model_response.usage.prompt_tokens_details is not None
        assert hasattr(
            model_response.usage.prompt_tokens_details, "web_search_requests"
        )

        web_search_requests = (
            model_response.usage.prompt_tokens_details.web_search_requests
        )
        assert web_search_requests == 3

    def test_enhance_usage_with_search_queries_from_root(self):
        """Test extraction of search queries from root level in API response."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

        # Mock raw response with search queries at root level
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "num_search_queries": 2,
        }

        # Enhance the usage with Perplexity fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)

        # Check that search queries were added to prompt_tokens_details
        assert hasattr(model_response.usage, "prompt_tokens_details")
        assert model_response.usage.prompt_tokens_details is not None
        assert hasattr(
            model_response.usage.prompt_tokens_details, "web_search_requests"
        )

        web_search_requests = (
            model_response.usage.prompt_tokens_details.web_search_requests
        )
        assert web_search_requests == 2

    def test_enhance_usage_with_both_citations_and_search_queries(self):
        """Test extraction of both citation tokens and search queries."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

        # Mock raw response with both citations and search queries
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "num_search_queries": 2,
            },
            "citations": [
                "Citation one with some content",
                "Citation two with more information",
            ],
        }

        # Enhance the usage with Perplexity fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)

        # Check that both fields were added
        assert hasattr(model_response.usage, "citation_tokens")
        assert hasattr(model_response.usage, "prompt_tokens_details")
        assert model_response.usage.prompt_tokens_details is not None
        assert hasattr(
            model_response.usage.prompt_tokens_details, "web_search_requests"
        )

        citation_tokens = getattr(model_response.usage, "citation_tokens")
        web_search_requests = (
            model_response.usage.prompt_tokens_details.web_search_requests
        )

        assert citation_tokens > 0
        assert web_search_requests == 2

    def test_enhance_usage_with_empty_citations(self):
        """Test handling of empty citations array."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

        # Mock raw response with empty citations
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "citations": [],
        }

        # Enhance the usage with Perplexity fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)

        # Should not set citation_tokens for empty citations
        citation_tokens = getattr(model_response.usage, "citation_tokens", 0)
        assert citation_tokens == 0

    def test_enhance_usage_with_missing_fields(self):
        """Test handling when both citations and search queries are missing."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

        # Mock raw response without citations or search queries
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

        # Should not raise an error
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)

        # Should not have added custom fields
        citation_tokens = getattr(model_response.usage, "citation_tokens", 0)
        assert citation_tokens == 0

        # prompt_tokens_details might be None or have web_search_requests as 0
        if (
            hasattr(model_response.usage, "prompt_tokens_details")
            and model_response.usage.prompt_tokens_details
        ):
            web_search_requests = getattr(
                model_response.usage.prompt_tokens_details, "web_search_requests", 0
            )
            assert web_search_requests == 0

    def test_citation_token_estimation(self):
        """Test that citation token estimation is reasonable."""
        config = PerplexityChatConfig()

        # Test cases with known character counts
        test_cases = [
            # (citation_text, expected_min_tokens, expected_max_tokens)
            ("Short", 1, 2),
            ("This is a longer citation with multiple words", 10, 15),
            (
                "A very long citation with many words and characters that should result in more tokens",
                18,
                25,
            ),
        ]

        for citation_text, min_tokens, max_tokens in test_cases:
            model_response = ModelResponse()
            model_response.usage = Usage(
                prompt_tokens=100, completion_tokens=50, total_tokens=150
            )

            raw_response_dict = {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "citations": [citation_text],
            }

            config._enhance_usage_with_perplexity_fields(
                model_response, raw_response_dict
            )

            citation_tokens = getattr(model_response.usage, "citation_tokens")

            # Should be within reasonable range
            assert (
                min_tokens <= citation_tokens <= max_tokens
            ), f"Citation '{citation_text}' resulted in {citation_tokens} tokens, expected {min_tokens}-{max_tokens}"

    def test_multiple_citations_aggregation(self):
        """Test that multiple citations are aggregated correctly."""
        config = PerplexityChatConfig()

        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

        raw_response_dict = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "citations": [
                "First citation with some text",
                "Second citation with different content",
                "Third citation with more information",
            ],
        }

        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)

        citation_tokens = getattr(model_response.usage, "citation_tokens")

        # Should have aggregated all citations
        total_chars = sum(len(citation) for citation in raw_response_dict["citations"])
        expected_tokens = total_chars // 4  # Our estimation logic

        assert citation_tokens == expected_tokens

    def test_search_queries_priority_usage_over_root(self):
        """Test that search queries from usage field take priority over root level."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

        # Mock raw response with search queries in both locations
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "num_search_queries": 5,  # This should take priority
            },
            "num_search_queries": 3,  # This should be ignored
        }

        # Enhance the usage with Perplexity fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)

        # Check that usage field took priority
        assert hasattr(model_response.usage, "prompt_tokens_details")
        assert model_response.usage.prompt_tokens_details is not None
        web_search_requests = (
            model_response.usage.prompt_tokens_details.web_search_requests
        )

        assert web_search_requests == 5  # Should use the usage field value, not root

    def test_no_usage_object_handling(self):
        """Test handling when model_response has no usage object."""
        config = PerplexityChatConfig()

        # Create a ModelResponse without usage
        model_response = ModelResponse()

        # Mock raw response with Perplexity-specific fields
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "num_search_queries": 2,
            },
            "citations": ["Some citation"],
        }

        # Should not raise an error when usage is None
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)

        # Usage should be created with the Perplexity fields
        assert model_response.usage is not None
        assert hasattr(model_response.usage, "citation_tokens")
        assert hasattr(model_response.usage, "prompt_tokens_details")
        assert model_response.usage.prompt_tokens_details is not None
        assert hasattr(
            model_response.usage.prompt_tokens_details, "web_search_requests"
        )

        citation_tokens = getattr(model_response.usage, "citation_tokens")
        web_search_requests = (
            model_response.usage.prompt_tokens_details.web_search_requests
        )

        assert citation_tokens > 0
        assert web_search_requests == 2

    @pytest.mark.parametrize("search_query_location", ["usage", "root"])
    def test_search_queries_extraction_locations(self, search_query_location):
        """Test search queries extraction from different response locations."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with basic usage
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

        # Create response dict based on parameter
        if search_query_location == "usage":
            raw_response_dict = {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "num_search_queries": 4,
                }
            }
        else:  # root
            raw_response_dict = {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "num_search_queries": 4,
            }

        # Enhance the usage with Perplexity fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)

        # Should extract search queries from either location
        assert hasattr(model_response.usage, "prompt_tokens_details")
        assert model_response.usage.prompt_tokens_details is not None
        web_search_requests = (
            model_response.usage.prompt_tokens_details.web_search_requests
        )

        assert web_search_requests == 4

    # Tests for citation annotations functionality
    def test_add_citations_as_annotations_basic(self):
        """Test basic citation annotation creation."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with content
        from litellm.types.utils import Choices, Message

        message = Message(
            content="This response has citations[1][2] in the text.", role="assistant"
        )
        choice = Choices(finish_reason="stop", index=0, message=message)
        model_response = ModelResponse()
        model_response.choices = [choice]

        # Mock raw response with citations and search results
        raw_response_json = {
            "citations": ["https://example.com/page1", "https://example.com/page2"],
            "search_results": [
                {"title": "Example Page 1", "url": "https://example.com/page1"},
                {"title": "Example Page 2", "url": "https://example.com/page2"},
            ],
        }

        # Add citations as annotations
        config._add_citations_as_annotations(model_response, raw_response_json)

        # Check that annotations were created
        annotations = getattr(message, "annotations", None)
        assert annotations is not None
        assert len(annotations) == 2

        # Check first annotation
        annotation1 = annotations[0]
        assert annotation1["type"] == "url_citation"
        url_citation1 = annotation1["url_citation"]
        assert url_citation1["url"] == "https://example.com/page1"
        assert url_citation1["title"] == "Example Page 1"
        # Check that start_index and end_index are valid positions
        assert url_citation1["start_index"] >= 0
        assert url_citation1["end_index"] > url_citation1["start_index"]
        # Verify the positions correspond to [1] in the text
        assert (
            message.content[url_citation1["start_index"] : url_citation1["end_index"]]
            == "[1]"
        )

        # Check second annotation
        annotation2 = annotations[1]
        assert annotation2["type"] == "url_citation"
        url_citation2 = annotation2["url_citation"]
        assert url_citation2["url"] == "https://example.com/page2"
        assert url_citation2["title"] == "Example Page 2"
        # Check that start_index and end_index are valid positions
        assert url_citation2["start_index"] >= 0
        assert url_citation2["end_index"] > url_citation2["start_index"]
        # Verify the positions correspond to [2] in the text
        assert (
            message.content[url_citation2["start_index"] : url_citation2["end_index"]]
            == "[2]"
        )

        # Check backward compatibility
        assert hasattr(model_response, "citations")
        assert hasattr(model_response, "search_results")
        assert model_response.citations == raw_response_json["citations"]
        assert model_response.search_results == raw_response_json["search_results"]

    def test_add_citations_as_annotations_empty_citations(self):
        """Test handling of empty citations array."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with content
        from litellm.types.utils import Choices, Message

        message = Message(
            content="This response has citations[1][2] but no citations array.",
            role="assistant",
        )
        choice = Choices(finish_reason="stop", index=0, message=message)
        model_response = ModelResponse()
        model_response.choices = [choice]

        # Mock raw response with empty citations
        raw_response_json = {"citations": [], "search_results": []}

        # Add citations as annotations
        config._add_citations_as_annotations(model_response, raw_response_json)

        # Check that no annotations were created
        annotations = getattr(message, "annotations", None)
        assert annotations is None or len(annotations) == 0

    def test_add_citations_as_annotations_no_citation_patterns(self):
        """Test handling when text has no citation patterns."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with content without citation patterns
        from litellm.types.utils import Choices, Message

        message = Message(
            content="This response has no citation markers in the text.",
            role="assistant",
        )
        choice = Choices(finish_reason="stop", index=0, message=message)
        model_response = ModelResponse()
        model_response.choices = [choice]

        # Mock raw response with citations
        raw_response_json = {
            "citations": ["https://example.com/page1", "https://example.com/page2"],
            "search_results": [
                {"title": "Example Page 1", "url": "https://example.com/page1"},
                {"title": "Example Page 2", "url": "https://example.com/page2"},
            ],
        }

        # Add citations as annotations
        config._add_citations_as_annotations(model_response, raw_response_json)

        # Check that no annotations were created
        annotations = getattr(message, "annotations", None)
        assert annotations is None or len(annotations) == 0

    def test_add_citations_as_annotations_mismatched_numbers(self):
        """Test handling of citation numbers that don't match available citations."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with content
        from litellm.types.utils import Choices, Message

        message = Message(
            content="This response has citations[1][5] but only 3 citations available.",
            role="assistant",
        )
        choice = Choices(finish_reason="stop", index=0, message=message)
        model_response = ModelResponse()
        model_response.choices = [choice]

        # Mock raw response with only 3 citations
        raw_response_json = {
            "citations": [
                "https://example.com/page1",
                "https://example.com/page2",
                "https://example.com/page3",
            ],
            "search_results": [
                {"title": "Example Page 1", "url": "https://example.com/page1"},
                {"title": "Example Page 2", "url": "https://example.com/page2"},
                {"title": "Example Page 3", "url": "https://example.com/page3"},
            ],
        }

        # Add citations as annotations
        config._add_citations_as_annotations(model_response, raw_response_json)

        # Check that only one annotation was created (for [1])
        annotations = getattr(message, "annotations", None)
        assert annotations is not None
        assert len(annotations) == 1

        # Check the annotation
        annotation = annotations[0]
        assert annotation["type"] == "url_citation"
        url_citation = annotation["url_citation"]
        assert url_citation["url"] == "https://example.com/page1"
        assert url_citation["title"] == "Example Page 1"

    def test_add_citations_as_annotations_missing_titles(self):
        """Test handling when search results don't have titles."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with content
        from litellm.types.utils import Choices, Message

        message = Message(
            content="This response has citations[1][2] with search results but no titles.",
            role="assistant",
        )
        choice = Choices(finish_reason="stop", index=0, message=message)
        model_response = ModelResponse()
        model_response.choices = [choice]

        # Mock raw response with missing titles
        raw_response_json = {
            "citations": ["https://example.com/page1", "https://example.com/page2"],
            "search_results": [
                {"url": "https://example.com/page1"},  # No title
                {"title": "Example Page 2", "url": "https://example.com/page2"},
            ],
        }

        # Add citations as annotations
        config._add_citations_as_annotations(model_response, raw_response_json)

        # Check that annotations were created
        annotations = getattr(message, "annotations", None)
        assert annotations is not None
        assert len(annotations) == 2

        # Check first annotation (no title)
        annotation1 = annotations[0]
        url_citation1 = annotation1["url_citation"]
        assert url_citation1["title"] == ""  # Empty title for missing title

        # Check second annotation (has title)
        annotation2 = annotations[1]
        url_citation2 = annotation2["url_citation"]
        assert url_citation2["title"] == "Example Page 2"

    def test_add_citations_as_annotations_non_numeric_patterns(self):
        """Test handling of non-numeric citation patterns."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with content containing non-numeric patterns
        from litellm.types.utils import Choices, Message

        message = Message(
            content="This response has patterns: [a] [b] [1] [c] [2].", role="assistant"
        )
        choice = Choices(finish_reason="stop", index=0, message=message)
        model_response = ModelResponse()
        model_response.choices = [choice]

        # Mock raw response with citations
        raw_response_json = {
            "citations": ["https://example.com/page1", "https://example.com/page2"],
            "search_results": [
                {"title": "Example Page 1", "url": "https://example.com/page1"},
                {"title": "Example Page 2", "url": "https://example.com/page2"},
            ],
        }

        # Add citations as annotations
        config._add_citations_as_annotations(model_response, raw_response_json)

        # Check that only numeric patterns were processed
        annotations = getattr(message, "annotations", None)
        assert annotations is not None
        assert len(annotations) == 2  # Only [1] and [2] should be processed

        # Check that the annotations correspond to [1] and [2]
        urls = [ann["url_citation"]["url"] for ann in annotations]
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls

    def test_add_citations_as_annotations_empty_content(self):
        """Test handling of empty content."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with empty content
        from litellm.types.utils import Choices, Message

        message = Message(content="", role="assistant")
        choice = Choices(finish_reason="stop", index=0, message=message)
        model_response = ModelResponse()
        model_response.choices = [choice]

        # Mock raw response with citations
        raw_response_json = {
            "citations": ["https://example.com/page1"],
            "search_results": [
                {"title": "Example Page 1", "url": "https://example.com/page1"}
            ],
        }

        # Add citations as annotations
        config._add_citations_as_annotations(model_response, raw_response_json)

        # Check that no annotations were created
        annotations = getattr(message, "annotations", None)
        assert annotations is None or len(annotations) == 0

    def test_add_citations_as_annotations_no_choices(self):
        """Test handling when model_response has no choices."""
        config = PerplexityChatConfig()

        # Create a ModelResponse without choices
        model_response = ModelResponse()
        model_response.choices = []  # Explicitly set empty choices

        # Mock raw response with citations
        raw_response_json = {
            "citations": ["https://example.com/page1"],
            "search_results": [
                {"title": "Example Page 1", "url": "https://example.com/page1"}
            ],
        }

        # Should not raise an error
        config._add_citations_as_annotations(model_response, raw_response_json)

        # No annotations should be created since choices is empty
        assert len(model_response.choices) == 0

    def test_add_citations_as_annotations_no_message(self):
        """Test handling when choice has no message."""
        config = PerplexityChatConfig()

        # Create a ModelResponse with choice but no message
        from litellm.types.utils import Choices

        choice = Choices(finish_reason="stop", index=0, message=None)
        model_response = ModelResponse()
        model_response.choices = [choice]

        # Mock raw response with citations
        raw_response_json = {
            "citations": ["https://example.com/page1"],
            "search_results": [
                {"title": "Example Page 1", "url": "https://example.com/page1"}
            ],
        }

        # Should not raise an error
        config._add_citations_as_annotations(model_response, raw_response_json)

        # Check that no annotations were created (message content is None)
        assert choice.message.content is None
        # No annotations should be created since content is None
        assert (
            not hasattr(choice.message, "annotations")
            or choice.message.annotations is None
        )


_PERPLEXITY_COMPLETION = {
    "id": "cmpl-1",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "sonar",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Paris [1] is in France [2][7]."},
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}
_FIRST_CITATION = {
    "type": "url_citation",
    "url_citation": {"url": "https://a.example", "title": "", "start_index": 6, "end_index": 9},
}


def _transform_perplexity_response(body: object) -> ModelResponse:
    logging_obj = Logging(
        model="sonar",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        start_time=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="perplexity-transform-test",
        function_id="perplexity-transform-test",
    )
    return PerplexityChatConfig().transform_response(
        model="sonar",
        raw_response=httpx.Response(200, json=body),
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        request_data={"model": "sonar"},
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def test_transform_response_adds_citation_annotations_and_search_usage_from_the_raw_body():
    response = _transform_perplexity_response(
        {
            **_PERPLEXITY_COMPLETION,
            "citations": ["https://a.example", "https://b.example"],
            "search_results": [
                {"url": "https://a.example", "title": "A"},
                {"url": "https://c.example", "title": "C"},
            ],
            "usage": {**_PERPLEXITY_COMPLETION["usage"], "num_search_queries": 3},
        }
    )

    assert response.choices[0].message.annotations == [
        {
            "type": "url_citation",
            "url_citation": {"url": "https://a.example", "title": "A", "start_index": 6, "end_index": 9},
        },
        {
            "type": "url_citation",
            "url_citation": {"url": "https://b.example", "title": "", "start_index": 23, "end_index": 26},
        },
    ]
    assert repr(response.usage.prompt_tokens_details.web_search_requests) == "3"
    assert repr(response.usage.citation_tokens) == "8"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (10, 5, 15)
    assert response.id == "cmpl-1"


@pytest.mark.parametrize(
    "extra",
    [{}, {"citations": None}, {"citations": []}, {"search_results": []}, {"num_search_queries": 0}],
)
def test_transform_response_leaves_a_completion_without_perplexity_fields_unchanged(extra: dict):
    response = _transform_perplexity_response({**_PERPLEXITY_COMPLETION, **extra})

    assert response.choices[0].message.content == "Paris [1] is in France [2][7]."
    assert not hasattr(response.choices[0].message, "annotations")
    assert not hasattr(response.usage, "citation_tokens")
    assert response.usage.prompt_tokens_details is None
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (10, 5, 15)


def test_transform_response_reads_search_queries_from_the_root_of_the_body():
    response = _transform_perplexity_response({**_PERPLEXITY_COMPLETION, "num_search_queries": 2})

    assert repr(response.usage.prompt_tokens_details.web_search_requests) == "2"
    assert not hasattr(response.usage, "citation_tokens")


def test_transform_response_annotates_and_counts_citations_when_the_provider_sends_no_usage():
    body = {key: value for key, value in _PERPLEXITY_COMPLETION.items() if key != "usage"}

    response = _transform_perplexity_response({**body, "citations": ["https://a.example"]})

    assert response.choices[0].message.annotations == [_FIRST_CITATION]
    assert repr(response.usage.citation_tokens) == "4"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (0, 0, 0)


@pytest.mark.parametrize(
    ("extra", "citation_tokens_recorded"),
    [
        ({"usage": None}, False),
        ({"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "num_search_queries": "3"}}, True),
        ({"search_results": 3}, True),
        ({"search_results": None}, True),
    ],
)
def test_transform_response_still_returns_the_completion_when_a_perplexity_field_is_malformed(
    extra: dict, citation_tokens_recorded: bool
):
    response = _transform_perplexity_response({**_PERPLEXITY_COMPLETION, "citations": ["https://a.example"], **extra})

    assert response.choices[0].message.content == "Paris [1] is in France [2][7]."
    assert not hasattr(response.choices[0].message, "annotations")
    assert hasattr(response.usage, "citation_tokens") is citation_tokens_recorded


@pytest.mark.parametrize("body", [[], [_PERPLEXITY_COMPLETION], "choices", ""])
def test_transform_response_rejects_a_body_that_is_not_an_object_as_an_invalid_response_object(body: object):
    with pytest.raises(Exception, match="Invalid response object") as exc_info:
        _transform_perplexity_response(body)

    assert type(exc_info.value) is Exception


@pytest.mark.parametrize("body", [3, 1.5, True, ["error"], "error"])
def test_transform_response_raises_type_error_for_a_scalar_or_error_bearing_non_object_body(body: object):
    with pytest.raises(TypeError):
        _transform_perplexity_response(body)


def test_transform_response_keeps_body_values_of_every_json_type_as_the_provider_sent_them():
    citations = ["https://a.example", 5, None, 1.5, True, {"url": "https://d.example"}, ["https://e.example"]]
    search_results = [{"url": "https://a.example", "title": "A", "score": 0.5, "tags": None}, "loose", 7]

    response = _transform_perplexity_response(
        {
            **_PERPLEXITY_COMPLETION,
            "citations": citations,
            "search_results": search_results,
            "related_score": 1.5,
            "is_final": True,
            "trace": {"hops": [1, {"next": None}]},
        }
    )

    assert response.choices[0].message.annotations == [
        {
            "type": "url_citation",
            "url_citation": {"url": "https://a.example", "title": "A", "start_index": 6, "end_index": 9},
        }
    ]
    assert response.citations == citations
    assert [type(citation) for citation in response.citations] == [str, int, type(None), float, bool, dict, list]
    assert response.search_results == search_results
    assert [type(result) for result in response.search_results] == [dict, str, int]
    assert repr(response.usage.citation_tokens) == "18"
