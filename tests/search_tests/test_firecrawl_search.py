from unittest.mock import Mock, patch
import litellm


def test_firecrawl_search_request_body():
    """
    Test that validates the Firecrawl search request body is correctly formatted.
    """
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "data": {
            "web": [
                {
                    "title": "Test Title",
                    "url": "https://example.com",
                    "markdown": "Test content"
                }
            ]
        }
    }
    
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", return_value=mock_response) as mock_post:
        litellm.search(
            query="test query",
            search_provider="firecrawl",
            max_results=10,
            country="US"
        )
        
        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        request_body = call_kwargs.get("json")
        
        assert request_body is not None
        assert request_body["query"] == "test query"
        assert request_body["limit"] == 10
        assert request_body["country"] == "US"

