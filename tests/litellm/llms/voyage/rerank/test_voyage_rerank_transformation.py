import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import litellm
from unittest.mock import patch



### Rerank Tests
@pytest.mark.asyncio()
async def test_voyage_ai_rerank():
    mock_response = AsyncMock()

    def return_val():
        return {
            "id": "cmpl-mockid",
            "results": [{"index": 2, "relevance_score": 0.84375}],
            "usage": {"total_tokens": 150},
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    expected_payload = {
        "model": "rerank-model",
        "query": "What is the capital of the United States?",
        # Voyage API uses top_k instead of top_n
        "top_k": 1,
        "documents": [
            "Carson City is the capital city of the American state of Nevada.",
            "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
            "Washington, D.C. is the capital of the United States.",
            "Capital punishment has existed in the United States since before it was a country."
        ],
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.arerank(
            model="voyage/rerank-model",
            query="What is the capital of the United States?",
            documents=["Carson City is the capital city of the American state of Nevada.", "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.", "Washington, D.C. is the capital of the United States.", "Capital punishment has existed in the United States since before it was a country."],
            top_n=1, # This will be converted to top_k internally
            api_base="https://api.voyageai.ai"
        )

        print("async re rank response: ", response)

        # Assert
        mock_post.assert_called_once()
        print("call args", mock_post.call_args)
        args_to_api = mock_post.call_args.kwargs["data"]
        _url = mock_post.call_args.kwargs["url"]
        print("Arguments passed to API=", args_to_api)
        print("url = ", _url)
        assert _url == "https://api.voyageai.ai/v1/rerank"

        request_data = json.loads(args_to_api)
        print("request data to voyage ai", json.dumps(request_data, indent=4))
        assert request_data["query"] == expected_payload["query"]
        assert request_data["documents"] == expected_payload["documents"]
        assert request_data["top_k"] == expected_payload["top_k"]
        assert request_data["model"] == expected_payload["model"]

        assert response.id is not None
        assert response.results is not None
        assert response.meta["tokens"]["output_tokens"] == 150
            