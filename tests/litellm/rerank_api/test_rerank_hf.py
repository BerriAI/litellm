import os
import sys

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch
from unittest import TestCase
import litellm
from litellm import rerank
from litellm.llms.custom_httpx.http_handler import HTTPHandler, _get_httpx_client

def test_rerank_hf(monkeypatch):
    mock_response = MagicMock()
    
    monkeypatch.setenv("HUGGINGFACE_API_BASE", "http://localhost:8000")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf-key")
    litellm._turn_on_debug()

    args = {
        "model": "huggingface/BAAI/bge-large-en-v1.5",
        "query": "hello",
        "documents": ["hello", "world"],
    }

    def return_val():
        return  [
            {"index": 0, "score": 1.0, "text": "hello"},
            {"index": 1, "score": 0.0004994205664843321, "text": "world"},
        ]

    mock_response.json = return_val
    mock_response.status_code = 200

    client = _get_httpx_client()

    with patch.object(client, "post", return_value=mock_response) as mock_post:
        res = rerank(
            model=args["model"],
            query=args["query"],
            documents=args["documents"],
            client=client,
        )
        mock_post.assert_called_once()