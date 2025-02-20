import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm import rerank
from litellm.llms.custom_httpx.http_handler import HTTPHandler


def test_rerank_infer_region_from_model_arn():
    mock_response = MagicMock()
    args = {
        "model": "bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
        "query": "hello",
        "documents": ["hello", "world"],
    }

    def return_val():
        return {
            "results": [
                {"index": 0, "relevanceScore": 0.6716859340667725},
                {"index": 1, "relevanceScore": 0.0004994205664843321},
            ]
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    client = HTTPHandler()

    with patch.object(client, "post", return_value=mock_response) as mock_post:
        rerank(
            model=args["model"],
            query=args["query"],
            documents=args["documents"],
            client=client,
        )
        mock_post.assert_called_once()
        print(f"mock_post.call_args: {mock_post.call_args.kwargs}")
