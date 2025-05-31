import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from unittest.mock import MagicMock, patch

import litellm
from litellm.passthrough.main import llm_passthrough_route


def test_llm_passthrough_route():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(
        client.client,
        "send",
        return_value=MagicMock(status_code=200, json={"message": "Hello, world!"}),
    ) as mock_post:
        response = llm_passthrough_route(
            model="gpt-3.5-turbo",
            endpoint="v1/chat/completions",
            method="POST",
            request_url="http://localhost:8000/v1/chat/completions",
            api_base="http://localhost:8090",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello, world!"}],
            },
            client=client,
        )

        mock_post.call_args.kwargs[
            "request"
        ].url == "http://localhost:8090/v1/chat/completions"

        assert response.status_code == 200
        assert response.json == {"message": "Hello, world!"}
