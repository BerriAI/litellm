import os
import json
import copy

from unittest.mock import patch

import litellm

model_name = "snowflake-arctic-embed"

embed_response = {
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [[0.1, 0.2, 0.3]],
            "index": 0,
        }
    ],
    "model": model_name,
    "usage": {"total_tokens": 4},
}


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_snowflake_jwt_account_id(mock_post):
    mock_post().json.return_value = copy.deepcopy(embed_response)

    response = litellm.embedding(
        f"snowflake/{model_name}",
        input=["document"],
        api_key="00000",
        account_id="AAAA-BBBB",
    )
    assert len(response.data) == 1
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]

    # check request
    post_kwargs = mock_post.call_args_list[-1][1]
    body = json.loads(post_kwargs["data"])
    assert body["model"] == model_name
    assert body["text"][0] == "document"

    # JWT key was used
    assert "00000" in post_kwargs["headers"]["Authorization"]
    # account id was used
    assert "AAAA-BBBB" in post_kwargs["url"]
    # is embedding
    assert post_kwargs["url"].endswith("cortex/inference:embed")


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_snowflake_pat_key_account_id(mock_post):
    mock_post().json.return_value = copy.deepcopy(embed_response)

    response = litellm.embedding(
        f"snowflake/{model_name}",
        input=["document"],
        api_key="pat/xxxxx",
        account_id="AAAA-BBBB",
    )
    assert len(response.data) == 1
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]

    # PAT key was used
    post_kwargs = mock_post.call_args_list[-1][1]
    assert "xxxxx" in post_kwargs["headers"]["Authorization"]
    assert (
        post_kwargs["headers"]["X-Snowflake-Authorization-Token-Type"]
        == "PROGRAMMATIC_ACCESS_TOKEN"
    )

    # account id was used
    assert "AAAA-BBBB" in post_kwargs["url"]


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_snowflake_env(mock_post):
    mock_post().json.return_value = copy.deepcopy(embed_response)

    os.environ["SNOWFLAKE_ACCOUNT_ID"] = "AAAA-BBBB"
    os.environ["SNOWFLAKE_JWT"] = "00000"

    response = litellm.embedding(f"snowflake/{model_name}", input=["document"])

    assert len(response.data) == 1
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]

    # JWT key was used
    post_kwargs = mock_post.call_args_list[-1][1]
    assert "00000" in post_kwargs["headers"]["Authorization"]
    # account id was used
    assert "AAAA-BBBB" in post_kwargs["url"]

    os.environ.pop("SNOWFLAKE_ACCOUNT_ID", None)
    os.environ.pop("SNOWFLAKE_JWT", None)
