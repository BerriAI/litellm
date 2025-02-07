import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse
from base_llm_unit_tests import BaseLLMChatTest
import asyncio


def test_openai_prediction_param():
    litellm.set_verbose = True
    code = """
    /// <summary>
    /// Represents a user with a first name, last name, and username.
    /// </summary>
    public class User
    {
        /// <summary>
        /// Gets or sets the user's first name.
        /// </summary>
        public string FirstName { get; set; }

        /// <summary>
        /// Gets or sets the user's last name.
        /// </summary>
        public string LastName { get; set; }

        /// <summary>
        /// Gets or sets the user's username.
        /// </summary>
        public string Username { get; set; }
    }
    """

    completion = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
            },
            {"role": "user", "content": code},
        ],
        prediction={"type": "content", "content": code},
    )

    print(completion)

    assert (
        completion.usage.completion_tokens_details.accepted_prediction_tokens > 0
        or completion.usage.completion_tokens_details.rejected_prediction_tokens > 0
    )


@pytest.mark.asyncio
async def test_openai_prediction_param_mock():
    """
    Tests that prediction parameter is correctly passed to the API
    """
    litellm.set_verbose = True

    code = """
    /// <summary>
    /// Represents a user with a first name, last name, and username.
    /// </summary>
    public class User
    {
        /// <summary>
        /// Gets or sets the user's first name.
        /// </summary>
        public string FirstName { get; set; }

        /// <summary>
        /// Gets or sets the user's last name.
        /// </summary>
        public string LastName { get; set; }

        /// <summary>
        /// Gets or sets the user's username.
        /// </summary>
        public string Username { get; set; }
    }
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
                    },
                    {"role": "user", "content": code},
                ],
                prediction={"type": "content", "content": code},
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        # Verify the request contains the prediction parameter
        assert "prediction" in request_body
        # verify prediction is correctly sent to the API
        assert request_body["prediction"] == {"type": "content", "content": code}


@pytest.mark.asyncio
async def test_openai_prediction_param_with_caching():
    """
    Tests using `prediction` parameter with caching
    """
    from litellm.caching.caching import LiteLLMCacheType
    import logging
    from litellm._logging import verbose_logger

    verbose_logger.setLevel(logging.DEBUG)
    import time

    litellm.set_verbose = True
    litellm.cache = litellm.Cache(type=LiteLLMCacheType.LOCAL)
    code = """
    /// <summary>
    /// Represents a user with a first name, last name, and username.
    /// </summary>
    public class User
    {
        /// <summary>
        /// Gets or sets the user's first name.
        /// </summary>
        public string FirstName { get; set; }

        /// <summary>
        /// Gets or sets the user's last name.
        /// </summary>
        public string LastName { get; set; }

        /// <summary>
        /// Gets or sets the user's username.
        /// </summary>
        public string Username { get; set; }
    }
    """

    completion_response_1 = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
            },
            {"role": "user", "content": code},
        ],
        prediction={"type": "content", "content": code},
    )

    time.sleep(0.5)

    # cache hit
    completion_response_2 = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
            },
            {"role": "user", "content": code},
        ],
        prediction={"type": "content", "content": code},
    )

    assert completion_response_1.id == completion_response_2.id

    completion_response_3 = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What is the first name of the user?"},
        ],
        prediction={"type": "content", "content": code + "FirstName"},
    )

    assert completion_response_3.id != completion_response_1.id


@pytest.mark.asyncio()
async def test_vision_with_custom_model():
    """
    Tests that an OpenAI compatible endpoint when sent an image will receive the image in the request

    """
    import base64
    import requests
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")

    litellm.set_verbose = True
    api_base = "https://my-custom.api.openai.com"

    # Fetch and encode a test image
    url = "https://dummyimage.com/100/100/fff&text=Test+image"
    response = requests.get(url)
    file_data = response.content
    encoded_file = base64.b64encode(file_data).decode("utf-8")
    base64_image = f"data:image/png;base64,{encoded_file}"

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            response = await litellm.acompletion(
                model="openai/my-custom-model",
                max_tokens=10,
                api_base=api_base,  # use the mock api
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_image},
                            },
                        ],
                    }
                ],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", request_body)

        assert request_body["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkBAMAAACCzIhnAAAAG1BMVEURAAD///+ln5/h39/Dv79qX18uHx+If39MPz9oMSdmAAAACXBIWXMAAA7EAAAOxAGVKw4bAAABB0lEQVRYhe2SzWrEIBCAh2A0jxEs4j6GLDS9hqWmV5Flt0cJS+lRwv742DXpEjY1kOZW6HwHFZnPmVEBEARBEARB/jd0KYA/bcUYbPrRLh6amXHJ/K+ypMoyUaGthILzw0l+xI0jsO7ZcmCcm4ILd+QuVYgpHOmDmz6jBeJImdcUCmeBqQpuqRIbVmQsLCrAalrGpfoEqEogqbLTWuXCPCo+Ki1XGqgQ+jVVuhB8bOaHkvmYuzm/b0KYLWwoK58oFqi6XfxQ4Uz7d6WeKpna6ytUs5e8betMcqAv5YPC5EZB2Lm9FIn0/VP6R58+/GEY1X1egVoZ/3bt/EqF6malgSAIgiDIH+QL41409QMY0LMAAAAASUVORK5CYII="
                        },
                    },
                ],
            },
        ]
        assert request_body["model"] == "my-custom-model"
        assert request_body["max_tokens"] == 10


class TestOpenAIChatCompletion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {"model": "gpt-4o-mini"}

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_multilingual_requests(self):
        """
        Tests that the provider can handle multilingual requests and invalid utf-8 sequences

        Context: https://github.com/openai/openai-python/issues/1921
        """
        base_completion_call_args = self.get_base_completion_call_args()
        response = self.completion_function(
            **base_completion_call_args,
            messages=[{"role": "user", "content": "你好世界！\ud83e, ö"}],
        )
        assert response is not None


def test_completion_bad_org():
    import litellm

    litellm.set_verbose = True
    _old_org = os.environ.get("OPENAI_ORGANIZATION", None)
    os.environ["OPENAI_ORGANIZATION"] = "bad-org"
    messages = [{"role": "user", "content": "hi"}]

    with pytest.raises(Exception) as exc_info:
        comp = litellm.completion(
            model="gpt-4o-mini", messages=messages, organization="bad-org"
        )

    print(exc_info.value)
    assert "header should match organization for API key" in str(exc_info.value)

    if _old_org is not None:
        os.environ["OPENAI_ORGANIZATION"] = _old_org
    else:
        del os.environ["OPENAI_ORGANIZATION"]


@patch("litellm.main.openai_chat_completions._get_openai_client")
def test_openai_max_retries_0(mock_get_openai_client):
    import litellm

    litellm.set_verbose = True
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        max_retries=0,
    )

    mock_get_openai_client.assert_called_once()
    assert mock_get_openai_client.call_args.kwargs["max_retries"] == 0
