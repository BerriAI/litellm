import os
import sys
import pytest

# Adds the parent directory to the system path
sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.bytez.chat.transformation import BytezChatConfig, API_BASE, version

TEST_API_KEY = "MOCK_BYTEZ_API_KEY"
TEST_MODEL_NAME = "google/gemma-3-4b-it"
TEST_MODEL = f"bytez/{TEST_MODEL_NAME}"
TEST_MESSAGES = [{"role": "user", "content": "Hello"}]


def mock_validate_model_is_suported(
    bytez_chat_config: BytezChatConfig, is_supported: bool
) -> None:
    def validate_model_is_supported_mock(model_id: str, headers: dict):
        return is_supported

    bytez_chat_config.validate_model_is_supported = validate_model_is_supported_mock


class TestBytezChatConfig:
    def test_validate_environment(self):
        config = BytezChatConfig()

        mock_validate_model_is_suported(config, is_supported=True)

        headers = {}

        result = config.validate_environment(
            headers=headers,
            model=TEST_MODEL,
            messages=TEST_MESSAGES,  # type: ignore
            optional_params={},
            litellm_params={},
            api_key=TEST_API_KEY,
            api_base=API_BASE,
        )

        assert result["Authorization"] == f"Key {TEST_API_KEY}"
        assert result["content-type"] == "application/json"
        assert result["user-agent"] == f"litellm/{version}"

    def test_missing_api_key(self):
        with pytest.raises(Exception) as excinfo:
            config = BytezChatConfig()

            mock_validate_model_is_suported(config, is_supported=True)

            headers = {}

            config.validate_environment(
                headers=headers,
                model=TEST_MODEL,
                messages=TEST_MESSAGES,  # type: ignore
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=API_BASE,
            )

        assert "Missing api_key, make sure you pass in your api key" in str(
            excinfo.value
        )

    def test_invalid_model(self):

        with pytest.raises(Exception) as excinfo:
            config = BytezChatConfig()

            mock_validate_model_is_suported(config, is_supported=False)

            headers = {}

            config.validate_environment(
                headers=headers,
                model=TEST_MODEL,
                messages=TEST_MESSAGES,  # type: ignore
                optional_params={},
                litellm_params={},
                api_key=TEST_API_KEY,
                api_base=API_BASE,
            )

        assert f"Model: {TEST_MODEL} does not support chat" in str(excinfo.value)

    def test_bytez_completion_mock(self, respx_mock):
        import litellm

        mock_validate_model_is_suported(litellm.main.bytez_transformation, True)

        output_content = "Hello, how can I help you today?"

        output = {
            "role": "assistant",
            "content": [{"type": "text", "text": output_content}],
        }

        # Mock the HTTP request
        respx_mock.post(f"{API_BASE}/{TEST_MODEL_NAME}").respond(
            json={
                "error": None,
                "output": output,
            },
            status_code=200,
        )

        # Make the actual API call through LiteLLM
        response = litellm.completion(
            model=TEST_MODEL,
            messages=[
                {"role": "user", "content": "write code for saying hi from LiteLLM"}
            ],
            api_key=TEST_API_KEY,
            api_base=API_BASE,
        )

        assert response.choices[0].message.content == output_content  # type: ignore
