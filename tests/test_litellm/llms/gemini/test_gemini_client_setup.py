import pytest
import litellm
import os
from unittest.mock import patch, Mock
from litellm import completion


@pytest.fixture(autouse=True)
def mock_gemini_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-gemini-key-for-testing")


def test_gemini_completion():
    response = completion(
        model="gemini/gemini-2.0-flash-exp-image-generation",
        messages=[{"role": "user", "content": "Test message"}],
        mock_response="Test Message",
    )
    assert response.choices[0].message.content is not None


def test_gemini_completion_no_api_key():
    """Test Gemini completion fails gracefully when no API key is provided."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove all API keys
        for key in ["GOOGLE_API_KEY", "GEMINI_API_KEY"]:
            if key in os.environ:
                del os.environ[key]

        # Test without mock_response to ensure actual API key validation
        with pytest.raises(Exception) as exc_info:
            completion(
                model="gemini/gemini-1.5-flash",
                messages=[{"role": "user", "content": "Test message"}],
            )

        # Check that the exception message contains API key related text
        error_message = str(exc_info.value).lower()
        assert any(
            keyword in error_message
            for keyword in [
                "api key",
                "authentication",
                "unauthorized",
                "invalid",
                "missing",
                "credential",
            ]
        )


def test_gemini_completion_no_api_key_with_mock():
    """Alternative test that properly mocks the API key validation."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove all API keys
        for key in ["GOOGLE_API_KEY", "GEMINI_API_KEY"]:
            if key in os.environ:
                del os.environ[key]

        with patch("litellm.get_secret") as mock_get_secret:
            mock_get_secret.return_value = None

            with pytest.raises(Exception) as exc_info:
                completion(
                    model="gemini/gemini-1.5-flash",
                    messages=[{"role": "user", "content": "Test message"}],
                )

            error_message = str(exc_info.value).lower()
            assert any(
                keyword in error_message
                for keyword in [
                    "api key",
                    "authentication",
                    "unauthorized",
                    "invalid",
                    "missing",
                    "credential",
                ]
            )


@pytest.mark.parametrize("api_key_env", ["GOOGLE_API_KEY", "GEMINI_API_KEY"])
def test_gemini_completion_both_env_vars(monkeypatch, api_key_env):
    """Test Gemini completion works with both environment variable names."""
    # Clear all API keys first
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Set the specific API key being tested
    monkeypatch.setenv(api_key_env, f"fake-{api_key_env.lower()}-for-testing")

    response = completion(
        model="gemini/gemini-1.5-flash",
        messages=[{"role": "user", "content": f"Test with {api_key_env}"}],
        mock_response=f"Mocked response using {api_key_env}",
    )
    assert (
        response["choices"][0]["message"]["content"]
        == f"Mocked response using {api_key_env}"
    )
