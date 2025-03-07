import os
import time
import pytest
from litellm import completion
import litellm
from litellm.caching import InMemoryCache


# Set up environment variables for API keys and Langfuse configuration
@pytest.fixture(scope="module", autouse=True)
def set_environment_variables():
    os.environ["OPENAI_API_KEY"] = "sk-proj-XXXXX"  # Replace with actual test key
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-XXXXX"  # Replace with actual public key
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-XXXXX"  # Replace with actual secret key
    os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com"

    yield

    # Cleanup after the test runs
    os.environ.pop("OPENAI_API_KEY")
    os.environ.pop("LANGFUSE_PUBLIC_KEY")
    os.environ.pop("LANGFUSE_SECRET_KEY")
    os.environ.pop("LANGFUSE_HOST")


# Initialize Langfuse callback
@pytest.fixture(scope="module", autouse=True)
def initialize_langfuse():
    litellm.success_callback = ["langfuse"]


# Main test function for checking caching behavior
def test_caching_and_langfuse_logging():
    cached_tokens_used = 0

    for i in range(2):
        response = completion(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Here is the full text of a complex legal agreement " * 400,
                },
                {
                    "role": "user",
                    "content": "What are the key terms and conditions in this agreement?",
                }
            ],
            temperature=0.2,
            max_tokens=10,
        )

        time.sleep(2)  # Ensure different timestamps for each request

        # Assertions for response
        assert response is not None, f"Response {i + 1} is None."
        assert hasattr(response, "usage"), f"Response {i + 1} missing usage attribute."

        # Check usage details
        usage = response.usage
        assert usage.prompt_tokens > 0, "Prompt tokens should be greater than 0."
        assert usage.completion_tokens > 0, "Completion tokens should be greater than 0."

        # Check cached tokens in the second run
        if i == 1:
            prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = getattr(prompt_tokens_details, "cached_tokens", 0) if prompt_tokens_details else 0
            cached_tokens_used = cached_tokens

    # Assert that caching worked on the second run
    assert cached_tokens_used > 0, "Cached tokens should have been used in the second run."


# Verify environment variables are set correctly
def test_environment_variables():
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set."
    assert os.getenv("LANGFUSE_PUBLIC_KEY") is not None, "LANGFUSE_PUBLIC_KEY is not set."
    assert os.getenv("LANGFUSE_SECRET_KEY") is not None, "LANGFUSE_SECRET_KEY is not set."
    assert os.getenv("LANGFUSE_HOST") == "https://us.cloud.langfuse.com", "LANGFUSE_HOST is not set correctly."
