import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
import litellm


def test_basic_openai_responses_api():
    response = litellm.responses(
        model="gpt-4o", input="Tell me a three sentence bedtime story about a unicorn."
    )

    # validate_responses_api_response()
