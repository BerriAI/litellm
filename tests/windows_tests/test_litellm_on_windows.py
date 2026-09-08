import asyncio
import subprocess
import time
import traceback
import platform

import pytest



def test_using_litellm_on_windows():
    """Test that LiteLLM can be imported on Windows systems."""

    try:
        import litellm

        print(
            f"litellm imported successfully on Windows ({platform.system()} {platform.release()})"
        )

        response = litellm.completion(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": "This should never fail. Email ishaan@berri.ai if this test ever fails.",
                }
            ],
            mock_response="Hello, how are you?",
        )
        print(response)
    except Exception as e:
        pytest.fail(
            f"Error occurred on Windows: {e}. Installing litellm on Windows failed."
        )
