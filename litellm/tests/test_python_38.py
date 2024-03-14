import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


def test_using_litellm():
    try:
        import litellm

        print("litellm imported successfully")
    except Exception as e:
        pytest.fail(
            f"Error occurred: {e}. Installing litellm on python3.8 failed please retry"
        )
