from typing import Final
from unittest.mock import MagicMock

import pytest
import respx

import litellm
from litellm import completion


def test_wandb_logging():
    try:
        response = completion(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hi 👋 - i'm claude"}],
            max_tokens=10,
            temperature=0.2,
        )
        print(response)
    except litellm.Timeout as e:
        pass
    except Exception as e:
        print(e)
