import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.llms.mistral.mistral_chat_transformation import MistralConfig


@pytest.mark.asyncio
async def test_mistral_chat_transformation():
    mistral_config = MistralConfig()
    result = mistral_config._transform_messages(
        **{
            "messages": [
                {
                    "content": [
                        {"type": "text", "text": "Here is a representation of text"},
                        {
                            "type": "image_url",
                            "image_url": "https://images.pexels.com/photos/13268478/pexels-photo-13268478.jpeg",
                        },
                    ],
                    "role": "user",
                }
            ],
            "model": "mistral-medium-latest",
            "is_async": True,
        }
    )
