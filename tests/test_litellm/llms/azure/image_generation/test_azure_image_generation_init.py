import json
import os
import sys
import traceback
from typing import Callable, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.azure.image_generation import (
    AzureDallE3ImageGenerationConfig,
    get_azure_image_generation_config,
)


@pytest.mark.parametrize(
    "received_model, expected_config",
    [
        ("dall-e-3", AzureDallE3ImageGenerationConfig),
        ("dalle-3", AzureDallE3ImageGenerationConfig),
        ("openai_dall_e_3", AzureDallE3ImageGenerationConfig),
    ],
)
def test_azure_image_generation_config(received_model, expected_config):
    assert isinstance(
        get_azure_image_generation_config(received_model), expected_config
    )
