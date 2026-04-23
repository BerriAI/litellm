import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm.llms.bedrock.image_generation.amazon_stability3_transformation import (
    AmazonStability3Config,
)


def test_stability_image_core_is_v3_model():
    model = "stability.stable-image-core-v1:1"
    assert AmazonStability3Config._is_stability_3_model(model)
