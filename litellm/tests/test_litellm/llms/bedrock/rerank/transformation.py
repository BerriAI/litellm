import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm import rerank
from litellm.llms.custom_httpx.http_handler import HTTPHandler
