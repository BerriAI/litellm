import json

import pytest
from fastapi.testclient import TestClient

from unittest.mock import MagicMock, patch

from litellm import rerank
from litellm.llms.custom_httpx.http_handler import HTTPHandler
