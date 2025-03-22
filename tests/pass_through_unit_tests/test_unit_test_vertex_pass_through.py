import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path


import httpx
import pytest
import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


from litellm.proxy.vertex_ai_endpoints.vertex_endpoints import (
    get_litellm_virtual_key,
    vertex_proxy_route,
    _get_vertex_env_vars,
    set_default_vertex_config,
    VertexPassThroughCredentials,
    default_vertex_config,
)
from litellm.proxy.vertex_ai_endpoints.vertex_passthrough_router import (
    VertexPassThroughRouter,
)
