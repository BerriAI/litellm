
import httpx
import json
import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
import os
import uuid
import time
import base64

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from abc import ABC, abstractmethod
from litellm.integrations.custom_logger import CustomLogger
import json
from litellm.types.utils import StandardLoggingPayload

class BaseVectorStoreTest(ABC):
    """
    Abstract base test class that enforces a common test across all test classes.
    """
    @abstractmethod
    def get_base_request_args(self) -> dict:
        """Must return the base request args"""
        pass

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_search_vector_store(self, sync_mode):
        litellm._turn_on_debug()
        litellm.set_verbose = True
        base_request_args = self.get_base_request_args()
        try: 
            if sync_mode:
                response = litellm.vector_stores.search(
                    query="Basic ping", 
                    **base_request_args
                )
            else:
                response = await litellm.vector_stores.asearch(
                    query="Basic ping", 
                    **base_request_args
                )
        except litellm.InternalServerError: 
            pytest.skip("Skipping test due to litellm.InternalServerError")
        print("litellm response=", json.dumps(response, indent=4, default=str))
