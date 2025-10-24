import pytest
import litellm
from typing import List, Union
import json
import asyncio
from typing import Optional

from tests.search_tests.base_search_unit_tests import BaseSearchTest
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload


class TestExaAISearch(BaseSearchTest):
    """
    Tests for Exa AI Search functionality.
    """
    
    def get_search_provider(self) -> str:
        """
        Return search_provider for Exa AI Search.
        """
        return "exa_ai"


class TestCustomLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.standard_logging_object: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("in async log success event kwargs", json.dumps(kwargs, indent=4, default=str))
        self.standard_logging_object = kwargs.get("standard_logging_object")



async def test_exa_ai_search_with_custom_logger():
    litellm._turn_on_debug()
    litellm.set_verbose = True
    litellm.instant_log_for_testing = True
    test_custom_logger = TestCustomLogger()
    litellm.logging_callback_manager.add_litellm_callback(test_custom_logger)
    

    USER_QUERY = "latest AI developments"
    response = await litellm.asearch(
        query=USER_QUERY,
        search_provider="exa_ai",
        max_results=1,
    )

    print("EXA AI Search response", json.dumps(response, indent=4, default=str))
    assert response is not None

    await asyncio.sleep(3)
    print("standard logging object", json.dumps(test_custom_logger.standard_logging_object, indent=4, default=str))
    assert test_custom_logger.standard_logging_object is not None

    # cost 
    assert test_custom_logger.standard_logging_object["response_cost"] is not None

    # user query is logged
    assert test_custom_logger.standard_logging_object["messages"][0]["content"] == USER_QUERY

    # response is logged
    assert test_custom_logger.standard_logging_object["response"] is not None
