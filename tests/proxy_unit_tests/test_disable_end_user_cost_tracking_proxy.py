import pytest
import asyncio
import os, sys
from unittest.mock import MagicMock, patch

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import litellm
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from datetime import datetime

@pytest.mark.asyncio
async def test_get_logging_payload_honors_disable_flag():
    """
    Test that get_logging_payload correctly suppresses end_user_id 
    when litellm.disable_end_user_cost_tracking is True.
    """
    # 1. Setup
    litellm.disable_end_user_cost_tracking = True
    
    kwargs = {
        "litellm_params": {
            "metadata": {
                "user_api_key_end_user_id": "test-user-123"
            }
        },
        "call_type": "completion",
        "standard_logging_object": {
            "metadata": {
                "user_api_key_end_user_id": "test-user-123"
            },
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "model_map_information": {}
        }
    }
    
    response_obj = {"id": "chatcmpl-123", "usage": {"total_tokens": 15}}
    start_time = datetime.now()
    end_time = datetime.now()
    
    # 2. Execute
    payload = get_logging_payload(kwargs, response_obj, start_time, end_time)
    
    # 3. Assert
    assert payload["end_user"] == ""
    
    # 4. Cleanup
    litellm.disable_end_user_cost_tracking = False

@pytest.mark.asyncio
async def test_get_logging_payload_tracks_when_not_disabled():
    """
    Test that get_logging_payload correctly includes end_user_id 
    when litellm.disable_end_user_cost_tracking is False.
    """
    # 1. Setup
    litellm.disable_end_user_cost_tracking = False
    
    kwargs = {
        "litellm_params": {
            "metadata": {
                "user_api_key_end_user_id": "test-user-456"
            }
        },
        "call_type": "completion",
        "standard_logging_object": {
            "metadata": {
                "user_api_key_end_user_id": "test-user-456"
            },
            "model_map_information": {}
        }
    }
    
    response_obj = {"id": "chatcmpl-456"}
    start_time = datetime.now()
    end_time = datetime.now()
    
    # 2. Execute
    payload = get_logging_payload(kwargs, response_obj, start_time, end_time)
    
    # 3. Assert
    assert payload["end_user"] == "test-user-456"
