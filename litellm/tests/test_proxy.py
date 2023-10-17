#### What this tests ####
#    This tests the OpenAI-proxy server

import sys, os
import traceback
sys.path.insert(0, os.path.abspath('../..'))  # Adds the parent directory to the system path
from dotenv import load_dotenv

load_dotenv()
import unittest
from unittest.mock import patch
from click.testing import CliRunner
import pytest
import litellm
from litellm.proxy.llm import litellm_completion

def test_azure_call():
    try: 
        data = {
            "model": "azure/chatgpt-v-2",
            "messages": [{"role": "user", "content": "Hey!"}]
        }
        result = litellm_completion(data=data, user_api_base=os.getenv("AZURE_API_BASE"), type="chat_completion", user_temperature=None, user_max_tokens=None, user_model=None, user_headers=None, user_debug=False)
        return result
    except Exception as e: 
        pytest.fail(f"An error occurred: {e}")

test_azure_call()