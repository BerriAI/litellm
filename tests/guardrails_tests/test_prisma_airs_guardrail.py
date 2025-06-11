# test_prisma_airs_guardrail.py

import unittest
from unittest import mock
import sys
import os
import requests # Need to import requests to mock its exceptions
import json # To create valid JSON responses for mock

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.proxy.guardrails.guardrail_hooks.prisma_airs_guardrail import prisma_airs_guardrail, call_airs_api
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache


class TestPrismaAirsGuardrail(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_api_key = "mock-api-key"
        self.mock_api_base = "http://mock-airs-api.com"
        self.mock_profile_name = "default_profile"

        os.environ["PRISMA_AIRS_API_BASE"] = self.mock_api_base
        os.environ["PRISMA_AIRS_API_KEY"] = self.mock_api_key
        os.environ["PRISMA_AIRS_PROFILE_NAME"] = self.mock_profile_name

    def tearDown(self):
        del os.environ["PRISMA_AIRS_API_BASE"]
        del os.environ["PRISMA_AIRS_API_KEY"]
        del os.environ["PRISMA_AIRS_PROFILE_NAME"]

    @mock.patch('prisma_airs_guardrail.requests.post')
    @mock.patch('prisma_airs_guardrail.os.environ.get')
    def call_airs_api_success_no_block(self, mock_getenv, mock_post):
        """
        Tests the call_airs_api function when the API call is successful and no block is returned.
        """
        print("\n--- Running call_airs_api_success_no_block ---")

        mock_getenv.side_effect = {
            "PRISMA_AIRS_API_BASE": self.mock_api_base,
            "PRISMA_AIRS_API_KEY": self.mock_api_key,
            "PRISMA_AIRS_PROFILE_NAME": self.mock_profile_name
        }.get

        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"action": "allow"}
        mock_post.return_value = mock_response

        response = call_airs_api("Hello, how are you?")

        mock_post.assert_called_once()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"action": "allow"})

        expected_json_payload = {
            "metadata": {
                "ai_model": "Test AI model",
                "app_name": "Google AI",
                "app_user": "test-user-1"
            },
            "contents": [
                {
                    "prompt": "Hello, how are you?"
                }
            ],
            "ai_profile": {
                "profile_name": self.mock_profile_name
            }
        }
        mock_post.assert_called_once_with(
            self.mock_api_base,
            headers={
                "x-pan-token": self.mock_api_key,
                "Content-Type": "application/json"
            },
            json=expected_json_payload,
            timeout=5,
            verify=False
        )
        print("--- Finished call_airs_api_success_no_block ---")


    @mock.patch('prisma_airs_guardrail.call_airs_api')
    async def test_async_pre_call_hook_no_block(self, mock_call_airs_api):
        """
        Tests async_pre_call_hook when AIRS allows the request.
        """
        print("\n--- Running test_async_pre_call_hook_no_block ---")

        mock_airs_response = mock.Mock()
        mock_airs_response.status_code = 200
        mock_airs_response.json.return_value = {"action": "allow"}
        mock_call_airs_api.return_value = mock_airs_response

        guardrail = prisma_airs_guardrail()

        call_kwargs = {"messages": [{"role": "user", "content": "This is a safe prompt."}]}
        user_api_key_dict = UserAPIKeyAuth(api_key="mock_key")
        cache = DualCache()
        data = {"messages": [{"role": "user", "content": "This is a safe prompt."}]}
        call_type = "completion"
        result = await guardrail.async_pre_call_hook(user_api_key_dict, cache, data, call_type)
        mock_call_airs_api.assert_called_once_with("This is a safe prompt.")
        self.assertIsNone(result)
        print("--- Finished test_async_pre_call_hook_no_block ---")


    @mock.patch('prisma_airs_guardrail.call_airs_api')
    async def test_async_pre_call_hook_blocked_by_airs(self, mock_call_airs_api):
        """
        Tests async_pre_call_hook when AIRS blocks the request.
        """
        print("\n--- Running test_async_pre_call_hook_blocked_by_airs ---")

        mock_airs_response = mock.Mock()
        mock_airs_response.status_code = 200
        mock_airs_response.json.return_value = {"action": "block"}
        mock_call_airs_api.return_value = mock_airs_response

        guardrail = prisma_airs_guardrail()

        data = {"messages": [{"role": "user", "content": "This is a malicious prompt."}]}
        user_api_key_dict = UserAPIKeyAuth(api_key="mock_key")
        cache = DualCache()
        call_type = "completion"

        result = await guardrail.async_pre_call_hook(user_api_key_dict, cache, data, call_type)

        mock_call_airs_api.assert_called_once_with("This is a malicious prompt.")
        self.assertEqual(result, "Request blocked by security policy.")

        print("--- Finished test_async_pre_call_hook_blocked_by_airs ---")

    @mock.patch('prisma_airs_guardrail.call_airs_api')
    async def test_async_pre_call_hook_api_error(self, mock_call_airs_api):
        """
        Tests async_pre_call_hook when the AIRS API call fails (e.g., HTTP error from call_airs_api).
        """
        print("\n--- Running test_async_pre_call_hook_api_error ---")

        mock_airs_response = mock.Mock()
        mock_airs_response.status_code = 500
        mock_airs_response.json.return_value = {"error": "Internal Server Error"}
        mock_call_airs_api.return_value = mock_airs_response

        guardrail = prisma_airs_guardrail()
        data = {"messages": [{"role": "user", "content": "Any prompt."}]}
        user_api_key_dict = UserAPIKeyAuth(api_key="mock_key")
        cache = DualCache()
        call_type = "completion"

        result = await guardrail.async_pre_call_hook(user_api_key_dict, cache, data, call_type)

        mock_call_airs_api.assert_called_once_with("Any prompt.")
        self.assertEqual(result, "airs call failed (HTTP 500).")
        print("--- Finished test_async_pre_call_hook_api_error ---")

    @mock.patch('prisma_airs_guardrail.call_airs_api')
    async def test_async_pre_call_hook_exception(self, mock_call_airs_api):
        """
        Tests async_pre_call_hook when an unexpected exception occurs during the AIRS call.
        """
        print("\n--- Running test_async_pre_call_hook_exception ---")

        mock_call_airs_api.side_effect = requests.exceptions.ConnectionError("Mocked connection failed")

        guardrail = prisma_airs_guardrail()
        data = {"messages": [{"role": "user", "content": "Another prompt."}]}
        user_api_key_dict = UserAPIKeyAuth(api_key="mock_key")
        cache = DualCache()
        call_type = "completion"

        result = await guardrail.async_pre_call_hook(user_api_key_dict, cache, data, call_type)

        mock_call_airs_api.assert_called_once_with("Another prompt.")
        self.assertIn("Error calling AIRS Mocked connection failed", result)
        print("--- Finished test_async_pre_call_hook_exception ---")

    @mock.patch('prisma_airs_guardrail.call_airs_api')
    async def test_async_pre_call_hook_invalid_input(self, mock_call_airs_api):
        """
        Tests async_pre_call_hook with invalid input data (missing messages).
        """
        print("\n--- Running test_async_pre_call_hook_invalid_input ---")

        guardrail = prisma_airs_guardrail()
        data = {"some_other_field": "value"} # Missing "messages"
        user_api_key_dict = UserAPIKeyAuth(api_key="mock_key")
        cache = DualCache()
        call_type = "completion"

        result = await guardrail.async_pre_call_hook(user_api_key_dict, cache, data, call_type)

        mock_call_airs_api.assert_not_called()
        self.assertEqual(result, "Invalid input: 'messages' missing or improperly formatted.")
        print("--- Finished test_async_pre_call_hook_invalid_input ---")
