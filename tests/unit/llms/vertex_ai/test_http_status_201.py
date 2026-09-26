import json
import unittest
from unittest.mock import MagicMock, patch

from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexAIError,
    make_call,
    make_sync_call,
)


class TestVertexAIHTTPStatus201(unittest.TestCase):
    def setUp(self):
        # Setup mock messages
        self.messages = [{"role": "user", "content": "Hello, how are you?"}]

        # Setup mock data
        self.mock_data = json.dumps({"messages": self.messages})

        # Setup mock headers
        self.mock_headers = {"Content-Type": "application/json"}

        # Setup mock model
        self.mock_model = "gemini-pro"

        # Setup mock logging object
        self.mock_logging_obj = MagicMock()
        self.mock_logging_obj.post_call = MagicMock()

    @patch(
        "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.get_async_httpx_client"
    )
    async def test_async_http_status_201(self, mock_get_client):
        """Test that async make_call handles HTTP 201 status code correctly"""
        # Create a mock response with status code 201
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.aiter_lines = MagicMock()
        mock_response.aiter_lines.return_value = ["test response"]

        # Setup mock client
        mock_client = MagicMock()
        mock_client.post = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Call the make_call function
        result = await make_call(
            client=None,
            api_base="https://mock-vertex-ai-api.com",
            headers=self.mock_headers,
            data=self.mock_data,
            model=self.mock_model,
            messages=self.messages,
            logging_obj=self.mock_logging_obj,
        )

        # Assert that the post method was called
        mock_client.post.assert_called_once()

        # Assert that no error was raised for status code 201
        self.assertIsNotNone(result)

        # Verify logging was called
        self.mock_logging_obj.post_call.assert_called_once()

    @patch.object(HTTPHandler, "post")
    def test_sync_http_status_201(self, mock_post):
        """Test that sync make_sync_call handles HTTP 201 status code correctly"""
        # Create a mock response with status code 201
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.iter_lines = MagicMock()
        mock_response.iter_lines.return_value = ["test response"]
        mock_post.return_value = mock_response

        # Call the make_sync_call function
        result = make_sync_call(
            client=None,
            gemini_client=None,
            api_base="https://mock-vertex-ai-api.com",
            headers=self.mock_headers,
            data=self.mock_data,
            model=self.mock_model,
            messages=self.messages,
            logging_obj=self.mock_logging_obj,
        )

        # Assert that no error was raised for status code 201
        self.assertIsNotNone(result)

        # Verify logging was called
        self.mock_logging_obj.post_call.assert_called_once()

    @patch.object(HTTPHandler, "post")
    def test_sync_http_status_error(self, mock_post):
        """Test that an error is raised for non-200/201 status codes"""
        # Create a mock response with status code 400
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.read = MagicMock(return_value=b"Bad Request")
        mock_response.headers = {}
        mock_post.return_value = mock_response

        # Call the make_sync_call function and expect an error
        with self.assertRaises(VertexAIError) as context:
            make_sync_call(
                client=None,
                gemini_client=None,
                api_base="https://mock-vertex-ai-api.com",
                headers=self.mock_headers,
                data=self.mock_data,
                model=self.mock_model,
                messages=self.messages,
                logging_obj=self.mock_logging_obj,
            )

        # Assert that the error has the correct status code
        self.assertEqual(context.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
