"""Unit tests for BurnCloud text-to-speech guardrail translation handler."""

import unittest
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from litellm.llms.burncloud.speech.guardrail_translation.handler import BurnCloudTextToSpeechHandler


class TestBurnCloudTextToSpeechHandler(unittest.TestCase):
    """Test suite for BurnCloudTextToSpeechHandler class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.handler = BurnCloudTextToSpeechHandler()
        self.test_input_text = "Hello, this is a test message."
        self.guardrailed_text = "Hello, this is a guarded test message."

    def test_init(self) -> None:
        """Test initialization of BurnCloudTextToSpeechHandler."""
        self.assertIsInstance(self.handler, BurnCloudTextToSpeechHandler)

    @patch("litellm.llms.burncloud.speech.guardrail_translation.handler.verbose_proxy_logger")
    async def test_process_input_messages_with_valid_input(self, mock_logger: Mock) -> None:
        """Test processing input messages with valid text input."""
        # Prepare test data
        data: Dict[str, Any] = {"input": self.test_input_text}

        # Create mock guardrail
        mock_guardrail = AsyncMock()
        mock_guardrail.apply_guardrail = AsyncMock(return_value=self.guardrailed_text)

        # Execute the method under test
        result = await self.handler.process_input_messages(
            data=data,
            guardrail_to_apply=mock_guardrail
        )

        # Assertions
        self.assertEqual(result["input"], self.guardrailed_text)
        mock_guardrail.apply_guardrail.assert_called_once_with(text=self.test_input_text)
        mock_logger.debug.assert_called()

    @patch("litellm.llms.burncloud.speech.guardrail_translation.handler.verbose_proxy_logger")
    async def test_process_input_messages_with_no_input(self, mock_logger: Mock) -> None:
        """Test processing input messages when no input text is provided."""
        # Prepare test data without input
        data: Dict[str, Any] = {"other_param": "value"}

        # Create mock guardrail
        mock_guardrail = AsyncMock()

        # Execute the method under test
        result = await self.handler.process_input_messages(
            data=data,
            guardrail_to_apply=mock_guardrail
        )

        # Assertions
        self.assertEqual(result, data)
        mock_guardrail.apply_guardrail.assert_not_called()
        mock_logger.debug.assert_called_once()

    @patch("litellm.llms.burncloud.speech.guardrail_translation.handler.verbose_proxy_logger")
    async def test_process_input_messages_with_non_string_input(self, mock_logger: Mock) -> None:
        """Test processing input messages with non-string input."""
        # Prepare test data with non-string input
        data: Dict[str, Any] = {"input": 12345}

        # Create mock guardrail
        mock_guardrail = AsyncMock()

        # Execute the method under test
        result = await self.handler.process_input_messages(
            data=data,
            guardrail_to_apply=mock_guardrail
        )

        # Assertions
        self.assertEqual(result, data)
        mock_guardrail.apply_guardrail.assert_not_called()
        mock_logger.debug.assert_called_once()

    @patch("litellm.llms.burncloud.speech.guardrail_translation.handler.verbose_proxy_logger")
    async def test_process_output_response(self, mock_logger: Mock) -> None:
        """Test processing output response (should return unchanged)."""
        # Prepare test data
        test_response = b"fake audio binary data"

        # Create mock guardrail
        mock_guardrail = AsyncMock()

        # Execute the method under test
        result = await self.handler.process_output_response(
            response=test_response,
            guardrail_to_apply=mock_guardrail
        )

        # Assertions
        self.assertEqual(result, test_response)
        mock_logger.debug.assert_called_once()
