import json
from unittest.mock import Mock, patch

import pytest

from litellm.llms.bedrock.passthrough.transformation import BedrockPassthroughConfig


class TestBedrockPassthroughTransformation:
    def test_handle_logging_collected_chunks(self):
        """Test handle_logging_collected_chunks method with real bedrock streaming data"""

        # Create an instance of BedrockPassthroughConfig
        config = BedrockPassthroughConfig()

        # Mock the LiteLLMLoggingObj
        mock_logging_obj = Mock()

        # Test data from the provided input
        all_chunks = [
            '{"type":"message_start","message":{"id":"msg_bdrk_019gLAysNkE5wfd7ZwVqXTfY","type":"message","role":"assistant","model":"claude-3-5-sonnet-20240620","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":86,"output_tokens":1}}}',
            '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{"}}',
            '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\\n  \\"isNewTopic\\":"}}',
            '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" false,\\n  \\"title\\": null"}}',
            '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\\n}"}}',
            '{"type":"content_block_stop","index":0}',
            '{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":22}}',
            '{"type":"message_stop","amazon-bedrock-invocationMetrics":{"inputTokenCount":86,"outputTokenCount":22,"invocationLatency":925,"firstByteLatency":921}}',
        ]

        model = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        custom_llm_provider = "bedrock"
        endpoint = "/model/us.anthropic.claude-3-5-sonnet-20240620-v1:0/invoke-with-response-stream"

        # Mock the AWSEventStreamDecoder to avoid actual AWS dependencies
        with patch(
            "litellm.llms.bedrock.chat.invoke_handler.AWSEventStreamDecoder"
        ) as mock_decoder_class:
            mock_decoder = Mock()
            mock_decoder_class.return_value = mock_decoder

            # Configure the mock to return some test chunks
            mock_decoder._chunk_parser.side_effect = [
                Mock(id="chunk_1"),
                Mock(id="chunk_2"),
                Mock(id="chunk_3"),
                Mock(id="chunk_4"),
                Mock(id="chunk_5"),
                Mock(id="chunk_6"),
                Mock(id="chunk_7"),
                Mock(id="chunk_8"),
                Mock(id="chunk_9"),
            ]

            # Call the method under test
            result = config.handle_logging_collected_chunks(
                all_chunks=all_chunks,
                litellm_logging_obj=mock_logging_obj,
                model=model,
                custom_llm_provider=custom_llm_provider,
                endpoint=endpoint,
            )

            # Assertions
            assert result is None  # Method currently returns None

            # Verify that AWSEventStreamDecoder was instantiated with correct model
            mock_decoder_class.assert_called_once_with(model=model)

            # Verify that _chunk_parser was called for each chunk
            assert mock_decoder._chunk_parser.call_count == len(all_chunks)

            # Verify that each chunk was properly parsed as JSON and passed to _chunk_parser
            for i, chunk in enumerate(all_chunks):
                call_args = mock_decoder._chunk_parser.call_args_list[i]
                expected_chunk_data = json.loads(chunk)
                actual_chunk_data = call_args[1]["chunk_data"]
                assert actual_chunk_data == expected_chunk_data

    def test_handle_logging_collected_chunks_empty_chunks(self):
        """Test handle_logging_collected_chunks with empty chunks list"""

        config = BedrockPassthroughConfig()
        mock_logging_obj = Mock()

        with patch(
            "litellm.llms.bedrock.chat.invoke_handler.AWSEventStreamDecoder"
        ) as mock_decoder_class:
            mock_decoder = Mock()
            mock_decoder_class.return_value = mock_decoder

            result = config.handle_logging_collected_chunks(
                all_chunks=[],
                litellm_logging_obj=mock_logging_obj,
                model="test-model",
                custom_llm_provider="bedrock",
                endpoint="/test-endpoint",
            )

            assert result is None
            mock_decoder_class.assert_called_once_with(model="test-model")
            mock_decoder._chunk_parser.assert_not_called()

    def test_handle_logging_collected_chunks_invalid_json(self):
        """Test handle_logging_collected_chunks with invalid JSON in chunks"""

        config = BedrockPassthroughConfig()
        mock_logging_obj = Mock()

        invalid_chunks = ['{"invalid": json}', "not json at all"]

        with patch(
            "litellm.llms.bedrock.chat.invoke_handler.AWSEventStreamDecoder"
        ) as mock_decoder_class:
            mock_decoder = Mock()
            mock_decoder_class.return_value = mock_decoder

            # This should raise a JSON decode error
            with pytest.raises(json.JSONDecodeError):
                config.handle_logging_collected_chunks(
                    all_chunks=invalid_chunks,
                    litellm_logging_obj=mock_logging_obj,
                    model="test-model",
                    custom_llm_provider="bedrock",
                    endpoint="/test-endpoint",
                )
