import unittest
from litellm.litellm_core_utils.token_counter import _count_content_list

class TestInputAudioTokenCounter(unittest.TestCase):
    def test_input_audio_block_does_not_raise(self):
        """
        Regression test for #38459:
        _count_content_list should handle input_audio blocks without raising ValueError.
        """
        content_list = [
            {"type": "text", "text": "Hello audio"},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=",
                    "format": "wav"
                }
            }
        ]
        
        # Simple character counter for testing
        def mock_count_func(s):
            return len(s)
            
        total_tokens = _count_content_list(
            count_function=mock_count_func,
            content_list=content_list,
            use_default_image_token_count=True,
            default_token_count=None
        )
        
        self.assertGreater(total_tokens, len("Hello audio"))
