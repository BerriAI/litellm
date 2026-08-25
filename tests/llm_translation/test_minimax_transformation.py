"""
Regression test for #38197: MiniMax M2.7 can return its entire answer
inside <think>...</think> with nothing trailing after the closing tag.
MinimaxChatConfig.transform_response() should fall back to
reasoning_content in that case instead of leaving content empty.

Scoped to MiniMax only — see the docstring on transform_response for why
this isn't in the shared _parse_content_for_reasoning function.
"""
from unittest.mock import MagicMock

from litellm.llms.minimax.chat.transformation import MinimaxChatConfig


class TestMinimaxTransformResponse:
    def test_empty_content_falls_back_to_reasoning_content(self):
        config = MinimaxChatConfig()

        fake_message = MagicMock()
        fake_message.content = ""
        fake_message.reasoning_content = "The answer to 2+2 is 4."

        fake_choice = MagicMock()
        fake_choice.message = fake_message

        fake_model_response = MagicMock()
        fake_model_response.choices = [fake_choice]

        import litellm.llms.openai.chat.gpt_transformation as parent_module

        original = parent_module.OpenAIGPTConfig.transform_response
        parent_module.OpenAIGPTConfig.transform_response = (
            lambda self, **kwargs: fake_model_response
        )

        try:
            result = config.transform_response(
                model="minimax/MiniMax-M2.7",
                raw_response=None,
                model_response=fake_model_response,
                logging_obj=None,
                request_data={},
                messages=[],
                optional_params={},
                litellm_params={},
                encoding=None,
            )
            assert result.choices[0].message.content == "The answer to 2+2 is 4."
        finally:
            parent_module.OpenAIGPTConfig.transform_response = original

    def test_normal_content_left_untouched(self):
        """Sanity check: when content is already populated, don't overwrite it."""
        config = MinimaxChatConfig()

        fake_message = MagicMock()
        fake_message.content = "The answer is 4."
        fake_message.reasoning_content = "Let me think about this."

        fake_choice = MagicMock()
        fake_choice.message = fake_message

        fake_model_response = MagicMock()
        fake_model_response.choices = [fake_choice]

        import litellm.llms.openai.chat.gpt_transformation as parent_module

        original = parent_module.OpenAIGPTConfig.transform_response
        parent_module.OpenAIGPTConfig.transform_response = (
            lambda self, **kwargs: fake_model_response
        )

        try:
            result = config.transform_response(
                model="minimax/MiniMax-M2.7",
                raw_response=None,
                model_response=fake_model_response,
                logging_obj=None,
                request_data={},
                messages=[],
                optional_params={},
                litellm_params={},
                encoding=None,
            )
            assert result.choices[0].message.content == "The answer is 4."
        finally:
            parent_module.OpenAIGPTConfig.transform_response = original
