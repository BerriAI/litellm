#### What this tests ####
#    This tests if prompts are being correctly formatted

import pytest


from typing import List
from unittest.mock import MagicMock, patch

# from litellm.litellm_core_utils.prompt_templates.factory import prompt_factory
import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    _bedrock_tools_pt,
    anthropic_messages_pt,
    anthropic_pt,
    claude_2_1_pt,
    convert_to_anthropic_tool_invoke,
    convert_url_to_base64,
    create_anthropic_image_param,
    get_tool_calls_from_response,
    has_tool_with_name,
    llama_2_chat_pt,
    prompt_factory,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_completion_messages,
)
from litellm.llms.vertex_ai.gemini.transformation import (
    gemini_convert_messages_with_history,
)
from litellm.types.llms.openai import AllMessageValues












# codellama_prompt_format()


def test_convert_url_to_img():
    response_url = convert_url_to_base64(
        url="https://images.pexels.com/photos/1319515/pexels-photo-1319515.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
    )

    assert "image/jpeg" in response_url




















































def test_alternating_roles_e2e():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    import json

    litellm.set_verbose = True
    http_handler = HTTPHandler()

    with patch.object(http_handler, "post", new=MagicMock()) as mock_post:
        try:
            response = litellm.completion(
                **{
                    "model": "databricks/databricks-meta-llama-3-1-70b-instruct",
                    "messages": [
                        {"role": "user", "content": "Hello!"},
                        {
                            "role": "assistant",
                            "content": "Hello! How can I assist you today?",
                        },
                        {"role": "user", "content": "What is Databricks?"},
                        {"role": "user", "content": "What is Azure?"},
                        {
                            "role": "assistant",
                            "content": "I don't know anyything, do you?",
                        },
                        {"role": "assistant", "content": "I can't repeat sentences."},
                    ],
                    "user_continue_message": {
                        "role": "user",
                        "content": "Ok",
                    },
                    "assistant_continue_message": {
                        "role": "assistant",
                        "content": "Please continue",
                    },
                    "ensure_alternating_roles": True,
                },
                client=http_handler,
            )
        except Exception as e:
            print(f"error: {e}")

        assert mock_post.call_args.kwargs["data"] == json.dumps(
            {
                "model": "databricks-meta-llama-3-1-70b-instruct",
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {
                        "role": "assistant",
                        "content": "Hello! How can I assist you today?",
                    },
                    {"role": "user", "content": "What is Databricks?"},
                    {
                        "role": "assistant",
                        "content": "Please continue",
                    },
                    {"role": "user", "content": "What is Azure?"},
                    {
                        "role": "assistant",
                        "content": "I don't know anyything, do you?",
                    },
                    {
                        "role": "user",
                        "content": "Ok",
                    },
                    {
                        "role": "assistant",
                        "content": "I can't repeat sentences.",
                    },
                    {
                        "role": "user",
                        "content": "Ok",
                    },
                ],
            }
        )




def test_convert_generic_image_chunk_to_openai_image_obj():
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_generic_image_chunk_to_openai_image_obj,
        convert_to_anthropic_image_obj,
    )

    url = "https://i.pinimg.com/736x/b4/b1/be/b4b1becad04d03a9071db2817fc9fe77.jpg"
    image_obj = convert_to_anthropic_image_obj(url, format=None)
    url_str = convert_generic_image_chunk_to_openai_image_obj(image_obj)
    image_obj = convert_to_anthropic_image_obj(url_str, format=None)
    print(image_obj)






# ============ Server Tool Use Reconstruction Tests ============
# Fixes: https://github.com/BerriAI/litellm/issues/17737




















# ============ parse_tool_call_arguments Tests ============
# Tests for the shared utility that parses tool call JSON arguments










# ============ _attempt_json_repair Tests ============
# Tests for the JSON repair utility that fixes truncated tool call arguments
