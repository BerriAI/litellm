import json
from litellm.integrations.vector_stores.bedrock_vector_store import BedrockVectorStore
from litellm.types.integrations.rag.bedrock_knowledgebase import BedrockKBResponse
import pytest 

def test_get_chat_completion_message_from_bedrock_kb_response():
    test_metadata = {
        "source": "litellm_docs",
        "id": "12345",
    }

    response = BedrockKBResponse(
        retrievalResults=[
            {
                "content": {
                    "text": "Litellm is a lightweight language model.",
                    "type": "text",
                },
                "metadata": test_metadata,
            }
        ]
    )
    message, context_string = BedrockVectorStore.get_chat_completion_message_from_bedrock_kb_response(response)

    expected_context_string = f"<context>\n<section>\n<content>\nLitellm is a lightweight language model.\n</content>\n<metadata>\n{json.dumps(test_metadata, indent=4, default=str)}\n</metadata>\n</section>\n</context>"

    assert context_string == expected_context_string
    assert message == {
        "role": "user",
        "content": expected_context_string,
    }

def test_get_chat_completion_message_from_bedrock_kb_response_multiple_results():    
    test_metadata_1 = {
        "source": "litellm_docs",
        "id": "12345",
    }

    test_metadata_2 = {
        "source": "litellm_docs",
        "id": "77777",
    }

    response = BedrockKBResponse(
        retrievalResults=[
            {
                "content": {
                    "text": "Litellm is a lightweight language model.",
                    "type": "text",
                },
                "metadata": test_metadata_1,
            },
            {
                "content": {
                    "text": "Litellm is designed for simplicity and efficiency.",
                    "type": "text",
                },
                "metadata": test_metadata_2,
            }
        ]
    )
    message, context_string = BedrockVectorStore.get_chat_completion_message_from_bedrock_kb_response(response)

    expected_context_string = f"<context>\n<section>\n<content>\nLitellm is a lightweight language model.\n</content>\n<metadata>\n{json.dumps(test_metadata_1, indent=4, default=str)}\n</metadata>\n</section>\n<section>\n<content>\nLitellm is designed for simplicity and efficiency.\n</content>\n<metadata>\n{json.dumps(test_metadata_2, indent=4, default=str)}\n</metadata>\n</section>\n</context>"

    assert context_string == expected_context_string
    assert message == {
        "role": "user",
        "content": expected_context_string,
    }


def test_get_chat_completion_message_from_bedrock_kb_response_no_metadata():
    response = BedrockKBResponse(
        retrievalResults=[
            {
                "content": {
                    "text": "Litellm is a lightweight language model.",
                    "type": "text",
                },
                "metadata": None
            }
        ]
    )
    message, context_string = BedrockVectorStore.get_chat_completion_message_from_bedrock_kb_response(response)

    expected_context_string = f"<context>\n<section>\n<content>\nLitellm is a lightweight language model.\n</content>\n</section>\n</context>"

    assert context_string == expected_context_string
    assert message == {
        "role": "user",
        "content": expected_context_string,
    }