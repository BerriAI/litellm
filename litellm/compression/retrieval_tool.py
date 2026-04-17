"""
Build the litellm_content_retrieve tool definition for the LLM.
"""

from typing import List


def build_retrieval_tool(available_keys: List[str]) -> dict:
    """
    Return an OpenAI-format tool definition that lets the model
    retrieve the full content of a compressed message.
    """
    return {
        "type": "function",
        "function": {
            "name": "litellm_content_retrieve",
            "description": (
                "Retrieve the full content of a file or message that was "
                "compressed to save tokens. Use this when you need the complete "
                "content to answer accurately. Available keys: "
                + ", ".join(available_keys)
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The identifier of the content to retrieve",
                        "enum": available_keys,
                    }
                },
                "required": ["key"],
            },
        },
    }
