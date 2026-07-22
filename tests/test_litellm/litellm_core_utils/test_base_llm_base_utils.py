from litellm.llms.base_llm.base_utils import map_developer_role_to_system_role
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def test_map_developer_role_leaves_messages_without_developer_role_unchanged():
    messages = [
        {"role": "system", "content": "Follow the product policy."},
        {"role": "user", "content": "Hello!"},
    ]

    assert map_developer_role_to_system_role(messages=messages) is messages


def test_map_developer_role_merges_leading_system_equivalent_messages():
    messages = [
        {"role": "system", "content": "Follow the product policy."},
        {"role": "developer", "content": "Prefer concise answers."},
        {"role": "system", "content": "Use markdown only when helpful."},
        {"role": "user", "content": "Hello!"},
    ]

    result = map_developer_role_to_system_role(messages=messages)

    assert result == [
        {
            "role": "system",
            "content": (
                "Follow the product policy.\n\n"
                "Prefer concise answers.\n\n"
                "Use markdown only when helpful."
            ),
        },
        {"role": "user", "content": "Hello!"},
    ]


def test_map_developer_role_preserves_structured_leading_system_content():
    messages = [
        {"role": "developer", "content": ""},
        {"role": "system", "content": [{"type": "text", "text": "System rules."}]},
        {"role": "developer", "content": None},
        {"role": "developer", "content": "Developer rules."},
        {"role": "user", "content": "Hello!"},
    ]

    result = map_developer_role_to_system_role(messages=messages)

    assert result == [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "System rules.\n\nDeveloper rules."},
            ],
        },
        {"role": "user", "content": "Hello!"},
    ]


def test_map_developer_role_avoids_standalone_separator_blocks():
    messages = [
        {"role": "system", "content": [{"type": "image", "url": "policy.png"}]},
        {"role": "developer", "content": "Developer rules."},
        {"role": "user", "content": "Hello!"},
    ]

    result = map_developer_role_to_system_role(messages=messages)

    assert result == [
        {
            "role": "system",
            "content": [
                {"type": "image", "url": "policy.png"},
                {"type": "text", "text": "\n\nDeveloper rules."},
            ],
        },
        {"role": "user", "content": "Hello!"},
    ]


def test_map_developer_role_converts_later_developer_messages_in_place():
    messages = [
        {"role": "system", "content": "Follow the product policy."},
        {"role": "user", "content": "Hello!"},
        {"role": "developer", "content": "Prefer concise answers."},
        {"role": "assistant", "content": "Hi."},
    ]

    result = map_developer_role_to_system_role(messages=messages)

    assert result == [
        {"role": "system", "content": "Follow the product policy."},
        {"role": "user", "content": "Hello!"},
        {"role": "system", "content": "Prefer concise answers."},
        {"role": "assistant", "content": "Hi."},
    ]


def test_map_developer_role_converts_later_developer_without_leading_system():
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "system", "content": "Keep later system messages in place."},
        {"role": "developer", "content": "Prefer concise answers."},
    ]

    result = map_developer_role_to_system_role(messages=messages)

    assert result == [
        {"role": "user", "content": "Hello!"},
        {"role": "system", "content": "Keep later system messages in place."},
        {"role": "system", "content": "Prefer concise answers."},
    ]


def test_responses_instructions_and_developer_input_become_single_system_message():
    request = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model="anthropic/claude-sonnet-4-5",
        input=[
            {"role": "developer", "content": "Prefer concise answers."},
            {"role": "user", "content": "Hello!"},
        ],
        responses_api_request={"instructions": "Follow the product policy."},
    )

    result = map_developer_role_to_system_role(messages=request["messages"])

    assert result == [
        {
            "role": "system",
            "content": "Follow the product policy.\n\nPrefer concise answers.",
        },
        {"role": "user", "content": "Hello!"},
    ]
