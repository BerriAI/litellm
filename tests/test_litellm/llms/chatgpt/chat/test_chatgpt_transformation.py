import os
import sys

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig


def test_chatgpt_transforms_system_messages_to_developer_role():
    config = ChatGPTConfig()
    messages = [
        {"role": "system", "content": "Follow the policy."},
        {"role": "user", "content": "Hello"},
    ]

    transformed = config._transform_messages(messages, model="gpt-5.5")

    assert transformed == [
        {"role": "developer", "content": "Follow the policy."},
        {"role": "user", "content": "Hello"},
    ]


def test_chatgpt_preserves_developer_messages_in_main_role_translation():
    config = ChatGPTConfig()
    messages = [
        {"role": "developer", "content": "Follow the policy."},
        {"role": "user", "content": "Hello"},
    ]

    assert config.translate_developer_role_to_system_role(messages) == messages
