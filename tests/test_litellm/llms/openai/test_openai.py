from litellm.llms.openai.openai import OpenAIConfig


def _developer_message_after_a_user_turn():
    return [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "developer", "content": "Answer with exactly one word."},
        {"role": "user", "content": "What is the capital of France?"},
    ]


class TestTranslateDeveloperRoleToSystemRole:
    def test_hosted_openai_keeps_the_developer_message_in_place(self):
        messages = OpenAIConfig().translate_developer_role_to_system_role(
            messages=_developer_message_after_a_user_turn(),
            custom_llm_provider="openai",
            api_base="https://api.openai.com/v1",
        )
        assert [message["role"] for message in messages] == ["system", "user", "assistant", "system", "user"]
        assert messages[3]["content"] == "Answer with exactly one word."

    def test_openai_compatible_provider_hoists_the_developer_message(self):
        messages = OpenAIConfig().translate_developer_role_to_system_role(
            messages=_developer_message_after_a_user_turn(),
            custom_llm_provider="azure_ai",
            api_base="https://example.services.ai.azure.com/models",
        )
        assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
        assert messages[0]["content"] == "You are terse.\n\nAnswer with exactly one word."
