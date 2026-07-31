from litellm.llms.litellm_proxy.chat.transformation import LiteLLMProxyChatConfig


def test_litellm_proxy_chat_transformation():
    """
    Assert messages are not transformed when calling litellm proxy
    """
    config = LiteLLMProxyChatConfig()
    file_content = [
        {"type": "text", "text": "What is this document about?"},
        {
            "type": "file",
            "file": {
                "file_id": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                "format": "application/pdf",
            },
        },
    ]
    messages = [{"role": "user", "content": file_content}]
    assert config.transform_request(
        model="model",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    ) == {"model": "model", "messages": messages}


def test_litellm_gateway_from_sdk_with_user_param():
    from litellm.llms.litellm_proxy.chat.transformation import LiteLLMProxyChatConfig

    supported_params = LiteLLMProxyChatConfig().get_supported_openai_params(
        "openai/gpt-4o"
    )
    assert "user" in supported_params


def test_litellm_proxy_forwards_request_metadata_to_downstream_proxy():
    config = LiteLLMProxyChatConfig()
    messages = [{"role": "user", "content": "hello"}]

    request = config.transform_request(
        model="gpt-4o",
        messages=messages,
        optional_params={},
        litellm_params={
            "metadata": {
                "requester_metadata": {
                    "tags": ["project-x", "cost-center-42"],
                    "user_api_key": "do-not-forward",
                }
            },
            "litellm_session_id": "session-abc-123",
        },
        headers={},
    )

    assert request["metadata"] == {"tags": ["project-x", "cost-center-42"]}
    assert request["extra_body"] == {"litellm_session_id": "session-abc-123"}
