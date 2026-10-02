import json
from unittest.mock import Mock

import pytest

import litellm


@pytest.mark.parametrize("tokenizer_config_cached", [False, True], ids=["tokenizer_config", "cached_config_jinja"])
async def test_watsonx_text_gpt_oss_async_completion_fetches_hf_template_off_the_event_loop(
    monkeypatch, tokenizer_config_cached
):
    import httpx

    from litellm._uuid import uuid
    from litellm.litellm_core_utils.prompt_templates import huggingface_template_handler
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    hf_model = f"openai/gpt-oss-{uuid.uuid4()}"
    chat_template = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}"
    if tokenizer_config_cached:
        cached_config = {"status": "success", "tokenizer": {"bos_token": None, "eos_token": None}}
        monkeypatch.setattr(litellm, "known_tokenizer_config", {hf_model: cached_config})
        expected_fetch = f"https://huggingface.co/{hf_model}/raw/main/chat_template.jinja"
    else:
        monkeypatch.setattr(litellm, "known_tokenizer_config", {})
        expected_fetch = f"https://huggingface.co/{hf_model}/raw/main/tokenizer_config.json"
    hf_fetched = []
    captured = {}

    def forbid_sync_client():
        raise AssertionError("sync HuggingFace fetch ran on the request path")

    async def serve_hf_file(url, **kwargs):
        hf_fetched.append(url)
        if url.endswith(".jinja"):
            return httpx.Response(200, content=chat_template.encode())
        return httpx.Response(200, json={"chat_template": chat_template, "bos_token": None, "eos_token": None})

    monkeypatch.setattr(huggingface_template_handler, "_get_httpx_client", forbid_sync_client)
    monkeypatch.setattr(huggingface_template_handler, "get_async_httpx_client", lambda **kwargs: Mock(get=serve_hf_file))

    def handle(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model_id": hf_model,
                "results": [
                    {
                        "generated_text": "Hi",
                        "generated_token_count": 1,
                        "input_token_count": 1,
                        "stop_reason": "eos_token",
                    }
                ],
            },
        )

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))

    response = await litellm.acompletion(
        model=f"watsonx_text/{hf_model}",
        messages=[{"role": "user", "content": "Hi there"}],
        api_base="https://test-api.watsonx.ai",
        project_id="test-project-id",
        token="test-token",
        client=client,
    )

    assert response.choices[0].message.content == "Hi"
    assert hf_fetched == [expected_fetch]
    assert captured["body"]["input"] == "<|user|>Hi there"
