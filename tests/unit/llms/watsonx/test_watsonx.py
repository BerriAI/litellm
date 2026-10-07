import asyncio
import importlib
import json
from typing import Final, Optional, cast
from unittest.mock import Mock, patch

import httpx
import pytest

import litellm
from litellm import completion, embedding
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


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

    monkeypatch.setattr(huggingface_template_handler, "get_httpx_client", forbid_sync_client)
    monkeypatch.setattr(
        huggingface_template_handler, "get_async_httpx_client", lambda **kwargs: Mock(get=serve_hf_file)
    )

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


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def setup_and_teardown(event_loop):
    import litellm

    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    asyncio.set_event_loop(event_loop)
    yield
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
    "cohere_key": getattr(litellm, "cohere_key", None),
}

@pytest.fixture
def watsonx_env_vars(monkeypatch):
    """Set required WatsonX env vars so the provider passes validation.
    Also clear WATSONX_ZENAPIKEY/WATSONX_TOKEN so they don't bypass the IAM token mock.
    """
    monkeypatch.setenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")
    monkeypatch.delenv("WATSONX_ZENAPIKEY", raising=False)
    monkeypatch.delenv("WATSONX_TOKEN", raising=False)

@pytest.fixture
def watsonx_chat_completion_call():
    def _call(
        model="watsonx/my-test-model",
        messages=None,
        api_key="test_api_key",
        space_id: Optional[str] = None,
        headers=None,
        client=None,
        patch_token_call=True,
    ):
        if messages is None:
            messages = [{"role": "user", "content": "Hello, how are you?"}]
        if client is None:
            client = HTTPHandler()

        if patch_token_call:
            mock_response = Mock()
            mock_response.json.return_value = {
                "access_token": "mock_access_token",
                "expires_in": 3600,
            }
            mock_response.raise_for_status = Mock()  # No-op to simulate no exception

            with (
                patch.object(client, "post") as mock_post,
                patch.object(litellm.module_level_client, "post", return_value=mock_response) as mock_get,
            ):
                try:
                    completion(
                        model=model,
                        messages=messages,
                        api_key=api_key,
                        headers=headers or {},
                        client=client,
                        space_id=space_id,
                    )
                except Exception as e:
                    print(e)

                return mock_post, mock_get
        else:
            with patch.object(client, "post") as mock_post:
                try:
                    completion(
                        model=model,
                        messages=messages,
                        api_key=api_key,
                        headers=headers or {},
                        client=client,
                        space_id=space_id,
                    )
                except Exception as e:
                    print(e)
                return mock_post, None

    return _call

@pytest.fixture
def watsonx_embedding_call():
    def _call(
        model="watsonx/my-test-model",
        input=None,
        api_key="test_api_key",
        space_id: Optional[str] = None,
        headers=None,
        client=None,
        patch_token_call=True,
    ):
        if input is None:
            input = ["Hello, how are you?"]
        if client is None:
            client = HTTPHandler()

        if patch_token_call:
            mock_response = Mock()
            mock_response.json.return_value = {
                "access_token": "mock_access_token",
                "expires_in": 3600,
            }
            mock_response.raise_for_status = Mock()  # No-op to simulate no exception

            with (
                patch.object(client, "post") as mock_post,
                patch.object(litellm.module_level_client, "post", return_value=mock_response) as mock_get,
            ):
                try:
                    embedding(
                        model=model,
                        input=input,
                        api_key=api_key,
                        headers=headers or {},
                        client=client,
                        space_id=space_id,
                    )
                except Exception as e:
                    print(e)

                return mock_post, mock_get
        else:
            with patch.object(client, "post") as mock_post:
                try:
                    embedding(
                        model=model,
                        input=input,
                        api_key=api_key,
                        headers=headers or {},
                        client=client,
                        space_id=space_id,
                    )
                except Exception as e:
                    print(e)
                return mock_post, None

    return _call

@pytest.mark.usefixtures("watsonx_env_vars", "_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.parametrize("with_custom_auth_header", [True, False])
def test_watsonx_custom_auth_header(with_custom_auth_header, watsonx_chat_completion_call):
    headers = {"Authorization": "Bearer my-custom-auth-header"} if with_custom_auth_header else {}

    mock_post, _ = watsonx_chat_completion_call(headers=headers)

    assert mock_post.call_count == 1
    if with_custom_auth_header:
        assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer my-custom-auth-header"
    else:
        assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer mock_access_token"

@pytest.mark.usefixtures("watsonx_env_vars", "_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.parametrize("env_var_key", ["WATSONX_ZENAPIKEY", "WATSONX_TOKEN"])
def test_watsonx_token_in_env_var(monkeypatch, watsonx_chat_completion_call, env_var_key):
    monkeypatch.setenv(env_var_key, "my-custom-token")

    mock_post, _ = watsonx_chat_completion_call(patch_token_call=False)

    assert mock_post.call_count == 1
    if env_var_key == "WATSONX_ZENAPIKEY":
        assert mock_post.call_args[1]["headers"]["Authorization"] == "ZenApiKey my-custom-token"
    else:
        assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer my-custom-token"

@pytest.mark.usefixtures("watsonx_env_vars", "_vcr_outcome_gate", "setup_and_teardown")
def test_watsonx_chat_completions_endpoint(watsonx_chat_completion_call):
    model = "watsonx/another-model"
    messages = [{"role": "user", "content": "Test message"}]

    mock_post, _ = watsonx_chat_completion_call(model=model, messages=messages)

    assert mock_post.call_count == 1
    assert "deployment" not in mock_post.call_args.kwargs["url"]


@pytest.mark.parametrize("sync_mode", [True])
def test_watsonx_tool_choice(sync_mode: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATSONX_API_KEY", "mock-api-key")
    monkeypatch.setenv("WATSONX_TOKEN", "mock-watsonx-token")
    monkeypatch.setenv("WATSONX_API_BASE", "https://us-south.ml.cloud.ibm.com")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "mock-project-id")
    model: Final = "watsonx/meta-llama/llama-3-1-8b-instruct"
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages: Final = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    def handle_request(request: httpx.Request) -> httpx.Response:
        request_body: Final = cast(dict[str, object], json.loads(request.content))
        assert request_body["tool_choice_option"] == "auto"
        return httpx.Response(
            200,
            json={
                "model_id": "meta-llama/llama-3-1-8b-instruct",
                "results": [
                    {
                        "generated_text": "The weather is sunny.",
                        "generated_token_count": 1,
                        "input_token_count": 1,
                        "stop_reason": "eos_token",
                    }
                ],
            },
            request=request,
        )

    transport: Final = httpx.MockTransport(handle_request)
    with httpx.Client(transport=transport) as http_client:
        client: Final = HTTPHandler(client=http_client)
        response: Final = completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            client=client,
        )

    assert len(response.choices) == 1


@pytest.mark.usefixtures("watsonx_env_vars", "_vcr_outcome_gate", "setup_and_teardown")
def test_watsonx_chat_completions_endpoint_space_id(monkeypatch, watsonx_chat_completion_call):
    my_fake_space_id = "xxx-xxx-xxx-xxx-xxx"
    monkeypatch.setenv("WATSONX_SPACE_ID", my_fake_space_id)

    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)

    model = "watsonx/another-model"
    messages = [{"role": "user", "content": "Test message"}]

    mock_post, _ = watsonx_chat_completion_call(model=model, messages=messages)

    assert mock_post.call_count == 1
    assert "deployment" not in mock_post.call_args.kwargs["url"]

    json_data = json.loads(mock_post.call_args.kwargs["data"])
    assert my_fake_space_id == json_data["space_id"]
    assert not json_data.get("project_id")

@pytest.mark.usefixtures("watsonx_env_vars", "_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.parametrize(
    "model",
    [
        "watsonx/deployment/<xxxx.xxx.xxx.xxxx>",
        "watsonx_text/deployment/<xxxx.xxx.xxx.xxxx>",
    ],
)
def test_watsonx_deployment_space_id(monkeypatch, watsonx_chat_completion_call, model):
    my_fake_space_id = "xxx-xxx-xxx-xxx-xxx"
    monkeypatch.setenv("WATSONX_SPACE_ID", my_fake_space_id)

    mock_post, _ = watsonx_chat_completion_call(
        model=model,
        messages=[{"content": "Hello, how are you?", "role": "user"}],
    )

    assert mock_post.call_count == 1
    json_data = json.loads(mock_post.call_args.kwargs["data"])
    assert my_fake_space_id not in json_data

@pytest.mark.usefixtures("watsonx_env_vars", "_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.parametrize(
    "model",
    [
        "watsonx/deployment/<xxxx.xxx.xxx.xxxx>",
        "watsonx_text/deployment/<xxxx.xxx.xxx.xxxx>",
    ],
)
def test_watsonx_deployment(watsonx_chat_completion_call, model):
    messages = [{"content": "Hello, how are you?", "role": "user"}]
    mock_post, _ = watsonx_chat_completion_call(
        model=model,
        messages=messages,
    )

    assert mock_post.call_count == 1
    json_data = json.loads(mock_post.call_args.kwargs["data"])

    # nor space_id or project_id is required by wx.ai API when inferencing deployment
    assert "project_id" not in json_data and "space_id" not in json_data

@pytest.mark.usefixtures("watsonx_env_vars", "_vcr_outcome_gate", "setup_and_teardown")
def test_watsonx_deployment_space_id_embedding(monkeypatch, watsonx_embedding_call):
    my_fake_space_id = "xxx-xxx-xxx-xxx-xxx"
    monkeypatch.setenv("WATSONX_SPACE_ID", my_fake_space_id)

    mock_post, _ = watsonx_embedding_call(model="watsonx/deployment/my-test-model")

    assert mock_post.call_count == 1
    json_data = json.loads(mock_post.call_args.kwargs["data"])

    # nor space_id or project_id is required by wx.ai API when inferencing deployment
    assert "project_id" not in json_data and "space_id" not in json_data
