import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from unittest.mock import AsyncMock, patch

import pytest

from litellm.a2a_protocol.providers.config_manager import A2AProviderConfigManager
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_flow_id_cannot_be_overridden_via_optional_params():
    from litellm.llms.langflow.chat.transformation import LangFlowConfig, LangFlowError

    config = LangFlowConfig()
    url = config.get_complete_url(
        api_base="http://localhost:7860",
        api_key=None,
        model="langflow/authorized-flow",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url.endswith("/api/v1/run/authorized-flow")

    with pytest.raises(LangFlowError):
        config.get_complete_url(
            api_base="http://localhost:7860",
            api_key=None,
            model="langflow/authorized-flow",
            optional_params={"flow_id": "malicious-flow"},
            litellm_params={},
            stream=False,
        )


def test_langflow_config_get_complete_url():
    from litellm.llms.langflow.chat.transformation import LangFlowConfig

    config = LangFlowConfig()
    url = config.get_complete_url(
        api_base="http://localhost:7860",
        api_key=None,
        model="langflow/my-flow-id",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == "http://localhost:7860/api/v1/run/my-flow-id"


def test_langflow_config_flow_id_is_path_segment_encoded():
    from litellm.llms.langflow.chat.transformation import LangFlowConfig

    config = LangFlowConfig()
    url = config.get_complete_url(
        api_base="http://localhost:7860",
        api_key=None,
        model="langflow/../../secret?x=1",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == "http://localhost:7860/api/v1/run/..%2F..%2Fsecret%3Fx%3D1"
    assert "/api/v1/run/" in url
    assert url.rsplit("/api/v1/run/", 1)[1] not in ("..", "../..")


@pytest.mark.parametrize("model", ["langflow/", "langflow/   "])
def test_langflow_config_rejects_empty_flow_id(model):
    from litellm.llms.langflow.chat.transformation import LangFlowConfig, LangFlowError

    config = LangFlowConfig()
    with pytest.raises(LangFlowError):
        config.get_complete_url(
            api_base="http://localhost:7860",
            api_key=None,
            model=model,
            optional_params={},
            litellm_params={},
            stream=False,
        )


def test_langflow_config_strips_flow_id_whitespace():
    from litellm.llms.langflow.chat.transformation import LangFlowConfig

    config = LangFlowConfig()
    url = config.get_complete_url(
        api_base="http://localhost:7860",
        api_key=None,
        model="langflow/  my-flow-id  ",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == "http://localhost:7860/api/v1/run/my-flow-id"


def test_langflow_config_transform_request_includes_session_id():
    from litellm.llms.langflow.chat.transformation import LangFlowConfig

    config = LangFlowConfig()
    request = config.transform_request(
        model="langflow/my-flow-id",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={"session_id": "sess-abc"},
        litellm_params={},
        headers={},
    )

    assert request["input_value"] == "hello"
    assert request["input_type"] == "chat"
    assert request["output_type"] == "chat"
    assert request["session_id"] == "sess-abc"


def test_langflow_config_rejects_tweaks_from_request_params():
    from litellm.llms.langflow.chat.transformation import LangFlowConfig, LangFlowError

    config = LangFlowConfig()
    with pytest.raises(LangFlowError):
        config.transform_request(
            model="langflow/my-flow-id",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"tweaks": {"HttpComponent": {"url": "http://attacker"}}},
            litellm_params={},
            headers={},
        )


def test_langflow_config_rejects_tweaks_from_request_body():
    from litellm.llms.langflow.chat.transformation import LangFlowConfig, LangFlowError

    config = LangFlowConfig()
    with pytest.raises(LangFlowError):
        config.sign_request(
            headers={},
            optional_params={},
            request_data={
                "input_value": "hi",
                "tweaks": {"HttpComponent": {"url": "http://attacker"}},
            },
            api_base="http://localhost:7860",
        )


def test_langflow_extra_body_cannot_inject_tweaks_into_run_payload():
    import json
    from unittest.mock import MagicMock

    import httpx

    import litellm
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    posted_bodies = []

    def fake_post(*args, **kwargs):
        body = kwargs.get("data")
        posted_bodies.append(json.loads(body) if isinstance(body, str) else body)
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "outputs": [{"outputs": [{"results": {"message": {"text": "hi"}}}]}]
        }
        resp.headers = {}
        resp.text = "{}"
        return resp

    with patch.object(HTTPHandler, "post", side_effect=fake_post):
        with pytest.raises(Exception):
            litellm.completion(
                model="langflow/my-flow",
                messages=[{"role": "user", "content": "hello"}],
                api_base="http://example.com",
                api_key="sk-test",
                extra_body={"tweaks": {"HttpComponent": {"url": "http://attacker"}}},
            )

    assert all("tweaks" not in (body or {}) for body in posted_bodies)


def test_langflow_config_extract_response():
    from litellm.llms.langflow.chat.transformation import LangFlowConfig

    config = LangFlowConfig()
    content = config._extract_content_from_response(
        {
            "session_id": "sess-abc",
            "outputs": [
                {
                    "outputs": [
                        {
                            "results": {
                                "message": {"text": "Hello from LangFlow"},
                            }
                        }
                    ]
                }
            ],
        }
    )
    assert content == "Hello from LangFlow"


def test_langflow_extract_response_returns_none_when_no_message():
    from litellm.llms.langflow.chat.transformation import LangFlowConfig

    config = LangFlowConfig()
    assert config._extract_content_from_response({"outputs": []}) is None
    assert config._extract_content_from_response({"detail": "flow failed"}) is None
    assert (
        config._extract_content_from_response(
            {"outputs": [{"outputs": [{"results": {"message": {"text": ""}}}]}]}
        )
        is None
    )


def test_langflow_transform_response_raises_on_unparseable_body():
    import httpx

    from litellm.llms.langflow.chat.transformation import LangFlowConfig, LangFlowError
    from litellm.types.utils import ModelResponse

    config = LangFlowConfig()
    raw_response = httpx.Response(status_code=200, json={"detail": "flow failed"})

    with pytest.raises(LangFlowError):
        config.transform_response(
            model="langflow/my-flow-id",
            raw_response=raw_response,
            model_response=ModelResponse(),
            logging_obj=None,
            request_data={},
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )


def test_langflow_provider_config_registered():
    cfg = ProviderConfigManager.get_provider_chat_config(
        model="langflow/flow-1",
        provider=LlmProviders.LANGFLOW,
    )
    assert cfg is not None
    assert cfg.__class__.__name__ == "LangFlowConfig"


def test_merge_a2a_session_into_litellm_params():
    from litellm.llms.langflow.a2a import merge_a2a_session_into_litellm_params

    merged = merge_a2a_session_into_litellm_params(
        {"custom_llm_provider": "langflow", "model": "langflow/flow-1"},
        {"message": {"contextId": "shared-session-99"}},
    )
    assert merged["session_id"] == "shared-session-99"


def test_langflow_a2a_provider_config_registered():
    cfg = A2AProviderConfigManager.get_provider_config(
        custom_llm_provider="langflow",
        model="langflow/flow-1",
    )
    assert cfg is not None
    assert cfg.__class__.__name__ == "LangFlowA2AConfig"


@pytest.mark.asyncio
async def test_langflow_a2a_config_passes_session_id_to_completion():
    from litellm.a2a_protocol.providers.langflow.config import LangFlowA2AConfig

    mock_response = type(
        "R",
        (),
        {
            "choices": [
                type(
                    "C",
                    (),
                    {"message": type("M", (), {"content": "ok"})()},
                )()
            ]
        },
    )()

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response

        await LangFlowA2AConfig().handle_non_streaming(
            request_id="req-1",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hi"}],
                    "contextId": "shared-session-99",
                }
            },
            litellm_params={
                "custom_llm_provider": "langflow",
                "model": "langflow/flow-1",
                "api_base": "http://localhost:7860",
            },
            api_base="http://localhost:7860",
        )

        assert (
            mock_acompletion.call_args.kwargs.get("session_id") == "shared-session-99"
        )
