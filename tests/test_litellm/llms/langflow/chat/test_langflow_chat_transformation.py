from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.langflow.chat.transformation import LangFlowConfig, LangFlowError
from litellm.types.utils import LlmProviders, ModelResponse
from litellm.utils import ProviderConfigManager


def test_flow_id_cannot_be_overridden_via_optional_params():
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


def test_langflow_config_get_complete_url_requires_api_base():
    config = LangFlowConfig()
    with pytest.raises(ValueError):
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="langflow/my-flow-id",
            optional_params={},
            litellm_params={},
            stream=False,
        )


def test_langflow_config_flow_id_is_path_segment_encoded():
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


def test_langflow_config_transform_request_uses_last_user_message():
    config = LangFlowConfig()
    request = config.transform_request(
        model="langflow/my-flow-id",
        messages=[
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ],
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert request["input_value"] == "second"
    assert "session_id" not in request


def test_langflow_config_transform_request_falls_back_to_last_message():
    config = LangFlowConfig()
    request = config.transform_request(
        model="langflow/my-flow-id",
        messages=[{"role": "assistant", "content": "only assistant"}],
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert request["input_value"] == "only assistant"


def test_langflow_config_transform_request_empty_messages():
    config = LangFlowConfig()
    request = config.transform_request(
        model="langflow/my-flow-id",
        messages=[],
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert request["input_value"] == ""


def test_langflow_config_rejects_tweaks_from_request_params():
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


def test_langflow_config_sign_request_passes_through_without_tweaks():
    config = LangFlowConfig()
    headers, body = config.sign_request(
        headers={"x-api-key": "secret"},
        optional_params={},
        request_data={"input_value": "hi"},
        api_base="http://localhost:7860",
    )
    assert headers == {"x-api-key": "secret"}
    assert body is None


def test_langflow_config_validate_environment_sets_api_key_header():
    config = LangFlowConfig()
    headers = config.validate_environment(
        headers={},
        model="langflow/my-flow-id",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        api_key="secret",
    )
    assert headers["Content-Type"] == "application/json"
    assert headers["x-api-key"] == "secret"


def test_langflow_extra_body_cannot_inject_tweaks_into_run_payload():
    import json

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


def test_langflow_config_extract_response_from_outputs_dict():
    config = LangFlowConfig()
    content = config._extract_content_from_response(
        {
            "outputs": [
                {
                    "outputs": [
                        {
                            "results": {},
                            "outputs": {
                                "message": {"message": {"text": "via outputs dict"}}
                            },
                        }
                    ]
                }
            ],
        }
    )
    assert content == "via outputs dict"


def test_langflow_extract_response_returns_none_when_no_message():
    config = LangFlowConfig()
    assert config._extract_content_from_response({"outputs": []}) is None
    assert config._extract_content_from_response({"detail": "flow failed"}) is None
    assert config._extract_content_from_response({"outputs": ["not-a-dict"]}) is None
    assert (
        config._extract_content_from_response({"outputs": [{"outputs": ["bad"]}]})
        is None
    )
    assert (
        config._extract_content_from_response(
            {"outputs": [{"outputs": [{"results": {"message": {"text": ""}}}]}]}
        )
        is None
    )


def test_langflow_transform_response_builds_model_response_with_usage():
    config = LangFlowConfig()
    raw_response = httpx.Response(
        status_code=200,
        json={
            "session_id": "sess-abc",
            "outputs": [
                {"outputs": [{"results": {"message": {"text": "Hello from LangFlow"}}}]}
            ],
        },
    )

    result = config.transform_response(
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

    assert result.choices[0].message.content == "Hello from LangFlow"
    assert result.choices[0].finish_reason == "stop"
    assert result.model == "langflow/my-flow-id"
    assert result.usage.completion_tokens > 0
    assert result.usage.total_tokens == (
        result.usage.prompt_tokens + result.usage.completion_tokens
    )


def test_langflow_transform_response_raises_on_unparseable_body():
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


def test_langflow_transform_response_raises_on_non_json_body():
    config = LangFlowConfig()
    raw_response = httpx.Response(
        status_code=200, content=b"not json", headers={"content-type": "text/plain"}
    )

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


def test_langflow_config_get_error_class():
    config = LangFlowConfig()
    err = config.get_error_class(error_message="boom", status_code=503, headers={})
    assert isinstance(err, LangFlowError)
    assert err.status_code == 503


def test_langflow_config_stream_behavior_flags():
    config = LangFlowConfig()
    assert config.supports_stream_param_in_request_body is False
    assert config.should_fake_stream(model="langflow/x", stream=True) is True
    assert config.should_fake_stream(model="langflow/x", stream=False) is False


def test_langflow_provider_config_registered():
    cfg = ProviderConfigManager.get_provider_chat_config(
        model="langflow/flow-1",
        provider=LlmProviders.LANGFLOW,
    )
    assert cfg is not None
    assert cfg.__class__.__name__ == "LangFlowConfig"
