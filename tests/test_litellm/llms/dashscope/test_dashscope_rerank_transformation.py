"""
Unit tests for DashScope rerank transformation.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest
import respx


from litellm.llms.dashscope.common_utils import DashScopeError, get_dashscope_family_rerank_config
from litellm.llms.dashscope.rerank.transformation import (
    DEFAULT_RERANK_URL,
    DashScopeRerankConfig,
)
from litellm.types.rerank import RerankResponse


class TestDashScopeRerankURL:
    def setup_method(self):
        self.config = DashScopeRerankConfig()

    def test_default_url(self):
        url = self.config.get_complete_url(api_base=None, model="qwen3-rerank")
        assert url == DEFAULT_RERANK_URL

    def test_explicit_v1_base_appends_reranks(self):
        url = self.config.get_complete_url(
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen3-rerank",
        )
        assert url == "https://dashscope.aliyuncs.com/compatible-mode/v1/reranks"

    def test_intl_v1_base_appends_reranks(self):
        url = self.config.get_complete_url(
            api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            model="qwen3-rerank",
        )
        assert url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/reranks"

    def test_already_complete_url_passthrough(self):
        full = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
        assert self.config.get_complete_url(api_base=full, model="qwen3-rerank") == full

    def test_trailing_slash_stripped(self):
        full = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks/"
        assert self.config.get_complete_url(
            api_base=full, model="qwen3-rerank"
        ) == full.rstrip("/")

    def test_custom_v1_base_appends_reranks(self):
        url = self.config.get_complete_url(
            api_base="https://my-proxy.example.com/v1", model="qwen3-rerank"
        )
        assert url == "https://my-proxy.example.com/v1/reranks"


class TestDashScopeRerankRequest:
    def setup_method(self):
        self.config = DashScopeRerankConfig()

    def test_validate_environment_with_explicit_key(self):
        headers = self.config.validate_environment(
            headers={}, model="qwen3-rerank", api_key="sk-test"
        )
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["content-type"] == "application/json"

    def test_validate_environment_missing_key(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            self.config.validate_environment(
                headers={}, model="qwen3-rerank", api_key=None
            )

    def test_validate_environment_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
        headers = self.config.validate_environment(
            headers={}, model="qwen3-rerank", api_key=None
        )
        assert headers["Authorization"] == "Bearer env-key"

    def test_supported_params(self):
        assert self.config.get_supported_cohere_rerank_params("qwen3-rerank") == [
            "query",
            "documents",
            "top_n",
            "return_documents",
        ]

    def test_map_params_drops_unsupported(self):
        # qwen3-rerank accepts query/documents/top_n/return_documents.
        # rank_fields and max_*_per_doc are silently dropped.
        params = self.config.map_cohere_rerank_params(
            non_default_params={},
            model="qwen3-rerank",
            drop_params=False,
            query="什么是文本排序模型",
            documents=["d1", "d2"],
            top_n=2,
            rank_fields=["title"],
            return_documents=True,
            max_chunks_per_doc=5,
            max_tokens_per_doc=100,
            instruction="Unsupported on the compatible protocol",
        )
        assert params == {
            "query": "什么是文本排序模型",
            "documents": ["d1", "d2"],
            "top_n": 2,
            "return_documents": True,
        }

    def test_transform_request_full(self):
        body = self.config.transform_rerank_request(
            model="qwen3-rerank",
            optional_rerank_params={
                "query": "如何制作美味的苹果派?",
                "documents": ["a", "b"],
                "top_n": 5,
                "return_documents": True,
            },
            headers={},
        )
        assert body == {
            "model": "qwen3-rerank",
            "query": "如何制作美味的苹果派?",
            "documents": ["a", "b"],
            "top_n": 5,
            "return_documents": True,
        }

    def test_transform_request_omits_unset_optional(self):
        body = self.config.transform_rerank_request(
            model="qwen3-rerank",
            optional_rerank_params={"query": "q", "documents": ["a"]},
            headers={},
        )
        assert "top_n" not in body
        assert "return_documents" not in body

    def test_transform_request_requires_query(self):
        with pytest.raises(ValueError, match="query"):
            self.config.transform_rerank_request(
                model="qwen3-rerank",
                optional_rerank_params={"documents": ["a"]},
                headers={},
            )

    def test_transform_request_requires_documents(self):
        with pytest.raises(ValueError, match="documents"):
            self.config.transform_rerank_request(
                model="qwen3-rerank",
                optional_rerank_params={"query": "q"},
                headers={},
            )


class TestDashScopeRerankResponse:
    def setup_method(self):
        self.config = DashScopeRerankConfig()
        self.logging = MagicMock()

    def _resp(self, body, status_code=200):
        return httpx.Response(
            status_code=status_code,
            content=json.dumps(body).encode(),
            request=httpx.Request("POST", "https://example.com"),
        )

    def test_success_response(self):
        body = {
            "object": "list",
            "results": [
                {"index": 0, "relevance_score": 0.93},
                {"index": 2, "relevance_score": 0.34},
            ],
            "model": "qwen3-rerank",
            "id": "85ba5752",
            "usage": {"total_tokens": 79},
        }
        out = self.config.transform_rerank_response(
            model="qwen3-rerank",
            raw_response=self._resp(body),
            model_response=RerankResponse(),
            logging_obj=self.logging,
            api_key="sk",
            request_data={"query": "q"},
        )
        assert out.id == "85ba5752"
        assert out.results == [
            {"index": 0, "relevance_score": 0.93},
            {"index": 2, "relevance_score": 0.34},
        ]
        assert out.meta == {
            "billed_units": {"total_tokens": 79},
            "tokens": {"input_tokens": 79},
        }

    def test_response_with_return_documents_real_payload(self):
        # Verbatim sample from a real qwen3-rerank call with return_documents=true.
        body = {
            "object": "list",
            "results": [
                {
                    "document": {
                        "text": "苹果派的制作步骤包括准备面团、切苹果、调制馅料、组装和烘烤。"
                    },
                    "index": 1,
                    "relevance_score": 0.8304247466067356,
                },
                {
                    "document": {
                        "text": "制作苹果派时，预先煮软苹果可以缩短烘烤时间。"
                    },
                    "index": 3,
                    "relevance_score": 0.7142660211908354,
                },
            ],
            "model": "qwen3-rerank",
            "id": "e191b077-97c4-9929-b121-c2fbd2c7b0af",
            "usage": {"total_tokens": 192},
        }
        out = self.config.transform_rerank_response(
            model="qwen3-rerank",
            raw_response=self._resp(body),
            model_response=RerankResponse(),
            logging_obj=self.logging,
            request_data={"query": "如何制作美味的苹果派?"},
        )
        assert out.id == "e191b077-97c4-9929-b121-c2fbd2c7b0af"
        assert out.results == [
            {
                "index": 1,
                "relevance_score": 0.8304247466067356,
                "document": {
                    "text": "苹果派的制作步骤包括准备面团、切苹果、调制馅料、组装和烘烤。"
                },
            },
            {
                "index": 3,
                "relevance_score": 0.7142660211908354,
                "document": {"text": "制作苹果派时，预先煮软苹果可以缩短烘烤时间。"},
            },
        ]
        assert out.meta == {
            "billed_units": {"total_tokens": 192},
            "tokens": {"input_tokens": 192},
        }

    def test_response_string_document_normalized(self):
        # Defensive path: if a future API revision returns a bare string,
        # normalize to {"text": ...} so downstream code stays consistent.
        body = {
            "results": [{"index": 0, "relevance_score": 0.9, "document": "hello"}],
            "model": "qwen3-rerank",
            "usage": {"total_tokens": 5},
        }
        out = self.config.transform_rerank_response(
            model="qwen3-rerank",
            raw_response=self._resp(body),
            model_response=RerankResponse(),
            logging_obj=self.logging,
        )
        assert out.results[0]["document"] == {"text": "hello"}

    def test_missing_id_generates_uuid(self):
        body = {"results": [{"index": 0, "relevance_score": 0.5}], "usage": {}}
        out = self.config.transform_rerank_response(
            model="qwen3-rerank",
            raw_response=self._resp(body),
            model_response=RerankResponse(),
            logging_obj=self.logging,
        )
        assert out.id is not None and len(out.id) > 0

    @pytest.mark.parametrize("model", ["qwen3-rerank", "qwen3.7-text-rerank"])
    def test_error_envelope_raises(self, model):
        body = {
            "code": "InvalidApiKey",
            "message": "Invalid API-key provided.",
            "request_id": "fb53",
        }
        with pytest.raises(DashScopeError) as exc_info:
            get_dashscope_family_rerank_config("dashscope", model).transform_rerank_response(
                model=model,
                raw_response=self._resp(body, status_code=401),
                model_response=RerankResponse(),
                logging_obj=self.logging,
            )
        assert "Invalid API-key provided." in str(exc_info.value)

    @pytest.mark.parametrize("model", ["qwen3-rerank", "qwen3.7-text-rerank"])
    def test_non_json_response_raises(self, model):
        bad = httpx.Response(
            status_code=500,
            content=b"<html>bad gateway</html>",
            request=httpx.Request("POST", "https://example.com"),
        )
        with pytest.raises(DashScopeError):
            get_dashscope_family_rerank_config("dashscope", model).transform_rerank_response(
                model=model,
                raw_response=bad,
                model_response=RerankResponse(),
                logging_obj=self.logging,
            )

    def test_get_error_class(self):
        err = self.config.get_error_class(
            error_message="boom", status_code=500, headers={}
        )
        assert isinstance(err, DashScopeError)
        assert err.status_code == 500


class TestProviderConfigManagerDispatch:
    def test_dashscope_returns_rerank_config(self):
        import litellm
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_rerank_config(
            model="qwen3-rerank",
            provider=litellm.LlmProviders.DASHSCOPE,
            api_base=None,
            present_version_params=[],
        )
        assert isinstance(cfg, DashScopeRerankConfig)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("return_documents", [False, True])
@pytest.mark.parametrize(
    "provider,host",
    [
        ("dashscope", "dashscope.aliyuncs.com"),
        ("qwencloud", "dashscope-intl.aliyuncs.com"),
        ("qwen_ai_platform", "dashscope.aliyuncs.com"),
    ],
)
async def test_qwen37_rerank_public_call(
    is_async, return_documents, provider, host, respx_mock: respx.MockRouter, monkeypatch
):
    import litellm

    monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_BASE_RERANK", raising=False)
    monkeypatch.delenv(f"{provider.upper()}_API_BASE", raising=False)
    monkeypatch.delenv(f"{provider.upper()}_API_BASE_RERANK", raising=False)
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "fake-brand-key")
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    route = respx_mock.post(f"https://{host}/api/v1/services/rerank/text-rerank/text-rerank")
    results = [{"index": 1, "relevance_score": 0.88, **({"document": {"text": "answer"}} if return_documents else {})}]
    route.respond(
        200,
        json={
            "output": {"results": results},
            "usage": {"prompt_tokens": 237, "total_tokens": 261, "details": {"provider_metadata": True}},
            "request_id": "qwen37-request-id",
        },
    )
    kwargs = {
        "model": f"{provider}/qwen3.7-text-rerank",
        "query": "question",
        "documents": ["unrelated", "answer"],
        "top_n": 1,
        "return_documents": return_documents,
        "instruction": "Retrieve semantically similar text.",
    }

    response = await litellm.arerank(**kwargs) if is_async else litellm.rerank(**kwargs)

    assert json.loads(route.calls[0].request.content) == {
        "model": "qwen3.7-text-rerank",
        "input": {"query": "question", "documents": ["unrelated", "answer"]},
        "parameters": {
            "top_n": 1,
            "return_documents": return_documents,
            "instruct": "Retrieve semantically similar text.",
        },
    }
    assert route.calls[0].request.headers["authorization"] == "Bearer fake-brand-key"
    assert response.id == "qwen37-request-id"
    assert response.results == results
    assert response.meta == {"billed_units": {"total_tokens": 261}, "tokens": {"input_tokens": 237}}


@pytest.mark.parametrize(
    "api_base",
    [
        "https://proxy.example/api/v1",
        "https://proxy.example/api/v1/services/rerank/text-rerank/text-rerank/",
    ],
)
def test_qwen37_rerank_custom_url(api_base):
    assert get_dashscope_family_rerank_config("dashscope", "qwen3.7-text-rerank").get_complete_url(
        api_base, "qwen3.7-text-rerank"
    ) == ("https://proxy.example/api/v1/services/rerank/text-rerank/text-rerank")


@pytest.mark.parametrize(
    "provider,host",
    [("dashscope", "dashscope-intl.aliyuncs.com"), ("qwencloud", "dashscope.aliyuncs.com")],
)
def test_qwen37_rerank_explicit_region(provider, host):
    config = get_dashscope_family_rerank_config(provider, "qwen3.7-text-rerank")
    assert config.get_complete_url(f"https://{host}/compatible-mode/v1", "qwen3.7-text-rerank") == (
        f"https://{host}/api/v1/services/rerank/text-rerank/text-rerank"
    )


def test_qwen37_rerank_response_logging():
    config = get_dashscope_family_rerank_config("dashscope", "qwen3.7-text-rerank")
    logging = MagicMock()
    request = {
        "model": "qwen3.7-text-rerank",
        "input": {"query": "question", "documents": ["answer"]},
        "parameters": {},
    }
    payload = {"request_id": "request-id", "output": {"results": [{"index": 0, "relevance_score": 0.88}]}}

    response = config.transform_rerank_response(
        model="qwen3.7-text-rerank",
        raw_response=httpx.Response(200, json=payload),
        model_response=RerankResponse(),
        logging_obj=logging,
        request_data=request,
    )

    logging.post_call.assert_called_once_with(
        input="question", api_key=None, additional_args={"complete_input_dict": request}, original_response=payload
    )
    assert response.id == "request-id"
    assert response.results == [{"index": 0, "relevance_score": 0.88}]


@pytest.mark.parametrize("provider", ["dashscope", "qwencloud", "qwen_ai_platform"])
def test_qwen37_rerank_environment_url(provider, respx_mock: respx.MockRouter, monkeypatch):
    import litellm

    monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)
    monkeypatch.delenv(f"{provider.upper()}_API_BASE", raising=False)
    monkeypatch.setenv(f"{provider.upper()}_API_BASE_RERANK", "https://proxy.example/api/v1")
    route = respx_mock.post("https://proxy.example/api/v1/services/rerank/text-rerank/text-rerank")
    route.respond(
        200,
        json={
            "output": {"results": [{"index": 0, "relevance_score": 0.88}]},
            "request_id": "all-results",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        },
    )

    response = litellm.rerank(
        model=f"{provider}/qwen3.7-text-rerank",
        query="question",
        documents=["answer"],
        return_documents=None,
        api_key="fake-dashscope-key",
    )

    assert json.loads(route.calls[0].request.content) == {
        "model": "qwen3.7-text-rerank",
        "input": {"query": "question", "documents": ["answer"]},
        "parameters": {},
    }
    assert response.results == [{"index": 0, "relevance_score": 0.88}]
    assert response.meta["tokens"]["input_tokens"] == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_qwen37_rerank_preserves_provider_error(is_async, respx_mock: respx.MockRouter, monkeypatch):
    import litellm

    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    route = respx_mock.post("https://proxy.example/api/v1/services/rerank/text-rerank/text-rerank")
    route.respond(
        400,
        json={"code": "InvalidParameter", "message": "documents must not be empty", "request_id": "invalid-documents"},
    )
    kwargs = {
        "model": "dashscope/qwen3.7-text-rerank",
        "query": "question",
        "documents": [],
        "api_key": "fake-dashscope-key",
        "api_base": "https://proxy.example/api/v1",
    }

    with pytest.raises(litellm.BadRequestError, match="documents must not be empty") as error:
        await litellm.arerank(**kwargs) if is_async else litellm.rerank(**kwargs)

    assert error.value.status_code == 400
    assert "DashscopeException" in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("provider", ["dashscope", "qwencloud", "qwen_ai_platform"])
@pytest.mark.parametrize("base_case", ["rerank_override", "explicit", "explicit_default", "general_only"])
async def test_native_rerank_base_precedence(provider, is_async, base_case, respx_mock, monkeypatch):
    import litellm

    prefix = provider.upper()
    default_host = "dashscope-intl.aliyuncs.com" if provider == "qwencloud" else "dashscope.aliyuncs.com"
    monkeypatch.setenv(f"{prefix}_API_BASE", "https://chat.example/compatible-mode/v1")
    monkeypatch.delenv(f"{prefix}_API_BASE_RERANK", raising=False)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    if base_case != "general_only":
        monkeypatch.setenv(f"{prefix}_API_BASE_RERANK", "https://rerank.example/api/v1")
    explicit_base = {
        "rerank_override": None,
        "explicit": "https://explicit.example/api/v1",
        "explicit_default": f"https://{default_host}/compatible-mode/v1",
        "general_only": None,
    }[base_case]
    expected_host = {
        "rerank_override": "rerank.example",
        "explicit": "explicit.example",
        "explicit_default": default_host,
        "general_only": "chat.example",
    }[base_case]
    route = respx_mock.post(f"https://{expected_host}/api/v1/services/rerank/text-rerank/text-rerank")
    route.respond(
        200, json={"request_id": "base-precedence", "output": {"results": [{"index": 0, "relevance_score": 0.9}]}}
    )
    kwargs = {
        "model": f"{provider}/qwen3.7-text-rerank",
        "query": "question",
        "documents": ["answer"],
        "api_key": "fake-review-key",
        "api_base": explicit_base,
    }

    response = await litellm.arerank(**kwargs) if is_async else litellm.rerank(**kwargs)

    assert json.loads(route.calls[0].request.content)["input"] == {"query": "question", "documents": ["answer"]}
    assert response.id == "base-precedence"
    assert response.results == [{"index": 0, "relevance_score": 0.9}]


@pytest.mark.parametrize("provider", ["dashscope", "qwencloud", "qwen_ai_platform"])
@pytest.mark.parametrize("rerank_api", ["native", "compatible"])
def test_rerank_protocol_uses_runtime_model_metadata(provider, rerank_api, respx_mock, monkeypatch):
    import litellm

    model = "custom-rerank" if rerank_api == "native" else "qwen3.7-text-rerank"
    monkeypatch.setitem(
        litellm.model_cost,
        model if provider == "dashscope" and rerank_api == "native" else f"{provider}/{model}",
        {
            "litellm_provider": provider,
            "mode": "rerank",
            "provider_specific_entry": {"rerank_api": rerank_api},
        },
    )
    results = [{"index": 0, "relevance_score": 0.9}]
    url = (
        "https://proxy.example/api/v1/services/rerank/text-rerank/text-rerank"
        if rerank_api == "native"
        else "https://proxy.example/api/v1/reranks"
    )
    route = respx_mock.post(url)
    route.respond(
        200,
        json={"request_id": "metadata", "output": {"results": results}}
        if rerank_api == "native"
        else {"id": "metadata", "results": results},
    )

    response = litellm.rerank(
        model=f"{provider}/{model}",
        query="question",
        documents=["answer"],
        api_key="fake-key",
        api_base="https://proxy.example/api/v1",
        return_documents=None,
    )

    assert json.loads(route.calls[0].request.content) == (
        {"model": model, "input": {"query": "question", "documents": ["answer"]}, "parameters": {}}
        if rerank_api == "native"
        else {"model": model, "query": "question", "documents": ["answer"]}
    )
    assert response.id == "metadata"
    assert response.results == results


@pytest.mark.parametrize("provider", ["dashscope", "qwencloud", "qwen_ai_platform"])
def test_rerank_uses_bundled_metadata_when_remote_map_lacks_model(provider, respx_mock, monkeypatch):
    import litellm

    for prefix in ("dashscope", "qwencloud", "qwen_ai_platform"):
        monkeypatch.delitem(litellm.model_cost, f"{prefix}/qwen3.7-text-rerank", raising=False)
    route = respx_mock.post("https://proxy.example/api/v1/services/rerank/text-rerank/text-rerank")
    route.respond(200, json={"request_id": "bundled", "output": {"results": [{"index": 0, "relevance_score": 0.9}]}})

    response = litellm.rerank(
        model=f"{provider}/qwen3.7-text-rerank",
        query="question",
        documents=["answer"],
        api_key="fake-key",
        api_base="https://proxy.example/api/v1",
    )

    assert json.loads(route.calls[0].request.content)["input"] == {"query": "question", "documents": ["answer"]}
    assert response.id == "bundled"
    assert response.results == [{"index": 0, "relevance_score": 0.9}]
