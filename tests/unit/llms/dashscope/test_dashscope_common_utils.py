"""
Unit tests for DashScope's protocol-specific base URL derivation.
"""

from typing import Final

import pytest

from litellm.llms.dashscope.common_utils import (
    DASHSCOPE_DEFAULT_ORIGIN,
    resolve_dashscope_anthropic_api_base,
    resolve_dashscope_endpoints,
    resolve_dashscope_family_rerank_api_base,
    resolve_dashscope_openai_api_base,
)

WORKSPACE: Final = "https://llm-abc123.cn-beijing.maas.aliyuncs.com"


@pytest.mark.parametrize(
    "api_base",
    [
        WORKSPACE,
        f"{WORKSPACE}/",
        f"{WORKSPACE}/compatible-mode/v1",
        f"{WORKSPACE}/compatible-mode/v1/chat/completions",
        f"{WORKSPACE}/apps/anthropic",
        f"{WORKSPACE}/apps/anthropic/v1/messages",
        f"{WORKSPACE}/api/v1",
        f"{WORKSPACE}/api/v1/services/aigc/video-generation/video-synthesis",
    ],
)
def test_workspace_base_derives_every_protocol(api_base: str):
    endpoints: Final = resolve_dashscope_endpoints(api_base)

    assert endpoints is not None
    assert endpoints.origin == WORKSPACE
    assert endpoints.openai == f"{WORKSPACE}/compatible-mode/v1"
    assert endpoints.anthropic == f"{WORKSPACE}/apps/anthropic"
    assert endpoints.native == f"{WORKSPACE}/api/v1"


@pytest.mark.parametrize(
    "origin",
    [
        "https://dashscope.aliyuncs.com",
        "https://dashscope-intl.aliyuncs.com",
        "https://dashscope-us.aliyuncs.com",
    ],
)
@pytest.mark.parametrize("path", ["", "/", "/compatible-mode/v1"])
def test_standard_hosts_derive_every_protocol(origin: str, path: str):
    endpoints: Final = resolve_dashscope_endpoints(f"{origin}{path}")

    assert endpoints is not None
    assert endpoints.openai == f"{origin}/compatible-mode/v1"
    assert endpoints.anthropic == f"{origin}/apps/anthropic"
    assert endpoints.native == f"{origin}/api/v1"


@pytest.mark.parametrize(
    "api_base",
    [
        None,
        "",
        "not-a-url",
        "https://gateway.example.com",
        "https://gateway.example.com/compatible-mode/v1",
        f"{WORKSPACE}/custom-path",
        f"{WORKSPACE}/compatible-mode/v1?api-version=1",
        "https://trial.cn-beijing.maas.aliyuncs.com",
        "https://token-plan.cn-beijing.maas.aliyuncs.com",
    ],
)
def test_unrecognized_bases_are_not_rewritten(api_base: str | None):
    assert resolve_dashscope_endpoints(api_base) is None


def test_openai_base_derives_from_the_domain(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)

    assert resolve_dashscope_openai_api_base(WORKSPACE) == f"{WORKSPACE}/compatible-mode/v1"
    assert resolve_dashscope_openai_api_base(f"{WORKSPACE}/apps/anthropic") == f"{WORKSPACE}/compatible-mode/v1"
    assert (
        resolve_dashscope_openai_api_base(f"{WORKSPACE}/compatible-api/v1/reranks") == f"{WORKSPACE}/compatible-mode/v1"
    )


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        f"{WORKSPACE}/api/v1/services/aigc/video-generation/video-synthesis",
    ],
)
def test_openai_base_keeps_a_native_endpoint_url(monkeypatch: pytest.MonkeyPatch, endpoint_url: str):
    monkeypatch.setenv("DASHSCOPE_API_BASE", endpoint_url)

    assert resolve_dashscope_openai_api_base(None) == endpoint_url
    assert resolve_dashscope_openai_api_base(endpoint_url) == endpoint_url
    assert resolve_dashscope_openai_api_base(f"{WORKSPACE}/api/v1") == f"{WORKSPACE}/compatible-mode/v1"


def test_openai_base_keeps_unrecognized_hosts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)

    assert resolve_dashscope_openai_api_base("https://gateway.example.com/v1") == "https://gateway.example.com/v1"


def test_openai_base_defaults_to_dashscope(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)

    assert resolve_dashscope_openai_api_base(None) == f"{DASHSCOPE_DEFAULT_ORIGIN}/compatible-mode/v1"


def test_anthropic_base_derives_from_the_domain(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)

    assert resolve_dashscope_anthropic_api_base(WORKSPACE) == f"{WORKSPACE}/apps/anthropic"
    assert resolve_dashscope_anthropic_api_base(f"{WORKSPACE}/compatible-mode/v1") == f"{WORKSPACE}/apps/anthropic"


def test_anthropic_base_keeps_unrecognized_hosts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)

    assert (
        resolve_dashscope_anthropic_api_base("https://gateway.example.com/anthropic")
        == "https://gateway.example.com/anthropic"
    )


def test_rerank_base_derives_from_the_domain(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DASHSCOPE_API_BASE_RERANK", raising=False)

    assert (
        resolve_dashscope_family_rerank_api_base(WORKSPACE, "DASHSCOPE_API_BASE_RERANK", "https://default/reranks")
        == f"{WORKSPACE}/compatible-api/v1/reranks"
    )


def test_rerank_base_keeps_unrecognized_hosts():
    gateway: Final = "https://gateway.example.com/v1"

    assert (
        resolve_dashscope_family_rerank_api_base(gateway, "DASHSCOPE_API_BASE_RERANK", "https://default/reranks")
        == gateway
    )


def test_rerank_base_uses_the_configured_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DASHSCOPE_API_BASE_RERANK", raising=False)

    assert (
        resolve_dashscope_family_rerank_api_base(None, "DASHSCOPE_API_BASE_RERANK", "https://default/reranks")
        == "https://default/reranks"
    )


@pytest.mark.parametrize(
    "origin",
    ["https://cn-hongkong.dashscope.aliyuncs.com", "https://token-plan.cn-beijing.maas.aliyuncs.com"],
)
def test_rerank_keeps_remapping_chat_bases_on_other_aliyun_hosts(monkeypatch: pytest.MonkeyPatch, origin: str):
    monkeypatch.delenv("DASHSCOPE_API_BASE_RERANK", raising=False)

    assert (
        resolve_dashscope_family_rerank_api_base(
            f"{origin}/compatible-mode/v1", "DASHSCOPE_API_BASE_RERANK", "https://default/reranks"
        )
        == f"{origin}/compatible-api/v1/reranks"
    )
