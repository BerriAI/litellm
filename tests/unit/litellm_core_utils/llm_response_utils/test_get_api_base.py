import json
from collections.abc import Mapping

import pytest

import litellm
from litellm.litellm_core_utils.llm_response_utils import get_api_base as get_api_base_module
from litellm.llms.chatgpt.common_utils import CHATGPT_API_BASE
from litellm.llms.github_copilot.common_utils import DEFAULT_GITHUB_COPILOT_API_BASE


@pytest.fixture
def isolated_token_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path / "github_copilot"))
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path / "chatgpt"))
    monkeypatch.delenv("GITHUB_COPILOT_API_BASE", raising=False)
    monkeypatch.delenv("CHATGPT_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_CHATGPT_API_BASE", raising=False)
    return tmp_path


@pytest.fixture
def resolution_lookups(monkeypatch):
    lookups: list = []

    def _record(*args, **kwargs):
        lookups.append((args, kwargs))
        raise RuntimeError("provider resolution must not run for an authenticating provider")

    monkeypatch.setattr(get_api_base_module, "get_llm_provider", _record)
    return lookups


class TestDeclaredAuthenticatingProvider:
    """get_llm_provider runs the OAuth device flow for github_copilot and chatgpt, and get_api_base
    runs on every response's hidden params and on every mapped exception, so it must answer from
    the declaration without resolving. The recorder appends before raising, and get_api_base
    swallows resolver errors, so an empty list proves the lookup never ran."""

    @pytest.mark.parametrize(
        "model, custom_llm_provider, expected",
        [
            ("github_copilot/gpt-4o", None, DEFAULT_GITHUB_COPILOT_API_BASE),
            ("gpt-4o", "github_copilot", DEFAULT_GITHUB_COPILOT_API_BASE),
            ("chatgpt/gpt-5", None, CHATGPT_API_BASE),
            ("gpt-5", "chatgpt", CHATGPT_API_BASE),
        ],
    )
    def test_answers_without_resolving(
        self, model, custom_llm_provider, expected, isolated_token_dirs, resolution_lookups
    ):
        api_base = litellm.get_api_base(model=model, optional_params={"custom_llm_provider": custom_llm_provider})

        assert resolution_lookups == []
        assert api_base == expected

    def test_copilot_keeps_the_enterprise_endpoint_from_disk(self, isolated_token_dirs, resolution_lookups):
        token_dir = isolated_token_dirs / "github_copilot"
        token_dir.mkdir()
        (token_dir / "api-key.json").write_text(
            json.dumps({"endpoints": {"api": "https://api.enterprise.githubcopilot.com"}})
        )

        api_base = litellm.get_api_base(model="github_copilot/gpt-4o", optional_params={})

        assert resolution_lookups == []
        assert api_base == "https://api.enterprise.githubcopilot.com"

    def test_explicit_api_base_still_wins(self, isolated_token_dirs, resolution_lookups):
        api_base = litellm.get_api_base(
            model="github_copilot/gpt-4o", optional_params={"api_base": "https://copilot.example/v1"}
        )

        assert resolution_lookups == []
        assert api_base == "https://copilot.example/v1"

    def test_other_providers_still_resolve(self, isolated_token_dirs, resolution_lookups):
        litellm.get_api_base(model="openai/gpt-4o", optional_params={})

        assert len(resolution_lookups) == 1


@pytest.mark.parametrize(
    "model, expected",
    [
        (
            "gemini/gemini-2.5-pro",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
        ),
        ("openai/gpt-4o", "https://api.openai.com"),
    ],
)
def test_providers_with_a_fixed_base_still_get_it(model, expected, monkeypatch):
    for env in ("GEMINI_API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL"):
        monkeypatch.delenv(env, raising=False)

    assert litellm.get_api_base(model=model, optional_params={}) == expected


def test_base_url_alias_is_reported_as_the_api_base():
    api_base = litellm.get_api_base(
        model="groq/whisper-large-v3", optional_params={"base_url": "https://groq.gateway.internal/openai/v1"}
    )

    assert api_base == "https://groq.gateway.internal/openai/v1"
    assert (
        litellm.get_api_base(
            model="groq/whisper-large-v3",
            optional_params={"api_base": "https://explicit.internal/v1", "base_url": "https://alias.internal/v1"},
        )
        == "https://explicit.internal/v1"
    )
    assert (
        litellm.get_api_base(model="groq/whisper-large-v3", optional_params={"base_url": ""})
        == "https://api.groq.com/openai/v1"
    )


@pytest.mark.parametrize(
    ("model", "optional_params", "toggle", "expected"),
    (
        (
            "gemini/gemini-3-flash-preview",
            {"model_info": {"mode": "interactions"}},
            False,
            "https://generativelanguage.googleapis.com/v1beta/interactions",
        ),
        (
            "gemini/gemini-3-flash-preview",
            {"model_info": {"mode": "interactions"}, "stream": True},
            False,
            "https://generativelanguage.googleapis.com/v1beta/interactions?alt=sse",
        ),
        (
            "vertex_ai/gemini-3-flash-preview",
            {"vertex_project": "proj-1", "vertex_location": "global", "model_info": {"mode": "interactions"}},
            False,
            "https://aiplatform.googleapis.com/v1beta1/projects/proj-1/locations/global/interactions",
        ),
        (
            "vertex_ai/gemini-3-flash-preview",
            {"vertex_project": "proj-1", "vertex_location": "us-central1"},
            True,
            "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/proj-1/locations/us-central1/interactions",
        ),
        (
            "vertex_ai/gemini-3-flash-preview",
            {"vertex_project": "proj-1", "vertex_location": "us"},
            True,
            "https://aiplatform.us.rep.googleapis.com/v1beta1/projects/proj-1/locations/us/interactions",
        ),
        (
            "vertex_ai/gemini-3-flash-preview",
            {"vertex_project": "proj-1", "model_info": {"mode": "interactions"}},
            False,
            "https://aiplatform.googleapis.com/v1beta1/projects/proj-1/locations/global/interactions",
        ),
        (
            "vertex_ai/gemini-3-flash-preview",
            {"vertex_project": "proj-1", "vertex_location": "us-central1", "model_info": {"mode": "chat"}},
            True,
            "us-central1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-central1/publishers/google/models/gemini-3-flash-preview:generateContent",
        ),
    ),
)
def test_interactions_api_base_respects_vertex_location_hosts_and_deployment_mode(
    model: str,
    optional_params: Mapping[str, object],
    toggle: bool,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_ROUTE_ALL_CHAT_GEMINI_TO_INTERACTIONS", raising=False)
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", toggle)

    assert litellm.get_api_base(model=model, optional_params=dict(optional_params)) == expected
