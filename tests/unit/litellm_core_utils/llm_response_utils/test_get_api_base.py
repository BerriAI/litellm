from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.llm_response_utils import get_api_base as get_api_base_module
from litellm.llms.chatgpt.common_utils import CHATGPT_API_BASE
from litellm.llms.github_copilot.common_utils import DEFAULT_GITHUB_COPILOT_API_BASE
from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession
from litellm.types.router import GenericLiteLLMParams


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

    def test_copilot_uses_the_user_endpoint_for_an_enterprise_api_base(self, isolated_token_dirs, resolution_lookups):
        token_dir = isolated_token_dirs / "github_copilot"
        token_dir.mkdir()
        token = f"gho_{isolated_token_dirs.name}"
        (token_dir / "access-token").write_text(token)
        with respx.mock(assert_all_called=True) as router:
            route = router.get("https://api.github.com/copilot_internal/user").respond(
                json={"endpoints": {"api": "https://api.enterprise.githubcopilot.com"}}
            )
            api_base = litellm.get_api_base(model="github_copilot/gpt-4o", optional_params={})

        assert resolution_lookups == []
        assert api_base == "https://api.enterprise.githubcopilot.com"
        assert route.calls.last.request.headers["authorization"] == f"token {token}"

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
        ("gemini/gemini-2.5-pro", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"),
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


@pytest.mark.usefixtures("isolated_token_dirs")
def test_github_app_deployment_reports_the_copilot_host_it_calls_over_a_configured_api_base():
    model: Final = "github_copilot/gpt-5.5"
    params: Final = {
        "api_base": "https://admin-set.internal/v1",
        "github_copilot_auth_type": "github_app",
        "github_copilot_app_id": "test-client",
        "github_copilot_app_installation_id": 123,
        "github_copilot_app_private_key_ref": "os.environ/TEST_APP_KEY",
    }
    called_api_base: Final = litellm.get_llm_provider(
        model=model, api_base=params["api_base"], litellm_params=GenericLiteLLMParams(**params)
    )[3]

    assert called_api_base != params["api_base"]
    assert litellm.get_api_base(model=model, optional_params=params) == called_api_base


@pytest.mark.usefixtures("isolated_token_dirs")
def test_per_user_session_still_reports_the_configured_api_base():
    params: Final = {
        "api_base": "https://admin-set.internal/v1",
        "github_copilot_user_session": GithubCopilotUserSession("user-token", "https://api.business.githubcopilot.com"),
    }

    assert litellm.get_api_base(model="github_copilot/gpt-5.5", optional_params=params) == params["api_base"]
