import httpx
import pytest
import respx

from litellm.integrations.gitlab import prompt_initializer_registry
from litellm.types.prompts.init_prompts import PromptLiteLLMParams, PromptSpec

_GITLAB_CONFIG = {
    "project": "group/repo",
    "access_token": "glpat-test",
    "base_url": "https://gitlab.example.test/api/v4",
    "prompts_path": "prompts",
}


def _prompt_spec(litellm_params: PromptLiteLLMParams) -> PromptSpec:
    return PromptSpec(prompt_id="chat/greet", litellm_params=litellm_params)


def test_gitlab_initializer_loads_the_prompt_file_at_the_per_prompt_git_ref(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(host="gitlab.example.test").mock(
        return_value=httpx.Response(
            200, text="---\nmodel: gpt-4o\n---\nHello {{name}}", headers={"content-type": "text/plain"}
        )
    )
    litellm_params = PromptLiteLLMParams(prompt_integration="gitlab", gitlab_config=_GITLAB_CONFIG, git_ref="v1.2.0")

    manager = prompt_initializer_registry["gitlab"](litellm_params, _prompt_spec(litellm_params))
    fetches_by_initializer = route.call_count

    assert manager.get_prompt_template("chat/greet", {"name": "Ada"}) == (
        "Hello Ada",
        {"model": "gpt-4o", "temperature": None, "max_tokens": None},
    )
    assert (fetches_by_initializer, route.call_count) == (1, 1)
    request = route.calls.last.request
    assert str(request.url) == (
        "https://gitlab.example.test/api/v4/projects/group%2Frepo/repository/files/prompts%2Fchat%2Fgreet.prompt/raw"
        "?ref=v1.2.0"
    )
    assert request.headers["Private-Token"] == "glpat-test"


@pytest.mark.parametrize(
    "integration_params",
    [
        pytest.param({}, id="config-absent"),
        pytest.param({"gitlab_config": None}, id="config-none"),
        pytest.param({"gitlab_config": {}}, id="config-empty"),
    ],
)
def test_gitlab_initializer_rejects_prompt_without_gitlab_config(integration_params: dict[str, object]) -> None:
    litellm_params = PromptLiteLLMParams(prompt_integration="gitlab", git_ref="v1.2.0", **integration_params)

    with pytest.raises(ValueError, match="gitlab_config is required for gitlab prompt integration"):
        prompt_initializer_registry["gitlab"](litellm_params, _prompt_spec(litellm_params))
