import logging
from collections.abc import Iterator, Mapping

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
from litellm.proxy.policy_engine.policy_registry import get_policy_registry
from litellm.proxy.policy_engine.response_retrieval import attach_post_call_pipelines_to_retrieval
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.router import Deployment, LiteLLM_Params

GOVERNED_MODEL_GROUP = "gpt-5.4-mini"
GOVERNED_MODEL_ID = "deployment-governed"
UNGOVERNED_MODEL_GROUP = "gpt-4.1-mini"
UNGOVERNED_MODEL_ID = "deployment-ungoverned"
WILDCARD_MODEL_GROUP = "openai/*"
WILDCARD_MODEL_ID = "deployment-wildcard"


class FakeRouter:
    def __init__(self, deployments: dict[str, Deployment], model_group_alias: dict[str, object] | None = None):
        self._deployments = deployments
        self.model_group_alias = model_group_alias or {}

    def get_deployment(self, model_id: str) -> Deployment | None:
        return self._deployments.get(model_id)


def _deployment(model_group: str, model_id: str) -> Deployment:
    return Deployment(
        model_name=model_group,
        litellm_params=LiteLLM_Params(model=f"openai/{model_group}"),
        model_info={"id": model_id},
    )


def _router(model_group_alias: dict[str, object] | None = None) -> FakeRouter:
    return FakeRouter(
        {
            GOVERNED_MODEL_ID: _deployment(GOVERNED_MODEL_GROUP, GOVERNED_MODEL_ID),
            UNGOVERNED_MODEL_ID: _deployment(UNGOVERNED_MODEL_GROUP, UNGOVERNED_MODEL_ID),
            WILDCARD_MODEL_ID: _deployment(WILDCARD_MODEL_GROUP, WILDCARD_MODEL_ID),
        },
        model_group_alias,
    )


def _encoded_response_id(model_id: str) -> str:
    return ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="openai", model_id=model_id, response_id="resp_upstream"
    )


def _pipeline_policy(guardrail: str, mode: str = "post_call") -> dict[str, object]:
    return {
        "guardrails": {"add": [guardrail]},
        "pipeline": {"mode": mode, "steps": [{"guardrail": guardrail, "on_pass": "allow", "on_fail": "block"}]},
    }


@pytest.fixture
def policy_engine() -> Iterator[None]:
    policy_registry = get_policy_registry()
    attachment_registry = get_attachment_registry()
    policy_registry.load_policies(
        {
            "response-governance": _pipeline_policy("output-word-filter"),
            "input-governance": _pipeline_policy("input-word-filter", mode="pre_call"),
            "team-governance": _pipeline_policy("team-word-filter"),
            "tag-governance": _pipeline_policy("tag-word-filter"),
        }
    )
    attachment_registry.load_attachments(
        [
            {"policy": "response-governance", "models": [GOVERNED_MODEL_GROUP]},
            {"policy": "input-governance", "models": [GOVERNED_MODEL_GROUP]},
            {"policy": "team-governance", "teams": ["governed-team"]},
            {"policy": "tag-governance", "tags": ["governed"]},
        ]
    )
    yield
    policy_registry.clear()
    attachment_registry.clear()


def _retrieval_data(model_id: str) -> dict[str, object]:
    return {"response_id": _encoded_response_id(model_id), "litellm_metadata": {}}


def _attached_pipelines(data: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    bucket = data["litellm_metadata"]
    assert isinstance(bucket, dict)
    return tuple(
        (policy_name, ",".join(step.guardrail for step in pipeline.steps))
        for policy_name, pipeline in bucket["_guardrail_pipelines"]
    )


def test_attaches_model_scoped_post_call_pipeline_to_retrieval(policy_engine: None) -> None:
    data = _retrieval_data(GOVERNED_MODEL_ID)

    attach_post_call_pipelines_to_retrieval(data=data, user_api_key_dict=UserAPIKeyAuth(), llm_router=_router())

    assert _attached_pipelines(data) == (("response-governance", "output-word-filter"),)
    assert data["litellm_metadata"]["_pipeline_managed_guardrails"] == frozenset({"output-word-filter"})
    assert data["litellm_metadata"]["applied_policies"] == ["response-governance"]
    assert data["litellm_metadata"]["applied_guardrails"] == ["output-word-filter"]
    assert data["litellm_metadata"]["policy_sources"] == {"response-governance": "model:gpt-5.4-mini"}
    assert "model" not in data
    assert "guardrails" not in data["litellm_metadata"]


def test_key_and_team_context_also_governs_retrieval(policy_engine: None) -> None:
    data = _retrieval_data(UNGOVERNED_MODEL_ID)

    attach_post_call_pipelines_to_retrieval(
        data=data, user_api_key_dict=UserAPIKeyAuth(team_alias="governed-team"), llm_router=_router()
    )

    assert _attached_pipelines(data) == (("team-governance", "team-word-filter"),)


def test_tag_attached_policy_is_not_re_matched_when_the_retrieval_carries_no_tag(policy_engine: None) -> None:
    data = _retrieval_data(GOVERNED_MODEL_ID)

    attach_post_call_pipelines_to_retrieval(data=data, user_api_key_dict=UserAPIKeyAuth(), llm_router=_router())

    assert _attached_pipelines(data) == (("response-governance", "output-word-filter"),)


def test_tag_attached_policy_governs_a_retrieval_whose_metadata_carries_the_tag(policy_engine: None) -> None:
    data: dict[str, object] = {
        "response_id": _encoded_response_id(UNGOVERNED_MODEL_ID),
        "litellm_metadata": {"tags": ["governed"]},
    }

    attach_post_call_pipelines_to_retrieval(data=data, user_api_key_dict=UserAPIKeyAuth(), llm_router=_router())

    assert _attached_pipelines(data) == (("tag-governance", "tag-word-filter"),)
    assert data["litellm_metadata"]["policy_sources"] == {"tag-governance": "tag:governed"}


def test_retrieval_of_an_ungoverned_model_attaches_nothing(policy_engine: None) -> None:
    data = _retrieval_data(UNGOVERNED_MODEL_ID)

    attach_post_call_pipelines_to_retrieval(data=data, user_api_key_dict=UserAPIKeyAuth(), llm_router=_router())

    assert data == _retrieval_data(UNGOVERNED_MODEL_ID)


def test_already_attached_policy_is_not_attached_twice(policy_engine: None) -> None:
    data = _retrieval_data(GOVERNED_MODEL_ID)
    router = _router()
    attach_post_call_pipelines_to_retrieval(data=data, user_api_key_dict=UserAPIKeyAuth(), llm_router=router)

    attach_post_call_pipelines_to_retrieval(data=data, user_api_key_dict=UserAPIKeyAuth(), llm_router=router)

    assert _attached_pipelines(data) == (("response-governance", "output-word-filter"),)
    assert data["litellm_metadata"]["applied_policies"] == ["response-governance"]


def _hidden_submit_model_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "the model name it was submitted as" in record.getMessage()
    ]


def test_wildcard_deployment_attaches_nothing_for_the_submitted_model_and_warns(
    policy_engine: None, caplog: pytest.LogCaptureFixture
) -> None:
    data = _retrieval_data(WILDCARD_MODEL_ID)

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        attach_post_call_pipelines_to_retrieval(data=data, user_api_key_dict=UserAPIKeyAuth(), llm_router=_router())

    assert data == _retrieval_data(WILDCARD_MODEL_ID)
    assert [
        "as model group openai/* (a wildcard deployment)" in message
        for message in _hidden_submit_model_warnings(caplog)
    ] == [True]


def test_aliased_model_group_still_attaches_its_own_policies_and_warns(
    policy_engine: None, caplog: pytest.LogCaptureFixture
) -> None:
    data = _retrieval_data(GOVERNED_MODEL_ID)
    router = _router({"gpt-mini": GOVERNED_MODEL_GROUP, "gpt-hidden": {"model": GOVERNED_MODEL_GROUP, "hidden": True}})

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        attach_post_call_pipelines_to_retrieval(data=data, user_api_key_dict=UserAPIKeyAuth(), llm_router=router)

    assert _attached_pipelines(data) == (("response-governance", "output-word-filter"),)
    assert [
        "(the target of model_group_alias gpt-mini, gpt-hidden)" in message
        for message in _hidden_submit_model_warnings(caplog)
    ] == [True]


def test_plain_model_group_retrieval_does_not_warn_about_the_submitted_model(
    policy_engine: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        attach_post_call_pipelines_to_retrieval(
            data=_retrieval_data(GOVERNED_MODEL_ID),
            user_api_key_dict=UserAPIKeyAuth(),
            llm_router=_router({"other-alias": UNGOVERNED_MODEL_GROUP}),
        )

    assert _hidden_submit_model_warnings(caplog) == []


def _ungoverned_retrieval_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "retrieved without its post_call policy pipelines" in record.getMessage()
    ]


@pytest.mark.parametrize(
    ("response_id", "reason"),
    [
        ("resp_plain_upstream_id", "response id names no deployment"),
        (_encoded_response_id("deployment-missing-from-router"), "deployment no longer in the router"),
        (None, "response id names no deployment"),
    ],
)
def test_unresolvable_response_id_attaches_nothing_and_warns(
    policy_engine: None, caplog: pytest.LogCaptureFixture, response_id: str, reason: str
) -> None:
    data = {"response_id": response_id, "litellm_metadata": {}}

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        attach_post_call_pipelines_to_retrieval(data=data, user_api_key_dict=UserAPIKeyAuth(), llm_router=_router())

    assert data == {"response_id": response_id, "litellm_metadata": {}}
    assert [message.endswith(f"({reason})") for message in _ungoverned_retrieval_warnings(caplog)] == [True]


def test_without_a_router_attaches_nothing_and_warns(policy_engine: None, caplog: pytest.LogCaptureFixture) -> None:
    data = _retrieval_data(GOVERNED_MODEL_ID)

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        attach_post_call_pipelines_to_retrieval(data=data, user_api_key_dict=UserAPIKeyAuth(), llm_router=None)

    assert data == _retrieval_data(GOVERNED_MODEL_ID)
    assert [message.endswith("(no router)") for message in _ungoverned_retrieval_warnings(caplog)] == [True]


def test_without_policy_engine_attaches_nothing_quietly(caplog: pytest.LogCaptureFixture) -> None:
    get_policy_registry().clear()
    data = _retrieval_data(GOVERNED_MODEL_ID)

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        attach_post_call_pipelines_to_retrieval(data=data, user_api_key_dict=UserAPIKeyAuth(), llm_router=None)

    assert data == _retrieval_data(GOVERNED_MODEL_ID)
    assert _ungoverned_retrieval_warnings(caplog) == []
