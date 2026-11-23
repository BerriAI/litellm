from typing import Final

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.project import LiteLLM_ProjectTable
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.auth.grants import (
    end_user_budget_grants,
    end_user_grants,
    organization_grants,
    owner_grants,
    project_grants,
    stamp,
    with_end_user,
    with_owner_model_budget,
    with_project,
)
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth


def _organization(budget: LiteLLM_BudgetTable | None) -> LiteLLM_OrganizationTable:
    return LiteLLM_OrganizationTable(
        organization_id="org-1",
        organization_alias="Org One",
        budget_id="b-1",
        metadata={"tier": "gold"},
        models=[],
        created_by="admin",
        updated_by="admin",
        litellm_budget_table=budget,
    )


def test_a_project_grants_its_alias_and_metadata():
    project = LiteLLM_ProjectTable(project_id="p-1", project_alias="Project One", metadata={"tags": ["a"]})

    token = with_project(UserAPIKeyAuth(token="hashed", project_id="p-1"), project)

    assert (token.token, token.project_alias, token.project_metadata) == ("hashed", "Project One", {"tags": ["a"]})


def test_a_key_with_no_project_row_keeps_what_it_was_read_with():
    key = UserAPIKeyAuth(token="hashed", project_id="p-1", project_alias="from the key read")

    assert with_project(key, None).project_alias == "from the key read"


def test_an_organization_grants_its_own_values_and_its_budgets_limits():
    grants = organization_grants(_organization(LiteLLM_BudgetTable(max_budget=50.0, tpm_limit=1000, rpm_limit=10)))

    assert grants == {
        "organization_alias": "Org One",
        "organization_metadata": {"tier": "gold"},
        "organization_max_budget": 50.0,
        "organization_tpm_limit": 1000,
        "organization_rpm_limit": 10,
    }


def test_an_organization_without_a_budget_row_grants_no_limits():
    grants = organization_grants(_organization(None))

    assert (grants["organization_alias"], grants["organization_metadata"]) == ("Org One", {"tier": "gold"})
    assert (
        grants["organization_max_budget"],
        grants["organization_tpm_limit"],
        grants["organization_rpm_limit"],
    ) == (None, None, None)


def _owner() -> LiteLLM_UserTable:
    return LiteLLM_UserTable(
        user_id="u-1",
        user_email="owner@corp.invalid",
        tpm_limit=500,
        rpm_limit=5,
        spend=3.5,
        max_budget=20.0,
        model_max_budget={"model-a": {"max_budget": 1.0}},
    )


def test_an_owner_grants_its_limits_spend_budgets_and_email():
    assert owner_grants(_owner()) == {
        "user_tpm_limit": 500,
        "user_rpm_limit": 5,
        "user_email": "owner@corp.invalid",
        "user_spend": 3.5,
        "user_max_budget": 20.0,
        "user_model_max_budget": {"model-a": {"max_budget": 1.0}},
    }


def test_before_the_key_path_ends_a_key_takes_only_its_owners_model_budget():
    key = UserAPIKeyAuth(token="hashed", user_id="u-1")

    token = with_owner_model_budget(key, _owner())

    assert token.user_model_max_budget == {"model-a": {"max_budget": 1.0}}
    assert (token.user_tpm_limit, token.user_rpm_limit, token.user_email, token.user_max_budget) == (
        None,
        None,
        None,
        None,
    )
    assert with_owner_model_budget(key, None) is key


def test_grants_are_written_onto_the_token_the_caller_keeps():
    token = UserAPIKeyAuth(token="hashed", project_id="p-1")
    project = LiteLLM_ProjectTable(project_id="p-1", project_alias="Project One", metadata={"tags": ["a"]})

    stamp(token, project_grants(project))

    assert (token.project_alias, token.project_metadata) == ("Project One", {"tags": ["a"]})


def test_a_budget_sets_only_the_limits_it_has() -> None:
    budget: Final = LiteLLM_BudgetTable(tpm_limit=7, max_budget=1.5, model_rpm_limit={"gpt-a": 2})

    assert end_user_budget_grants(budget) == {
        "end_user_tpm_limit": 7,
        "end_user_max_budget": 1.5,
        "end_user_model_rpm_limit": {"gpt-a": 2},
    }


def test_a_budget_sets_every_limit_it_has() -> None:
    budget: Final = LiteLLM_BudgetTable(
        max_budget=1.5,
        tpm_limit=7,
        rpm_limit=3,
        tpd_limit=900,
        model_max_budget={"gpt-a": {"max_budget": 0.2, "budget_duration": "1d"}},
        model_tpm_limit={"gpt-a": 4},
        model_rpm_limit={"gpt-a": 2},
    )

    assert end_user_budget_grants(budget) == {
        "end_user_max_budget": 1.5,
        "end_user_tpm_limit": 7,
        "end_user_rpm_limit": 3,
        "end_user_tpd_limit": 900,
        "end_user_model_max_budget": {"gpt-a": {"max_budget": 0.2, "budget_duration": "1d"}},
        "end_user_model_tpm_limit": {"gpt-a": 4},
        "end_user_model_rpm_limit": {"gpt-a": 2},
    }


def test_a_known_end_user_sets_its_region_even_when_it_has_none() -> None:
    end_user: Final = LiteLLM_EndUserTable(user_id="eu", blocked=False)

    assert end_user_grants("eu", end_user, None) == {"end_user_id": "eu", "allowed_model_region": None}


def test_an_unknown_end_user_takes_the_default_budget_and_sets_no_region() -> None:
    assert end_user_grants("eu", None, LiteLLM_BudgetTable(rpm_limit=3)) == {
        "end_user_id": "eu",
        "end_user_rpm_limit": 3,
    }


def test_end_user_limits_replace_the_keys_while_an_unset_limit_and_the_max_budget_do_not() -> None:
    key: Final = UserAPIKeyAuth(
        token="hashed", end_user_id="stale", end_user_tpm_limit=1, end_user_rpm_limit=3, allowed_model_region="eu"
    )

    stamped: Final = with_end_user(
        key, {"end_user_tpm_limit": 7, "end_user_max_budget": 1.5, "allowed_model_region": None}
    )

    assert (
        stamped.end_user_id,
        stamped.end_user_tpm_limit,
        stamped.end_user_rpm_limit,
        stamped.end_user_max_budget,
        stamped.allowed_model_region,
    ) == (None, 7, 3, None, "eu")
    assert (key.end_user_id, key.end_user_tpm_limit) == ("stale", 1)
