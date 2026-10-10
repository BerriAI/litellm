import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, object_value, string_value
from tests.integration._support.database import read_rows


def _budget_rows(budget_id: str) -> list[dict[str, object]]:
    return read_rows(
        'SELECT tpm_limit, rpm_limit, max_budget FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget_id,)
    )


@pytest.mark.covers("mgmt.organization.update.null_clears_budget_limit")
def test_patch_organization_update_with_null_tpm_limit_clears_it_and_keeps_sibling_limits(gateway: Gateway) -> None:
    created: Final = gateway.post(
        "/organization/new",
        {
            "organization_alias": f"integration-{uuid.uuid4().hex}",
            "tpm_limit": 4000,
            "rpm_limit": 40,
            "max_budget": 12.5,
        },
    )
    organization_id: Final = string_value(created["organization_id"])
    budget_id: Final = string_value(created["budget_id"])
    try:
        assert _budget_rows(budget_id) == [{"tpm_limit": 4000, "rpm_limit": 40, "max_budget": 12.5}]
        updated: Final = gateway.request(
            "PATCH", "/organization/update", {"organization_id": organization_id, "tpm_limit": None}
        )
        assert updated.status_code == 200, updated.text
        updated_budget: Final = object_value(object_value(updated.json())["litellm_budget_table"])
        assert (updated_budget["tpm_limit"], updated_budget["rpm_limit"], updated_budget["max_budget"]) == (
            None,
            40,
            12.5,
        ), updated.text
        assert _budget_rows(budget_id) == [{"tpm_limit": None, "rpm_limit": 40, "max_budget": 12.5}]
        info: Final = gateway.request("GET", "/organization/info", params={"organization_id": organization_id})
        assert info.status_code == 200, info.text
        info_budget: Final = object_value(object_value(info.json())["litellm_budget_table"])
        assert (info_budget["tpm_limit"], info_budget["rpm_limit"], info_budget["max_budget"]) == (
            None,
            40,
            12.5,
        ), info.text
    finally:
        deleted: Final = gateway.request("DELETE", "/organization/delete", {"organization_ids": [organization_id]})
        assert deleted.status_code == 200, deleted.text
        gateway.post("/budget/delete", {"id": budget_id})
        assert (
            read_rows(
                'SELECT organization_id FROM "LiteLLM_OrganizationTable" WHERE organization_id = %s', (organization_id,)
            )
            == []
        )
