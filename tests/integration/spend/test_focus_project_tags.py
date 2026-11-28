from typing import Final

from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from pydantic import JsonValue, TypeAdapter

_ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])


def test_focus_dry_run_tags_spend_by_project_without_changing_the_team_subaccount(
    gateway: Gateway, scenario: Scenario
) -> None:
    model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
    team: Final = scenario.team(team_alias=f"{model}-team")
    project: Final = scenario.project(team, project_alias=f"{model}-project")
    user: Final = scenario.member(team)
    scoped: Final = scenario.key(
        user_id=user, team_id=team, project_id=project, key_alias=f"{model}-scoped", models=[model]
    )
    unscoped: Final = scenario.key(user_id=user, team_id=team, key_alias=f"{model}-unscoped", models=[model])
    gateway.chat(model, key=scoped)
    gateway.chat(model, key=unscoped)

    def export_rows() -> dict[str, dict[str, JsonValue]]:
        data: Final = object_value(gateway.post("/vantage/dry-run", {"limit": 50})["dry_run_data"])
        return {
            string_value(row["BillingAccountName"]): row
            for row in _ROWS.validate_python(data["normalized_data"])
            if row["ServiceName"] == model
        }

    rows: Final = eventually(export_rows, lambda sample: len(sample) == 2, seconds=70)
    scoped_row: Final = rows[f"{model}-scoped"]
    unscoped_row: Final = rows[f"{model}-unscoped"]
    assert (scoped_row["SubAccountId"], scoped_row["SubAccountName"]) == (team, f"{model}-team")
    assert (unscoped_row["SubAccountId"], unscoped_row["SubAccountName"]) == (team, f"{model}-team")
    scoped_tags: Final = JSON_OBJECT.validate_json(string_value(scoped_row["Tags"]))
    unscoped_tags: Final = JSON_OBJECT.validate_json(string_value(unscoped_row["Tags"]))
    assert {"project_id": scoped_tags.get("project_id"), "project_alias": scoped_tags.get("project_alias")} == {
        "project_id": project,
        "project_alias": f"{model}-project",
    }
    assert "project_id" not in unscoped_tags
    assert "project_alias" not in unscoped_tags
