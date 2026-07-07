"""The dashboard's key create/edit Models dropdown scopes its options to the key's team.

A teamless key must not offer the all-team-models sentinel (the backend expands it
to the full proxy model list when no team is attached), and a team key must offer
all-team-models but never the all-proxy-models sentinel, even when the team's own
model list carries it. The create case also walks the full product path: submit the
modal with All Team Models and read the persisted key back through /key/info.
"""

import pytest

from e2e_config import UI_BASE_URL, unique_marker
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import KeyGenerateBody, TeamNewBody

pytest.importorskip("playwright.sync_api", reason="playwright not installed")

from playwright.sync_api import Locator, Page, expect  # noqa: E402

REAL_MODEL = "gpt-5.5"
ALL_TEAM_MODELS_LABEL = "All Team Models"
PROXY_SENTINEL_LABELS = ("All Proxy Models", "all-proxy-models")


def _form_item(page: Page, label: str) -> Locator:
    return page.locator(".ant-form-item").filter(has=page.get_by_text(label, exact=True)).first


def _open_dropdown(page: Page, label: str) -> Locator:
    _form_item(page, label).locator(".ant-select-selector").first.click()
    dropdown = page.locator(".ant-select-dropdown:not(.ant-select-dropdown-hidden)").last
    expect(dropdown).to_be_visible()
    return dropdown


def _models_dropdown_texts(page: Page, must_contain: str) -> list[str]:
    dropdown = _open_dropdown(page, "Models")
    expect(dropdown.locator(".ant-select-item-option-content", has_text=must_contain).first).to_be_visible()
    return dropdown.locator(".ant-select-item-option-content").all_inner_texts()


def _open_create_key_modal(page: Page) -> None:
    page.goto(f"{UI_BASE_URL}/api-keys/?create=true")
    expect(page.locator(".ant-modal").first).to_be_visible()


def _select_team(page: Page, alias: str) -> None:
    dropdown = _open_dropdown(page, "Team")
    dropdown.get_by_text(alias).first.click()


def _open_key_edit_form(page: Page, key_alias: str) -> None:
    page.goto(f"{UI_BASE_URL}/api-keys/")
    page.get_by_text(key_alias).first.click()
    page.get_by_role("tab", name="Settings").click()
    page.get_by_role("button", name="Edit Settings").click()
    expect(_form_item(page, "Models")).to_be_visible()


def _provision_team(client: ManagementClient, resources: ResourceManager, alias: str) -> str:
    team_id = client.create_team(TeamNewBody(team_alias=alias, models=["all-proxy-models", REAL_MODEL]))
    resources.defer(lambda: client.delete_team(team_id))
    return team_id


def _provision_key(
    client: ManagementClient, resources: ResourceManager, alias: str, team_id: str | None = None
) -> str:
    key = client.gateway.generate_key(KeyGenerateBody(key_alias=alias, models=[REAL_MODEL], team_id=team_id))
    resources.defer(lambda: client.gateway.delete_key(key))
    return key


@pytest.mark.e2e
class TestKeyModelsDropdownUI:
    @pytest.mark.covers("mgmt.key.generate.happy_path", exercised_on=[])
    def test_create_key_without_team_hides_all_team_models(self, ui_page: Page) -> None:
        _open_create_key_modal(ui_page)

        options = _models_dropdown_texts(ui_page, must_contain=REAL_MODEL)

        assert ALL_TEAM_MODELS_LABEL not in options, f"teamless create offered {ALL_TEAM_MODELS_LABEL!r}: {options}"
        assert not set(PROXY_SENTINEL_LABELS) & set(options), f"create offered a proxy-wide sentinel: {options}"

    @pytest.mark.covers("mgmt.key.generate.happy_path", exercised_on=[])
    def test_create_team_key_offers_team_scope_and_persists(
        self, ui_page: Page, client: ManagementClient, resources: ResourceManager
    ) -> None:
        team_alias = f"e2e-ui-team-{unique_marker()}"
        team_id = _provision_team(client, resources, team_alias)

        _open_create_key_modal(ui_page)
        _select_team(ui_page, team_alias)

        options = _models_dropdown_texts(ui_page, must_contain=ALL_TEAM_MODELS_LABEL)
        assert REAL_MODEL in options, f"team key create lost the team's own model: {options}"
        assert not set(PROXY_SENTINEL_LABELS) & set(options), f"team key create offered a proxy-wide sentinel: {options}"

        dropdown = ui_page.locator(".ant-select-dropdown:not(.ant-select-dropdown-hidden)").last
        dropdown.locator(".ant-select-item-option-content", has_text=ALL_TEAM_MODELS_LABEL).first.click()
        ui_page.keyboard.press("Escape")
        _form_item(ui_page, "Key Name").locator("input").first.fill(f"e2e-ui-key-{unique_marker()}")
        ui_page.get_by_role("button", name="Create Key", exact=True).click()

        expect(ui_page.get_by_text("Save your Key")).to_be_visible()
        key = ui_page.locator(".ant-modal pre").last.inner_text().strip()
        assert key.startswith("sk-"), f"expected the created key in the success modal, got {key!r}"
        resources.defer(lambda: client.gateway.delete_key(key))

        info = client.gateway.key_info(key)
        assert info.models == ["all-team-models"], f"persisted models {info.models}"
        assert info.team_id == team_id, f"persisted team {info.team_id}, expected {team_id}"

    @pytest.mark.covers("mgmt.key.update.happy_path", exercised_on=[])
    def test_edit_teamless_key_hides_all_team_models(
        self, ui_page: Page, client: ManagementClient, resources: ResourceManager
    ) -> None:
        key_alias = f"e2e-ui-teamless-{unique_marker()}"
        _provision_key(client, resources, key_alias)

        _open_key_edit_form(ui_page, key_alias)

        options = _models_dropdown_texts(ui_page, must_contain=REAL_MODEL)
        assert ALL_TEAM_MODELS_LABEL not in options, f"teamless edit offered {ALL_TEAM_MODELS_LABEL!r}: {options}"

    @pytest.mark.covers("mgmt.key.update.happy_path", exercised_on=[])
    def test_edit_team_key_offers_team_scope_only(
        self, ui_page: Page, client: ManagementClient, resources: ResourceManager
    ) -> None:
        team_alias = f"e2e-ui-team-{unique_marker()}"
        team_id = _provision_team(client, resources, team_alias)
        key_alias = f"e2e-ui-teamkey-{unique_marker()}"
        _provision_key(client, resources, key_alias, team_id=team_id)

        _open_key_edit_form(ui_page, key_alias)

        options = _models_dropdown_texts(ui_page, must_contain=ALL_TEAM_MODELS_LABEL)
        assert REAL_MODEL in options, f"team key edit lost the team's own model: {options}"
        assert not set(PROXY_SENTINEL_LABELS) & set(options), f"team key edit offered a proxy-wide sentinel: {options}"
