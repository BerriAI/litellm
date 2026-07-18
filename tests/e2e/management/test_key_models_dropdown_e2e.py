"""The dashboard's key create/edit Models dropdown scopes its options to the key's team.

A teamless key offers All Proxy Models but not the all-team-models sentinel (the
backend expands the latter to the full proxy model list when no team is attached),
and a team key offers all-team-models plus the team's own models but never the
all-proxy-models sentinel, even when the team's model list carries it. The create
cases also walk the full product path: submit the modal with the offered sentinel
and read the persisted key back through /key/info.

The tests drive gpt-5.5, one of the example models prewired in the proxy config in
tests/e2e/docker-compose.yml; the dropdown wait fails with a pointer there when the
proxy under test does not serve it.
"""

import pytest

from e2e_config import UI_BASE_URL, unique_marker
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import KeyGenerateBody, TeamNewBody

pytest.importorskip("playwright.sync_api", reason="playwright not installed")

from playwright.sync_api import Locator, Page, expect  # noqa: E402  # import must follow the importorskip guard above


def _form_item(page: Page, label: str) -> Locator:
    return page.locator(".ant-form-item").filter(has=page.get_by_text(label, exact=True)).first


def _open_dropdown(page: Page, label: str) -> Locator:
    _form_item(page, label).locator(".ant-select-selector").first.click()
    dropdown = page.locator(".ant-select-dropdown:not(.ant-select-dropdown-hidden)").last
    expect(dropdown).to_be_visible()
    return dropdown


def _models_dropdown_texts(page: Page, must_contain: str) -> list[str]:
    dropdown = _open_dropdown(page, "Models")
    expect(
        dropdown.locator(".ant-select-item-option-content", has_text=must_contain).first,
        f"{must_contain!r} never appeared in the Models dropdown; the proxy must serve it "
        f"(see the model_list in tests/e2e/docker-compose.yml)",
    ).to_be_visible()
    return dropdown.locator(".ant-select-item-option-content").all_inner_texts()


def _open_create_key_modal(page: Page) -> None:
    # Avoid /ui/api-keys/?create=true: on stage the SPA auth redirect often
    # aborts that navigation mid-flight ("interrupted by another navigation").
    # Land on the list, wait for the shell, then open create via the button.
    page.goto(f"{UI_BASE_URL}/ui/api-keys/", wait_until="domcontentloaded")
    create_btn = page.get_by_role("button", name="+ Create New Key")
    expect(create_btn).to_be_visible(timeout=60_000)
    create_btn.click()
    expect(page.locator(".ant-modal").first).to_be_visible(timeout=15_000)


def _select_team(page: Page, alias: str) -> None:
    dropdown = _open_dropdown(page, "Team")
    dropdown.get_by_text(alias).first.click()


def _submit_create_modal(page: Page, sentinel_label: str) -> str:
    dropdown = page.locator(".ant-select-dropdown:not(.ant-select-dropdown-hidden)").last
    dropdown.locator(".ant-select-item-option-content", has_text=sentinel_label).first.click()
    page.keyboard.press("Escape")
    _form_item(page, "Key Name").locator("input").first.fill(f"e2e-ui-key-{unique_marker()}")
    page.get_by_role("button", name="Create Key", exact=True).click()

    expect(page.get_by_text("Save your Key")).to_be_visible()
    key = page.locator(".ant-modal pre").last.inner_text().strip()
    assert key.startswith("sk-"), f"expected the created key in the success modal, got {key!r}"
    return key


def _open_key_edit_form(page: Page, key_alias: str) -> None:
    page.goto(f"{UI_BASE_URL}/ui/api-keys/")
    # The list is async; wait for the provisioned row before opening detail.
    row = page.locator("tr").filter(has_text=key_alias).first
    expect(row).to_be_visible(timeout=60_000)
    # Key Alias is plain text. KeyInfoView opens from the Key ID control in the
    # same row (mono hash button on the tremor table / IdCell on the newer
    # DataTable). Prefer that button; fall back to the alias text for layouts
    # where the Key column itself is the click target.
    key_id_button = row.locator("button.font-mono").first
    if key_id_button.count() == 0:
        key_id_button = row.locator("button").first
    if key_id_button.count() > 0:
        key_id_button.click()
    else:
        row.get_by_text(key_alias, exact=True).click()
    page.get_by_role("tab", name="Settings").click()
    page.get_by_role("button", name="Edit Settings").click()
    expect(_form_item(page, "Models")).to_be_visible()


def _provision_team(client: ManagementClient, resources: ResourceManager, alias: str) -> str:
    team_id = client.create_team(TeamNewBody(team_alias=alias, models=["all-proxy-models", "gpt-5.5"]))
    resources.defer(lambda: client.delete_team(team_id))
    return team_id


def _provision_key(
    client: ManagementClient, resources: ResourceManager, alias: str, team_id: str | None = None
) -> str:
    key = client.proxy.generate_key(KeyGenerateBody(key_alias=alias, models=["gpt-5.5"], team_id=team_id))
    resources.defer(lambda: client.proxy.delete_key(key))
    return key


@pytest.mark.e2e
class TestKeyModelsDropdownUI:
    @pytest.mark.covers("mgmt.key.generate.happy_path", exercised_on=[])
    def test_create_teamless_key_offers_proxy_scope_and_persists(
        self, ui_page: Page, client: ManagementClient, resources: ResourceManager
    ) -> None:
        _open_create_key_modal(ui_page)

        options = _models_dropdown_texts(ui_page, must_contain="gpt-5.5")
        assert "All Proxy Models" in options, f"teamless create lost 'All Proxy Models': {options}"
        assert "All Team Models" not in options, f"teamless create offered 'All Team Models': {options}"

        key = _submit_create_modal(ui_page, sentinel_label="All Proxy Models")
        resources.defer(lambda: client.proxy.delete_key(key))

        info = client.proxy.key_info(key)
        assert info.models == ["all-proxy-models"], f"persisted models {info.models}"
        assert info.team_id is None, f"teamless key persisted with team {info.team_id}"

    @pytest.mark.covers("mgmt.key.generate.happy_path", exercised_on=[])
    def test_create_team_key_offers_team_scope_and_persists(
        self, ui_page: Page, client: ManagementClient, resources: ResourceManager
    ) -> None:
        team_alias = f"e2e-ui-team-{unique_marker()}"
        team_id = _provision_team(client, resources, team_alias)

        _open_create_key_modal(ui_page)
        _select_team(ui_page, team_alias)

        options = _models_dropdown_texts(ui_page, must_contain="All Team Models")
        assert "gpt-5.5" in options, f"team key create lost the team's own model: {options}"
        assert "All Proxy Models" not in options, f"team key create offered 'All Proxy Models': {options}"
        assert "all-proxy-models" not in options, f"team key create offered the raw sentinel: {options}"

        key = _submit_create_modal(ui_page, sentinel_label="All Team Models")
        resources.defer(lambda: client.proxy.delete_key(key))

        info = client.proxy.key_info(key)
        assert info.models == ["all-team-models"], f"persisted models {info.models}"
        assert info.team_id == team_id, f"persisted team {info.team_id}, expected {team_id}"

    @pytest.mark.covers("mgmt.key.update.happy_path", exercised_on=[])
    def test_edit_teamless_key_offers_proxy_scope(
        self, ui_page: Page, client: ManagementClient, resources: ResourceManager
    ) -> None:
        key_alias = f"e2e-ui-teamless-{unique_marker()}"
        _provision_key(client, resources, key_alias)

        _open_key_edit_form(ui_page, key_alias)

        options = _models_dropdown_texts(ui_page, must_contain="gpt-5.5")
        assert "All Proxy Models" in options, f"teamless edit lost 'All Proxy Models': {options}"
        assert "All Team Models" not in options, f"teamless edit offered 'All Team Models': {options}"

    @pytest.mark.covers("mgmt.key.update.happy_path", exercised_on=[])
    def test_edit_team_key_offers_team_scope_only(
        self, ui_page: Page, client: ManagementClient, resources: ResourceManager
    ) -> None:
        team_alias = f"e2e-ui-team-{unique_marker()}"
        team_id = _provision_team(client, resources, team_alias)
        key_alias = f"e2e-ui-teamkey-{unique_marker()}"
        _provision_key(client, resources, key_alias, team_id=team_id)

        _open_key_edit_form(ui_page, key_alias)

        # Wait on a real team model: All Team Models is rendered immediately while
        # availableModels is still fetching, so requiring only the sentinel races
        # the async team-model load and can read an incomplete dropdown.
        options = _models_dropdown_texts(ui_page, must_contain="gpt-5.5")
        assert "All Team Models" in options, f"team key edit lost 'All Team Models': {options}"
        assert "All Proxy Models" not in options, f"team key edit offered 'All Proxy Models': {options}"
        assert "all-proxy-models" not in options, f"team key edit offered the raw sentinel: {options}"
