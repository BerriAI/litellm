"""Live e2e: `general_settings.model_list_return_wildcard_routes`, the proxy-wide
default for whether GET /v1/models lists a wildcard route such as `openai/*` next to
the models it expands to.

The setting is written through /config/field/update, the route behind the admin UI's
General Settings toggle, and teardown puts back whatever the database held before, so
the shared proxy ends the test the way it started. The other listings the suites run either pass
return_wildcard_routes explicitly or only check that a named model is present, so the
window with the setting on changes none of them. The wildcard deployment gets a unique prefix instead of `openai/*`, which would
claim every `openai/...` request the other suites send.
"""

from __future__ import annotations

from typing import Final, Literal

import pytest
from pydantic import BaseModel, ConfigDict, JsonValue, RootModel

from e2e_config import unique_marker
from e2e_http import NoBody, Success, unwrap
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import ConfigListParams, LiteLLMParamsBody, ModelsListParams, ModelsListResponse

pytestmark = pytest.mark.e2e

_SETTING: Final = "model_list_return_wildcard_routes"
_DUMMY_API_KEY: Final = "e2e-dummy-key"


class ConfigFieldUpdateBody(BaseModel):
    field_name: str
    field_value: JsonValue
    config_type: str = "general_settings"


class ConfigFieldDeleteBody(BaseModel):
    field_name: str
    config_type: str = "general_settings"


class StoredField(BaseModel):
    model_config = ConfigDict(extra="ignore")
    field_name: str
    field_value: JsonValue = None
    source: Literal["config", "db", "env", "default", "unset"] = "unset"


class StoredFieldList(RootModel[tuple[StoredField, ...]]):
    pass


def _write_setting(client: ManagementClient, value: JsonValue) -> None:
    _ = unwrap(
        client.proxy.transport.post(
            "/config/field/update",
            headers=client.proxy.transport.master,
            json=ConfigFieldUpdateBody(field_name=_SETTING, field_value=value),
            response_type=NoBody,
        )
    )


def _delete_setting(client: ManagementClient) -> None:
    _ = unwrap(
        client.proxy.transport.post(
            "/config/field/delete",
            headers=client.proxy.transport.master,
            json=ConfigFieldDeleteBody(field_name=_SETTING),
            response_type=NoBody,
        )
    )


def _stored_setting(client: ManagementClient) -> StoredField | None:
    fields: Final = unwrap(
        client.proxy.transport.get(
            "/config/list",
            headers=client.proxy.transport.master,
            params=ConfigListParams(config_type="general_settings"),
            response_type=StoredFieldList,
        )
    ).root
    return next((field for field in fields if field.field_name == _SETTING and field.source == "db"), None)


def _restore_setting(client: ManagementClient, stored: StoredField | None) -> None:
    if stored is None:
        _delete_setting(client)
        return
    _write_setting(client, stored.field_value)


def _await_listing(client: ManagementClient, pattern: str, query: BaseModel, *, listed: bool) -> None:
    """Poll /v1/models under `query` on every replica until each one lists `pattern`,
    or each one leaves it out, per `listed`."""
    _ = client.proxy.read_back_everywhere(
        "/v1/models",
        params=query,
        response_type=ModelsListResponse,
        converged=lambda result: (
            isinstance(result, Success) and any(entry.id == pattern for entry in result.data.data) is listed
        ),
    )


class TestModelListWildcardRoutesSetting:
    @pytest.mark.covers("mgmt.general_settings.model_list_return_wildcard_routes.persists")
    def test_setting_lists_wildcard_route_unless_request_opts_out(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        pattern = f"e2e-wildcard-{unique_marker()}/*"
        model_id = client.proxy.create_model(pattern, LiteLLMParamsBody(model="openai/*", api_key=_DUMMY_API_KEY))
        resources.defer(lambda: client.proxy.delete_model(model_id))

        stored: Final = _stored_setting(client)
        resources.defer(lambda: _restore_setting(client, stored))
        _write_setting(client, False)
        _await_listing(client, pattern, NoBody(), listed=False)

        _write_setting(client, True)
        _await_listing(client, pattern, NoBody(), listed=True)
        _await_listing(client, pattern, ModelsListParams(return_wildcard_routes=False), listed=False)

        _delete_setting(client)
        _await_listing(client, pattern, NoBody(), listed=False)
