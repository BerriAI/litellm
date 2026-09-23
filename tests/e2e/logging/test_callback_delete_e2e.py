"""Live e2e: a callback the Admin UI lists must also be deletable.

GET /get/config/callbacks enumerates litellm_settings.success_callback,
failure_callback and callbacks, plus runtime-only loggers, but
POST /config/callback/delete only looks inside success_callback. A callback
configured under `callbacks` or `failure_callback` is listed as active on the
Logging & Alerts -> Logging Callbacks page yet its delete answers 404 "Callback
not found in active configuration". The success_callback case is the control:
it proves the add/list/delete round trip works when the configured key is the
one the delete route actually reads.
"""

from collections.abc import Callable

import pytest
from e2e_http import unwrap
from lifecycle import ResourceManager
from logging_client import LoggingClient
from models import LitellmCallbackSettings

pytestmark = pytest.mark.e2e

CALLBACK_NAME = "datadog"


def _assert_listed_then_deleted(
    client: LoggingClient,
    resources: ResourceManager,
    settings: LitellmCallbackSettings,
    cleanup: Callable[[], object],
) -> None:
    response = client.set_litellm_callbacks(settings)
    assert "success" in response.message.lower(), (
        f"/config/update reported {response.message!r}, expected a success message"
    )
    resources.defer(cleanup)

    names = client.config_callback_names()
    assert CALLBACK_NAME in names, (
        f"GET /get/config/callbacks does not list {CALLBACK_NAME!r} after it was "
        f"configured via /config/update; listed {sorted(names)}"
    )

    deleted = unwrap(client.delete_config_callback(CALLBACK_NAME))
    assert deleted.removed_callback == CALLBACK_NAME, (
        f"/config/callback/delete removed {deleted.removed_callback!r}, expected {CALLBACK_NAME!r}"
    )

    names_after = client.config_callback_names()
    assert CALLBACK_NAME not in names_after, (
        f"GET /get/config/callbacks still lists {CALLBACK_NAME!r} after a successful "
        f"delete; listed {sorted(names_after)}"
    )


class TestCallbackDelete:
    def test_callback_under_callbacks_key_is_listed_and_deletable(
        self, client: LoggingClient, resources: ResourceManager
    ) -> None:
        _assert_listed_then_deleted(
            client,
            resources,
            LitellmCallbackSettings(callbacks=[CALLBACK_NAME]),
            lambda: client.clear_litellm_callbacks("callbacks"),
        )

    def test_callback_under_failure_callback_key_is_listed_and_deletable(
        self, client: LoggingClient, resources: ResourceManager
    ) -> None:
        _assert_listed_then_deleted(
            client,
            resources,
            LitellmCallbackSettings(failure_callback=[CALLBACK_NAME]),
            lambda: client.clear_litellm_callbacks("failure_callback"),
        )

    def test_callback_under_success_callback_key_is_deletable(
        self, client: LoggingClient, resources: ResourceManager
    ) -> None:
        _assert_listed_then_deleted(
            client,
            resources,
            LitellmCallbackSettings(success_callback=[CALLBACK_NAME]),
            lambda: client.delete_config_callback(CALLBACK_NAME),
        )
