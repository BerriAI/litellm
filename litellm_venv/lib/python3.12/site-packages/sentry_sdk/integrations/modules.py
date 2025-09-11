import sentry_sdk
from sentry_sdk.integrations import Integration
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.utils import _get_installed_modules

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from sentry_sdk._types import Event


class ModulesIntegration(Integration):
    identifier = "modules"

    @staticmethod
    def setup_once():
        # type: () -> None
        @add_global_event_processor
        def processor(event, hint):
            # type: (Event, Any) -> Event
            if event.get("type") == "transaction":
                return event

            if sentry_sdk.get_client().get_integration(ModulesIntegration) is None:
                return event

            event["modules"] = _get_installed_modules()
            return event
