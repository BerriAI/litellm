import sys

import sentry_sdk
from sentry_sdk.integrations import Integration
from sentry_sdk.scope import add_global_event_processor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    from sentry_sdk._types import Event, Hint


class ArgvIntegration(Integration):
    identifier = "argv"

    @staticmethod
    def setup_once():
        # type: () -> None
        @add_global_event_processor
        def processor(event, hint):
            # type: (Event, Optional[Hint]) -> Optional[Event]
            if sentry_sdk.get_client().get_integration(ArgvIntegration) is not None:
                extra = event.setdefault("extra", {})
                # If some event processor decided to set extra to e.g. an
                # `int`, don't crash. Not here.
                if isinstance(extra, dict):
                    extra["sys.argv"] = sys.argv

            return event
