from functools import wraps
from typing import Any

import sentry_sdk
from sentry_sdk.integrations import Integration, DidNotEnable

try:
    from UnleashClient import UnleashClient
except ImportError:
    raise DidNotEnable("UnleashClient is not installed")


class UnleashIntegration(Integration):
    identifier = "unleash"

    @staticmethod
    def setup_once():
        # type: () -> None
        # Wrap and patch evaluation methods (instance methods)
        old_is_enabled = UnleashClient.is_enabled

        @wraps(old_is_enabled)
        def sentry_is_enabled(self, feature, *args, **kwargs):
            # type: (UnleashClient, str, *Any, **Any) -> Any
            enabled = old_is_enabled(self, feature, *args, **kwargs)

            # We have no way of knowing what type of unleash feature this is, so we have to treat
            # it as a boolean / toggle feature.
            flags = sentry_sdk.get_current_scope().flags
            flags.set(feature, enabled)

            return enabled

        UnleashClient.is_enabled = sentry_is_enabled  # type: ignore
