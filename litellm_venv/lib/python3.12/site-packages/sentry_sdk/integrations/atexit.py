import os
import sys
import atexit

import sentry_sdk
from sentry_sdk.utils import logger
from sentry_sdk.integrations import Integration
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Optional


def default_callback(pending, timeout):
    # type: (int, int) -> None
    """This is the default shutdown callback that is set on the options.
    It prints out a message to stderr that informs the user that some events
    are still pending and the process is waiting for them to flush out.
    """

    def echo(msg):
        # type: (str) -> None
        sys.stderr.write(msg + "\n")

    echo("Sentry is attempting to send %i pending events" % pending)
    echo("Waiting up to %s seconds" % timeout)
    echo("Press Ctrl-%s to quit" % (os.name == "nt" and "Break" or "C"))
    sys.stderr.flush()


class AtexitIntegration(Integration):
    identifier = "atexit"

    def __init__(self, callback=None):
        # type: (Optional[Any]) -> None
        if callback is None:
            callback = default_callback
        self.callback = callback

    @staticmethod
    def setup_once():
        # type: () -> None
        @atexit.register
        def _shutdown():
            # type: () -> None
            client = sentry_sdk.get_client()
            integration = client.get_integration(AtexitIntegration)

            if integration is None:
                return

            logger.debug("atexit: got shutdown signal")
            logger.debug("atexit: shutting down client")
            sentry_sdk.get_isolation_scope().end_session()

            client.close(callback=integration.callback)
