"""Run the normal CLI with the behavior-suite test entitlement, patched at import so spawned workers inherit it."""

import signal
import sys
from types import FrameType
from unittest.mock import patch

from litellm import run_server

patch(  # test-quality-ok: route entitlement only; license validation is outside these HTTP/DB contracts
    "litellm.proxy.auth.litellm_license.LicenseCheck.is_premium", return_value=True
).start()


def _exit_on_reraised_term(signum: int, frame: FrameType | None) -> None:
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGTERM, _exit_on_reraised_term)
    run_server()


if __name__ == "__main__":
    main()
