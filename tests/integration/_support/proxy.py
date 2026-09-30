"""Run the normal CLI with the existing behavior-suite test entitlement in the parent and every spawned worker."""

import signal
import sys
from types import FrameType
from typing import Final
from unittest.mock import patch

from litellm import run_server

_ENTITLEMENT: Final = patch(  # test-quality-ok: route entitlement only; license checks are outside these contracts
    "litellm.proxy.auth.litellm_license.LicenseCheck.is_premium", return_value=True
)

if __name__ == "__mp_main__":
    _ENTITLEMENT.start()


def _exit_on_reraised_term(signum: int, frame: FrameType | None) -> None:
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGTERM, _exit_on_reraised_term)
    with _ENTITLEMENT:
        run_server()


if __name__ == "__main__":
    main()
