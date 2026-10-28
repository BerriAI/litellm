import hashlib
import os
import secrets
from typing import Final

_xdist_test_run_uid: Final = os.environ.get("PYTEST_XDIST_TESTRUNUID")
MASTER_KEY: Final = (
    f"sk-{hashlib.sha256(_xdist_test_run_uid.encode()).hexdigest()[:32]}"
    if _xdist_test_run_uid is not None
    else f"sk-{secrets.token_hex(16)}"
)
