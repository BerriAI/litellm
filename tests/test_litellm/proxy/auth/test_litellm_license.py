import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.auth.litellm_license import LicenseCheck


def test_is_over_limit():
    license_check = LicenseCheck()
    license_check.airgapped_license_data = {"max_users": 100}
    assert license_check.is_over_limit(101) is True
    assert license_check.is_over_limit(100) is False
    assert license_check.is_over_limit(99) is False

    license_check.airgapped_license_data = {}
    assert license_check.is_over_limit(101) is False
    assert license_check.is_over_limit(100) is False
    assert license_check.is_over_limit(99) is False

    license_check.airgapped_license_data = None
    assert license_check.is_over_limit(101) is False
    assert license_check.is_over_limit(100) is False
    assert license_check.is_over_limit(99) is False
