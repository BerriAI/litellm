# What is this?
## This tests the batch update spend logic on the proxy server


import asyncio
import os
import random
import sys
import time
import traceback
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Request

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import logging
from litellm.proxy.management_endpoints.sso_helper_utils import (
    check_is_admin_only_access,
    has_admin_ui_access,
)
from litellm.proxy._types import LitellmUserRoles


def test_check_is_admin_only_access():
    assert check_is_admin_only_access("admin_only") is True
    assert check_is_admin_only_access("user_only") is False


def test_has_admin_ui_access():
    assert has_admin_ui_access(LitellmUserRoles.PROXY_ADMIN.value) is True
    assert has_admin_ui_access(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value) is True
    assert has_admin_ui_access(LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value) is False
