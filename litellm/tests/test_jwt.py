#### What this tests ####
#    Unit tests for JWT-Auth

import sys, os, asyncio, time, random
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
from litellm.proxy._types import LiteLLMProxyRoles


def test_load_config_with_custom_role_names():
    config = {
        "general_settings": {
            "litellm_proxy_roles": {"proxy_admin": "litellm-proxy-admin"}
        }
    }

    proxy_roles = LiteLLMProxyRoles(
        **config.get("general_settings", {}).get("litellm_proxy_roles", {})
    )

    print(f"proxy_roles: {proxy_roles}")

    assert proxy_roles.proxy_admin == "litellm-proxy-admin"


# test_load_config_with_custom_role_names()
