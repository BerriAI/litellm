import os
import sys
from typing import Optional


def get_proxy_signing_key() -> Optional[str]:
    salt_key = os.getenv("LITELLM_SALT_KEY")
    if salt_key is not None:
        return salt_key

    proxy_server_module = sys.modules.get("litellm.proxy.proxy_server")
    if proxy_server_module is not None:
        proxy_master_key = getattr(proxy_server_module, "master_key", None)
        if isinstance(proxy_master_key, str):
            return proxy_master_key

    return os.getenv("LITELLM_MASTER_KEY")
