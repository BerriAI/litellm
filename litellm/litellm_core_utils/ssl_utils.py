import os
from typing import Union, Optional

import litellm


def get_ssl_verify() -> Union[str, bool]:
    ssl_verify_env = os.getenv("SSL_VERIFY")

    if ssl_verify_env is None:
        return litellm.ssl_verify

    if isinstance(ssl_verify_env, str):
        if ssl_verify_env.lower() == "true":
            return True
        elif ssl_verify_env.lower() == "false":
            return False
        else:
            return ssl_verify_env


def get_ssl_certificate() -> Optional[str]:
    return os.getenv("SSL_CERTIFICATE", litellm.ssl_certificate)

