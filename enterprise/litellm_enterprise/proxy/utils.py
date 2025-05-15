from typing import Optional, Union

from litellm.secret_managers.main import str_to_bool


def _should_block_robots():
    """
    Returns True if the robots.txt file should block web crawlers

    Controlled by

    ```yaml
    general_settings:
      block_robots: true
    ```
    """
    from litellm.proxy.proxy_server import (
        CommonProxyErrors,
        general_settings,
        premium_user,
    )

    _block_robots: Union[bool, str] = general_settings.get("block_robots", False)
    block_robots: Optional[bool] = None
    if isinstance(_block_robots, bool):
        block_robots = _block_robots
    elif isinstance(_block_robots, str):
        block_robots = str_to_bool(_block_robots)
    if block_robots is True:
        if premium_user is not True:
            raise ValueError(
                f"Blocking web crawlers is an enterprise feature. {CommonProxyErrors.not_premium_user.value}"
            )
        return True
    return False
