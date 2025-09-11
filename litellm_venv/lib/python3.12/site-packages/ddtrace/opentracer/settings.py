from collections import namedtuple
from typing import Any  # noqa:F401
from typing import Dict  # noqa:F401
from typing import List  # noqa:F401


# Keys used for the configuration dict
ConfigKeyNames = namedtuple(
    "ConfigKeyNames",
    [
        "AGENT_HOSTNAME",
        "AGENT_HTTPS",
        "AGENT_PORT",
        "DEBUG",
        "ENABLED",
        "GLOBAL_TAGS",
        "SAMPLER",
        "PRIORITY_SAMPLING",
        "UDS_PATH",
        "SETTINGS",
    ],
)

ConfigKeys = ConfigKeyNames(
    AGENT_HOSTNAME="agent_hostname",
    AGENT_HTTPS="agent_https",
    AGENT_PORT="agent_port",
    DEBUG="debug",
    ENABLED="enabled",
    GLOBAL_TAGS="global_tags",
    SAMPLER="sampler",
    PRIORITY_SAMPLING="priority_sampling",
    UDS_PATH="uds_path",
    SETTINGS="settings",
)


def config_invalid_keys(config):
    # type: (Dict[str, Any]) -> List[str]
    """Returns a list of keys that exist in *config* and not in KEYS."""
    return [key for key in config.keys() if key not in ConfigKeys]
