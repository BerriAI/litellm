from ddtrace import config


def get_poll_interval_seconds():
    # type:() -> float
    return config._remote_config_poll_interval
