from ddtrace import config
from ddtrace.internal.remoteconfig.client import config as rc_config


def post_preload():
    pass


def start():
    if config._remote_config_enabled:
        from ddtrace.internal.remoteconfig.worker import remoteconfig_poller

        remoteconfig_poller.enable()
        config.enable_remote_configuration()


def restart(join=False):
    if config._remote_config_enabled:
        from ddtrace.internal.remoteconfig.worker import remoteconfig_poller

        remoteconfig_poller.reset_at_fork()


def stop(join=False):
    if config._remote_config_enabled:
        from ddtrace.internal.remoteconfig.worker import remoteconfig_poller

        remoteconfig_poller.disable(join=join)


def at_exit(join=False):
    if config._remote_config_enabled and not rc_config.skip_shutdown:
        stop(join=join)
