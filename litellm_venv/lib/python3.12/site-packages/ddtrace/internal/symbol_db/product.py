from ddtrace.settings.symbol_db import config


requires = ["remote-configuration"]


def post_preload():
    pass


def start():
    if config.enabled:
        from ddtrace.internal import symbol_db

        symbol_db.bootstrap()


def restart(join=False):
    if not config._force:
        from ddtrace.internal import symbol_db

        symbol_db.restart()


def stop(join=False):
    # Controlled via RC
    pass


def at_exit(join=False):
    stop(join=join)
