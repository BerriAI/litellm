from ddtrace.settings.dynamic_instrumentation import config


requires = ["remote-configuration"]


def post_preload():
    pass


def start():
    if config.enabled:
        from ddtrace.debugging import DynamicInstrumentation

        DynamicInstrumentation.enable()


def restart(join=False):
    # Nothing to do
    pass


def stop(join=False):
    if config.enabled:
        from ddtrace.debugging import DynamicInstrumentation

        DynamicInstrumentation.disable(join=join)


def at_exit(join=False):
    stop(join=join)
