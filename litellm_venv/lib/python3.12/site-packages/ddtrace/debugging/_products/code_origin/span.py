from ddtrace.settings.code_origin import config


# TODO[gab]: Uncomment this when the feature is ready
# requires = ["tracer"]


def post_preload():
    pass


def start():
    if config.span.enabled:
        from ddtrace.debugging._origin.span import SpanCodeOriginProcessor

        SpanCodeOriginProcessor.enable()


def restart(join=False):
    pass


def stop(join=False):
    if config.span.enabled:
        from ddtrace.debugging._origin.span import SpanCodeOriginProcessor

        SpanCodeOriginProcessor.disable()


def at_exit(join=False):
    stop(join=join)
