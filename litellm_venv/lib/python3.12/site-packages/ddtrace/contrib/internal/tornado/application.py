from tornado import template

import ddtrace
from ddtrace import config
from ddtrace.contrib.internal.tornado import decorators
from ddtrace.contrib.internal.tornado.constants import CONFIG_KEY
from ddtrace.contrib.internal.tornado.stack_context import context_provider
from ddtrace.internal.schema import schematize_service_name


def tracer_config(__init__, app, args, kwargs):
    """
    Wrap Tornado web application so that we can configure services info and
    tracing settings after the initialization.
    """
    # call the Application constructor
    __init__(*args, **kwargs)

    # default settings
    settings = {
        "tracer": ddtrace.tracer,
        "default_service": schematize_service_name(config._get_service("tornado-web")),
        "distributed_tracing": None,
        "analytics_enabled": None,
    }

    # update defaults with users settings
    user_settings = app.settings.get(CONFIG_KEY)
    if user_settings:
        settings.update(user_settings)

    app.settings[CONFIG_KEY] = settings
    tracer = settings["tracer"]
    service = settings["default_service"]

    # extract extra settings
    extra_settings = settings.get("settings", {})

    # the tracer must use the right Context propagation and wrap executor;
    # this action is done twice because the patch() method uses the
    # global tracer while here we can have a different instance (even if
    # this is not usual).
    tracer.configure(
        context_provider=context_provider,
        wrap_executor=decorators.wrap_executor,
        enabled=settings.get("enabled", None),
        hostname=settings.get("agent_hostname", None),
        port=settings.get("agent_port", None),
        settings=extra_settings,
    )

    # set global tags if any
    tags = settings.get("tags", None)
    if tags:
        tracer.set_tags(tags)

    # configure the PIN object for template rendering
    ddtrace.Pin(service=service, tracer=tracer).onto(template)
