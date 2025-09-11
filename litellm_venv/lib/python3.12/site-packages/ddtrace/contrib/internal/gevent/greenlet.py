import gevent

from ddtrace._trace.provider import _DD_CONTEXTVAR


GEVENT_VERSION = gevent.version_info[0:3]


class TracingMixin(object):
    def __init__(self, *args, **kwargs):
        # Storse the current Datadog context.
        # This is necessary to ensure tracing context is passed to greenlets.
        # Avoids setting Greenlet.gr_context, setting field could introduce
        # unintended side-effects in third party libraries.
        self.trace_context = _DD_CONTEXTVAR.get()
        super(TracingMixin, self).__init__(*args, **kwargs)

    def run(self):
        # Propagates Datadog context to spawned greenlets
        _DD_CONTEXTVAR.set(self.trace_context)
        super(TracingMixin, self).run()


class TracedGreenlet(TracingMixin, gevent.Greenlet):
    """
    ``Greenlet`` class that is used to replace the original ``gevent``
    class. This class ensures any greenlet inherits the contextvars from the parent Greenlet.

    There is no need to inherit this class to create or optimize greenlets
    instances, because this class replaces ``gevent.greenlet.Greenlet``
    through the ``patch()`` method. After the patch, extending the gevent
    ``Greenlet`` class means extending automatically ``TracedGreenlet``.
    """

    def __init__(self, *args, **kwargs):
        super(TracedGreenlet, self).__init__(*args, **kwargs)


class TracedIMapUnordered(TracingMixin, gevent.pool.IMapUnordered):
    def __init__(self, *args, **kwargs):
        super(TracedIMapUnordered, self).__init__(*args, **kwargs)


class TracedIMap(TracedIMapUnordered, gevent.pool.IMap):
    def __init__(self, *args, **kwargs):
        super(TracedIMap, self).__init__(*args, **kwargs)
