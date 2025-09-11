"""
The Celery integration will trace all tasks that are executed in the
background. Functions and class based tasks are traced only if the Celery API
is used, so calling the function directly or via the ``run()`` method will not
generate traces. However, calling ``apply()``, ``apply_async()`` and ``delay()``
will produce tracing data. To trace your Celery application, call the patch method::

    import celery
    from ddtrace import patch

    patch(celery=True)
    app = celery.Celery()

    @app.task
    def my_task():
        pass

    class MyTask(app.Task):
        def run(self):
            pass

Configuration
~~~~~~~~~~~~~
.. py:data:: ddtrace.config.celery['distributed_tracing']

   Whether or not to pass distributed tracing headers to Celery workers.
   Note: this flag applies to both Celery workers and callers separately.

   On the caller: enabling propagation causes the caller and worker to
   share a single trace while disabling causes them to be separate.

   On the worker: enabling propagation causes context to propagate across
   tasks, such as when Task A queues work for Task B, or if Task A retries.
   Disabling propagation causes each celery.run task to be in its own
   separate trace.

   Can also be enabled with the ``DD_CELERY_DISTRIBUTED_TRACING`` environment variable.

   Default: ``False``

.. py:data:: ddtrace.config.celery['producer_service_name']

   Sets service name for producer

   Default: ``'celery-producer'``

.. py:data:: ddtrace.config.celery['worker_service_name']

   Sets service name for worker

   Default: ``'celery-worker'``

"""
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["celery"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.celery.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.celery.app import patch_app
        from ddtrace.contrib.internal.celery.app import unpatch_app
        from ddtrace.contrib.internal.celery.patch import get_version
        from ddtrace.contrib.internal.celery.patch import patch
        from ddtrace.contrib.internal.celery.patch import unpatch

        __all__ = ["patch", "patch_app", "unpatch", "unpatch_app", "get_version"]
