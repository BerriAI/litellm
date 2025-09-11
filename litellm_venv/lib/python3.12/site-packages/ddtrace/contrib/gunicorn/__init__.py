"""
ddtrace works with Gunicorn.

.. note::
    If you cannot wrap your Gunicorn server with the ``ddtrace-run`` command and
    it uses ``gevent`` workers be sure to ``import ddtrace.auto`` as early as
    possible in your application's lifecycle.
    Do not use ``ddtrace-run`` with ``import ddtrace.auto``.
"""


def get_version():
    # type: () -> str
    return ""


def patch():
    pass


def unpatch():
    pass
