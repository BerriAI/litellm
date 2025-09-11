import pylibmc

from .client import TracedClient


# Original Client class
_Client = pylibmc.Client


def get_version():
    # type: () -> str
    return getattr(pylibmc, "__version__", "")


def patch():
    if getattr(pylibmc, "_datadog_patch", False):
        return

    pylibmc._datadog_patch = True
    pylibmc.Client = TracedClient


def unpatch():
    if getattr(pylibmc, "_datadog_patch", False):
        pylibmc._datadog_patch = False
    pylibmc.Client = _Client
