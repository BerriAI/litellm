import pymemcache
import pymemcache.client.hash

from ddtrace.ext import memcached as memcachedx
from ddtrace.internal.schema import schematize_service_name
from ddtrace.pin import _DD_PIN_NAME
from ddtrace.pin import _DD_PIN_PROXY_NAME
from ddtrace.pin import Pin

from .client import WrappedClient
from .client import WrappedHashClient


_Client = pymemcache.client.base.Client
_hash_Client = pymemcache.client.hash.Client
_hash_HashClient = pymemcache.client.hash.Client


def get_version():
    # type: () -> str
    return getattr(pymemcache, "__version__", "")


def patch():
    if getattr(pymemcache, "_datadog_patch", False):
        return

    pymemcache._datadog_patch = True
    pymemcache.client.base.Client = WrappedClient
    pymemcache.client.hash.Client = WrappedClient
    pymemcache.client.hash.HashClient = WrappedHashClient

    # Create a global pin with default configuration for our pymemcache clients
    service = schematize_service_name(memcachedx.SERVICE)
    Pin(service=service).onto(pymemcache)


def unpatch():
    """Remove pymemcache tracing"""
    if not getattr(pymemcache, "_datadog_patch", False):
        return
    pymemcache._datadog_patch = False
    pymemcache.client.base.Client = _Client
    pymemcache.client.hash.Client = _hash_Client
    pymemcache.client.hash.HashClient = _hash_HashClient

    # Remove any pins that may exist on the pymemcache reference
    setattr(pymemcache, _DD_PIN_NAME, None)
    setattr(pymemcache, _DD_PIN_PROXY_NAME, None)
