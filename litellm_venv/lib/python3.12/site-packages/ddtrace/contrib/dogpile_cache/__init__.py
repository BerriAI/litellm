"""
Instrument dogpile.cache__ to report all cached lookups.

This will add spans around the calls to your cache backend (e.g. redis, memory,
etc). The spans will also include the following tags:

- key/keys: The key(s) dogpile passed to your backend. Note that this will be
  the output of the region's ``function_key_generator``, but before any key
  mangling is applied (i.e. the region's ``key_mangler``).
- region: Name of the region.
- backend: Name of the backend class.
- hit: If the key was found in the cache.
- expired: If the key is expired. This is only relevant if the key was found.

While cache tracing will generally already have keys in tags, some caching
setups will not have useful tag values - such as when you're using consistent
hashing with memcached - the key(s) will appear as a mangled hash.
::

    # Patch before importing dogpile.cache
    from ddtrace import patch
    patch(dogpile_cache=True)

    from dogpile.cache import make_region

    region = make_region().configure(
        "dogpile.cache.pylibmc",
        expiration_time=3600,
        arguments={"url": ["127.0.0.1"]},
    )

    @region.cache_on_arguments()
    def hello(name):
        # Some complicated, slow calculation
        return "Hello, {}".format(name)

.. __: https://dogpilecache.sqlalchemy.org/
"""
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["dogpile.cache"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.dogpile_cache.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.dogpile_cache.patch import get_version
        from ddtrace.contrib.internal.dogpile_cache.patch import patch
        from ddtrace.contrib.internal.dogpile_cache.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
