"""HTTP recording proxy used by e2e CI jobs.

This is intentionally separate from the in-process VCR persister under
``tests/_vcr_redis_persister.py``: that one only sees HTTP traffic from
the pytest process, and so cannot record requests originating from the
LiteLLM proxy when it runs in a Docker container (the case in every CI
job under ``e2e_*`` and ``proxy_*``). This package runs as a sidecar
mitmproxy that any container can route egress through via ``HTTPS_PROXY``,
making the recording layer transport-agnostic and language-agnostic.

Public surface for unit tests:

- ``cache_key.derive_cache_key`` — pure function.
- ``redis_store.RedisCassetteStore`` — thin wrapper over ``redis.Redis``.
- ``addon.CassetteAddon`` — mitmproxy addon class.
"""
