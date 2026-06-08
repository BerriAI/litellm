"""Identity extractors.

Each extractor wraps an existing helper in ``litellm/proxy/auth/`` and
returns a piece of an ``IdentityContext``. Extractors must not introduce
new behavior. If you need to change *how* a field is resolved, change the
underlying helper and update the extractor's tests.
"""
