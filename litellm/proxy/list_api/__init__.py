"""Surface-neutral machinery for LiteLLM's own paginated list endpoints.

Lives outside `management_endpoints/` because `/public/v1` builds on it too, and a
control-plane package is the wrong home for something the public surface imports.
"""
