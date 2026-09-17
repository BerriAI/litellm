"""Static file serving for the Next.js dashboard export."""

import os

from starlette.staticfiles import StaticFiles


class UiStaticFiles(StaticFiles):
    """StaticFiles that falls back to ``<route>.html`` when ``<route>/index.html`` is missing.

    Next.js exports built without ``trailingSlash: true`` emit ``chat.html`` plus an
    index-less ``chat/`` data directory; Starlette matches the directory first, finds no
    ``index.html``, and returns 404 (https://github.com/BerriAI/litellm/issues/24037).
    Resolving the fallback at lookup time serves both export layouts as-is, so the UI
    works on read-only filesystems without restructuring files on disk at startup.
    """

    def lookup_path(self, path: str) -> "tuple[str, os.stat_result | None]":
        full_path, stat_result = super().lookup_path(path)
        if stat_result is not None:
            return full_path, stat_result
        route = path.replace(os.sep, "/").rstrip("/")
        if route.endswith("/index.html"):
            return super().lookup_path(f"{route.removesuffix('/index.html')}.html")
        if route and not route.endswith(".html"):
            return super().lookup_path(f"{route}.html")
        return full_path, stat_result
