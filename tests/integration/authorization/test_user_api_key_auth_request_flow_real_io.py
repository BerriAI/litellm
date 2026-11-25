import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Final
from unittest.mock import ANY

import litellm
import litellm.proxy.proxy_server
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.videos.utils import encode_video_id_with_provider


class _RoutingRequest:
    def __init__(self, headers=None, query_params=None):
        self.headers = headers or {}
        self.query_params = query_params or {}
        self.state = SimpleNamespace()


# ---------------------------------------------------------------------------

# _run_centralized_common_checks — centralized authz gate
# ---------------------------------------------------------------------------


_RECORDING_DDTRACE = dedent(
    """
    import functools
    import inspect


    class _Span:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None


    class _Tracer:
        def __init__(self):
            self.spans = []

        def wrap(self, name=None, **kwargs):
            def decorator(f):
                span_name = name or f"{f.__module__}.{f.__name__}"
                if inspect.iscoroutinefunction(f):

                    @functools.wraps(f)
                    async def async_wrapped(*args, **kw):
                        self.spans.append(span_name)
                        return await f(*args, **kw)

                    return async_wrapped

                @functools.wraps(f)
                def wrapped(*args, **kw):
                    self.spans.append(span_name)
                    return f(*args, **kw)

                return wrapped

            return decorator

        def trace(self, name, **kwargs):
            return _Span()

        def current_span(self):
            return None

        def current_root_span(self):
            return None


    tracer = _Tracer()
    """
)

_DDTRACE_AUTH_PROBE = dedent(
    """
    import asyncio
    import json

    import ddtrace
    from fastapi import Request
    from starlette.datastructures import URL

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    proxy_server.master_key = "sk-probe"


    async def auth(api_key):
        request = Request(scope={"type": "http", "headers": [], "method": "POST", "path": "/chat/completions"})
        request._url = URL(url="/chat/completions")
        try:
            await user_api_key_auth(
                request=request,
                api_key=api_key,
                azure_api_key_header="",
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                custom_litellm_key_header=None,
            )
            return "accepted"
        except ProxyException:
            return "rejected"


    async def main():
        outcomes = [await auth("Bearer sk-probe"), await auth("Bearer sk-wrong")]
        print(json.dumps({"outcomes": outcomes, "spans": ddtrace.tracer.spans}))


    asyncio.run(main())
    """
)


def test_user_api_key_auth_opens_a_datadog_span_for_accepted_and_rejected_keys(tmp_path: Path):
    stub_root = tmp_path / "site"
    (stub_root / "ddtrace").mkdir(parents=True)
    (stub_root / "ddtrace" / "__init__.py").write_text(_RECORDING_DDTRACE)
    probe = tmp_path / "probe.py"
    probe.write_text(_DDTRACE_AUTH_PROBE)
    repo_root = Path(litellm.__file__).resolve().parent.parent
    env = {
        **os.environ,
        "USE_DDTRACE": "true",
        "PYTHONPATH": os.pathsep.join(
            [str(stub_root), str(repo_root)] + [p for p in (os.environ.get("PYTHONPATH"),) if p]
        ),
    }

    result = subprocess.run(
        [sys.executable, str(probe)], env=env, cwd=repo_root, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["outcomes"] == ["accepted", "rejected"]
    auth_span = "litellm.proxy.auth.user_api_key_auth.user_api_key_auth"
    assert [span for span in report["spans"] if span == auth_span] == [auth_span, auth_span]


_VIDEO_ID: Final = encode_video_id_with_provider(video_id="video_raw", provider="openai", model_id="sora-2")
_EXPIRED_BEFORE_ANY_TEST_RUN: Final = datetime(2000, 1, 1, tzinfo=timezone.utc)
