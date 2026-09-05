import asyncio
import os
import sys
from typing import Final
from unittest.mock import patch

from callback_support import CallbackRecorder, OcrArguments, call_ocr, verify_installed_package

import litellm
from litellm.rust_bridge import get_native_bridge


async def smoke(*, baseline: bool) -> None:
    verify_installed_package()
    litellm.rust(True)
    for asynchronous, rejected in ((False, False), (True, False), (True, True)):
        await smoke_case(asynchronous, rejected, baseline=baseline)


async def smoke_case(asynchronous: bool, rejected: bool, *, baseline: bool) -> None:
    recorder: Final = CallbackRecorder(asynchronous, name=f"live-{asynchronous}-{rejected}")
    arguments: Final[OcrArguments] = {
        "model": "mistral/mistral-ocr-latest",
        "document": {
            "type": "document_url",
            "document_url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        },
        "api_key": "invalid-smoke-key" if rejected else os.environ["MISTRAL_API_KEY"],
        "callbacks": [recorder],
        "api_base": "https://api.mistral.ai/v1",
        "num_retries": 0,
        "timeout": 60,
    }
    response, error = await call_ocr(arguments, asynchronous)
    assert (error is not None) is rejected
    if error is not None and not baseline:
        assert isinstance(error, litellm.AuthenticationError) and error.status_code == 401
    if response is not None:
        assert len(response.pages) == 1
    outcome: Final = (
        f"pages={len(response.pages)}"
        if response is not None
        else f"{type(error).__name__} status={getattr(error, 'status_code', None)}"
    )
    events: Final = await recorder.wait()
    names: Final = tuple(event.name for event in events)
    if rejected:
        assert names[0] == "pre" and sorted(names[1:]) == ["async_failure", "failure"]
    else:
        assert names == ("pre", *(() if baseline else ("post",)), "async_success" if asynchronous else "success")
    if not baseline:
        assert all(event.native_provider_hook for event in events if event.name in ("pre", "post"))
    sys.stdout.write(f"async={asynchronous} rejected={rejected} {outcome} events={names}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    if "--run-live" not in sys.argv:
        raise SystemExit("Pass --run-live and set MISTRAL_API_KEY to make real, billable provider calls")
    if "--baseline" in sys.argv:
        asyncio.run(smoke(baseline=True))
    else:
        native: Final = get_native_bridge()
        assert native is not None
        with patch.object(native, "ready_endpoints", {"ocr": frozenset({"callbacks"})}):
            asyncio.run(smoke(baseline=False))
