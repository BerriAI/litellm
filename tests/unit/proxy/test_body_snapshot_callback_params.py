"""The stored request body never carries callback parameters.

Every ``StandardCallbackDynamicParams`` key and ``litellm_trusted_callback_vars`` is set on the
request dict with a unique value, the body snapshot is refreshed, and none of the keys or values
may be in ``proxy_server_request["body"]``. A control key proves the snapshot was rebuilt.
"""

from __future__ import annotations

import json
import uuid
from typing import Final

from litellm.proxy.litellm_pre_call_utils import refresh_proxy_server_request_body_snapshot
from litellm.types.utils import TRUSTED_CALLBACK_VARS_FIELD, StandardCallbackDynamicParams


def test_body_snapshot_excludes_every_callback_dynamic_param_and_the_trusted_vars() -> None:
    core: Final = uuid.uuid4().hex
    params: Final = {name: f"lkc-{name}-{core}" for name in StandardCallbackDynamicParams.__annotations__}
    control: Final = f"control-{uuid.uuid4().hex}"
    data: Final = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": control}],
        **params,
        TRUSTED_CALLBACK_VARS_FIELD: dict(params),
        "proxy_server_request": {"url": "http://proxy/v1/chat/completions", "body": {}},
    }

    refresh_proxy_server_request_body_snapshot(data)

    body: Final = data["proxy_server_request"]["body"]
    assert control in json.dumps(body), "Sensitivity control: the snapshot was not rebuilt from the request"
    present: Final = sorted({*params, TRUSTED_CALLBACK_VARS_FIELD} & set(body))
    assert present == [], f"Callback parameters copied into the stored request body: {present}"
    assert core not in json.dumps(body, default=str)
