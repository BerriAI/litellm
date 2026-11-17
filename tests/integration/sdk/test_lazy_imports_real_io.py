import os
import subprocess
import sys
from typing import Final

import litellm
import litellm.utils


def test_import_litellm_does_not_load_fastapi_or_bpe_table():
    result: Final = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, litellm; print(','.join(m for m in ('fastapi','starlette','litellm.proxy.proxy_cli','litellm.litellm_core_utils.default_encoding','openai.types.realtime') if m in sys.modules))",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )

    assert result.stdout.strip() == ""
