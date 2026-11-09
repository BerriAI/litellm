import os
import subprocess
import sys
import threading
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import litellm
from integration._support.wire import Reply, Request, wire_server

_SCRIPT_SECONDS: Final = 60
_CONCURRENT_LLAMA_COUNTS: Final = """
import sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, sys.argv[1])
from litellm.utils import _select_tokenizer_helper
with ThreadPoolExecutor(max_workers=2) as pool:
    tuple(pool.map(lambda _: _select_tokenizer_helper("meta-llama/llama-3-8b"), range(2)))
"""


def test_concurrent_first_llama_counts_download_without_waiting_on_each_other(tmp_path: Path) -> None:
    both_downloading: Final = threading.Barrier(2, timeout=5)
    downloads: Final = SimpleQueue[str]()

    def stalled_hub(request: Request) -> Reply:
        if request.target.endswith("/tokenizer.json"):
            try:
                both_downloading.wait()
                downloads.put("concurrent")
            except threading.BrokenBarrierError:
                downloads.put("serialized")
        return Reply(status=404)

    with wire_server(stalled_hub) as hub:
        result: Final = subprocess.run(
            [sys.executable, "-I", "-c", _CONCURRENT_LLAMA_COUNTS, str(Path(litellm.__file__).parent.parent)],
            capture_output=True,
            text=True,
            timeout=_SCRIPT_SECONDS,
            check=False,
            env={
                **os.environ,
                "HF_ENDPOINT": hub.url,
                "HF_HOME": str(tmp_path / "home"),
                "HF_HUB_CACHE": str(tmp_path / "cache"),
                "HF_HUB_OFFLINE": "0",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            },
        )

    assert result.returncode == 0, result.stdout + result.stderr
    assert downloads.get_nowait() == "concurrent"
