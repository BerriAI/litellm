import subprocess
import sys
from pathlib import Path
from typing import Final


def test_parallel_first_usage_construction_does_not_race_schema_build() -> None:
    probe: Final = r'''
import inspect
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, sys.argv[1])
from pydantic import BaseModel
from litellm.types.utils import Usage, PromptTokensDetailsWrapper, CompletionTokensDetailsWrapper

source, first_line = inspect.getsourcelines(BaseModel.model_rebuild)
delete_line = next((first_line + i for i, line in enumerate(source) if "delattr(cls, attr)" in line), None)
sys.setswitchinterval(1e-6)

for model in (Usage, PromptTokensDetailsWrapper, CompletionTokensDetailsWrapper):
    barrier = threading.Barrier(8, timeout=5)
    start_barrier = threading.Barrier(8, timeout=5)
    def trace(frame, event, arg):
        if (event == "line" and frame.f_code is BaseModel.model_rebuild.__func__.__code__
            and delete_line is not None and frame.f_lineno == delete_line and frame.f_locals.get("cls") is model
            and frame.f_locals.get("attr") == "__pydantic_core_schema__"):
            barrier.wait()
        return trace

    def construct(index):
        sys.settrace(trace)
        try:
            start_barrier.wait()
            return model().model_dump()
        finally:
            sys.settrace(None)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(construct, range(8)))
    assert len(results) == 8
    assert all(result == results[0] for result in results)
    print(model.__name__ + ": 8 parallel constructions passed")
'''
    root: Final = str(Path(__file__).resolve().parents[3])
    result: Final = subprocess.run(
        [sys.executable, "-I", "-c", probe, root], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
