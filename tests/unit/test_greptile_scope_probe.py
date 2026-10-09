import subprocess
import time


def test_echo_finishes_quickly():
    started = time.monotonic()
    result = subprocess.run(["echo", "hi"], capture_output=True, text=True, check=True)
    assert result.stdout == "hi\n"
    assert time.monotonic() - started < 5
