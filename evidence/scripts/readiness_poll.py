# Bounded readiness polling for live service and cache signals (no litellm import, used by the proxy driver).
import time

def poll(check, what, timeout=30.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while True:
        value = check()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError("not ready: " + what)
        time.sleep(interval)
