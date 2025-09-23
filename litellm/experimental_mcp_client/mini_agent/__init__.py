# Minimal namespace for mini-agent helpers used in smokes.
# Ensure an event loop exists for tests that call asyncio.get_event_loop().run_until_complete(...)
import asyncio  # noqa: E402

# Normalize policy first (helps Python 3.12+ behavior)
try:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except Exception:
    pass

try:
    asyncio.get_running_loop()
except RuntimeError:
    try:
        _loop_pkg = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop_pkg)
    except Exception:
        # Best-effort; test runners may manage the loop differently
        pass

# Final guard for get_event_loop() callers
try:
    asyncio.get_event_loop()
except RuntimeError:
    try:
        _loop_pkg2 = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop_pkg2)
    except Exception:
        pass
