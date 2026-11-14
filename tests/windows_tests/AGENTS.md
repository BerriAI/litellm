# Windows tests

The `windows-sdk` job in `.github/workflows/test-merge-smoke.yml` is the only place these run, on a Windows runner with no proxy dependencies installed. They prove the SDK works on Windows without them, so a test here must not import uvicorn, fastapi, psutil or anything else outside the `dev` group, and must not need the `litellm-gateway.exe` binary

A test that cannot run off Windows carries a `skipif` on `sys.platform`, so collecting this folder on another OS never fails

Windows-only names such as `CTRL_BREAK_EVENT` come from `litellm/proxy/rust_forward/platform/windows.py`, which spells them out so the type checker accepts them on every OS. `test_rust_forward_platform.py` pins those constants to the stdlib
