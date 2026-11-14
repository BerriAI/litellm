# Platforms

Every OS-specific operation sits behind `Platform`, which `supervise` picks once with `current_platform()`. The supervisor, `mode.py` and the entry points never check the OS themselves, so a new OS difference becomes a `Platform` member and never an `os.name` check somewhere else

A `Platform` answers these questions and nothing else. What the gateway binary is called, which servers it can supervise, how the private listener reaches Python, which spawn options the children get, which signals are relayed and which of them stop Rust, and how a child is signalled and stopped

## POSIX

`PosixPlatform` binds the private socket itself and hands Python the open descriptor (`server_listener.InheritedFd`). Nobody else can take the port between the supervisor picking it and Python serving it. The socket does not listen until Python does, so the readiness probe only succeeds against Python. Children run in their own session (`start_new_session`) only so that a terminal Ctrl-C reaches Python once, through the supervisor

## Windows

Windows is supported for running the proxy locally from the CLI. Nobody deploys the proxy on Windows, but `LITELLM_RUST_FIRST` is meant to become the default, and then this is the path every Windows laptop runs. Revisit the choices below before treating Windows as a deployment target

- Python gets an assigned loopback port (`server_listener.AssignedPort`) instead of an open socket, which uvicorn on Windows cannot serve. Another process can take that port before Python binds it, and the readiness probe would then accept it as Python. Python's failed bind still ends the run
- Each child runs in its own process group, which can only be sent `CTRL_BREAK_EVENT`. Both relayed signals arrive as that, so a second Ctrl-C does not force Python's exit the way it does on POSIX
- gunicorn is refused, since it does not run on Windows
- The Windows-only names are spelled out as constants in `windows.py` so the module imports and type checks on every OS. `tests/windows_tests/test_rust_forward_platform.py` pins them to the stdlib on the Windows runner
- No CI job runs the supervisor itself on Windows, because the `windows-sdk` job installs no proxy dependencies and no `litellm-gateway.exe`. Run a change to `windows.py` by hand on Windows until one does
