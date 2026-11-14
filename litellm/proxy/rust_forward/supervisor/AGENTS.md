# Forwarding supervisor

`supervise` serves the public port from Rust and runs the entry point's own command behind it, or says why that cannot start. That command is the supervisor's own interpreter command line (`OWN_COMMAND`, from `sys.orig_argv`), so the child has the same import path and interpreter flags as the process the operator started. An entry point never builds a second way to start itself. It is transparent to Python, so a supervised proxy behaves on signals and exits exactly as Python alone. `tests/integration/configuration/test_rust_forward_lifecycle.py` checks that equivalence against Python alone, and a change that breaks one of the rules below breaks it

## Split by job

- `preflight` decides whether forwarding can start. It is the only place that refuses, and a refusal is a returned `StartupError`, never an exit, so the entry point can serve from Python alone
- `children` says what each child runs with, and it is the only writer of both environment contracts. `python_spec` writes the `mode.Supervised` variable and hands the Python child the entry point's own arguments with the private port appended. `rust_spec` writes the `LITELLM_FORWARD_*` variables. Each child gets only those variables on top of the supervisor's own environment
- `lifecycle` decides. `step` is a pure function from a `Lifecycle` of facts and an event to the next facts and the actions to run, and `exit_status` reads the result from those facts. It touches no process, socket or clock
- `runtime` acts. It only executes the actions `step` returns and feeds events back. The `contextlib.AsyncExitStack` it enters each child into owns their cleanup, last spawned first

A new behavior is a new fact, event or action in `lifecycle` with a property test in `test_lifecycle.py`, never a condition in `runtime`

## Rules

- Signals pass through. Every signal in `Platform.relayed_signals` goes to Python unchanged, and the ones in `Platform.stop_signals` go to Rust first. The supervisor never reads meaning into a signal, so a repeated SIGINT reaches Python again and forces its exit as it would alone
- Python's exit status is the result. When Python exits for any reason, Rust gets SIGTERM, and once Rust is gone the supervisor exits with Python's status, dying by the same signal when Python died by one. The one exception is Rust exiting while Python serves with no stop requested, which stops Python and exits 1
- A container's PID 1 cannot die by a signal it sends itself, since the kernel drops it. uvicorn alone survives its own re-raise there and exits 0, so `runtime.exit_like` exits 0 when a stop signal killed Python and with 128 plus the signal number for any other signal, which is how a kill of Python alone is reported
- Children cannot outlive the supervisor. Each child's stdin is a pipe only the supervisor holds open. The Python process the supervisor started (`mode.stop_with_supervisor`, never its workers) and the Rust gateway (`litellm_gateway_forward::input_closed`) shut down gracefully on EOF, so a supervisor killed by SIGKILL or a closed terminal leaves nothing behind
- Rust starts once Python listens, never after a stop was requested or Python exited. Python alone takes no connections until it has booted, so the public port refuses them until then too
- Only a preflight refusal falls back to Python alone. `before_start` runs once forwarding is known to be possible, so a refused start leaves nothing behind, and a child that fails to spawn after that exits 1
