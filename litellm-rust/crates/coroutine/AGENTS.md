# Requirements

Core must pause mid-call to ask the host for things it cannot do itself (Python callbacks, secret and token reads, `before_send` rewrites, stream demand), then continue where it stopped. Any change to this crate must keep every requirement below. The alternatives section says which one each rejected design breaks

- R1 Core never calls the host: it names an op and waits for the answer, so it stays free of PyO3 and of any other host runtime
- R2 Async host work is awaited by the host's own driver in the caller's asyncio task (`litellm/rust_bridge/lifecycle.py` driving `host-python/src/task.rs`), so every hook of one call runs in one task and one `contextvars` context, and a write in one hook is read by the next and by the caller. Nothing in litellm's own callbacks depends on that today; it is kept because it costs nothing here and user callbacks may rely on it. A Rust-side `pyo3_async_runtimes::into_future` schedules a new task per awaitable from a context snapshot, adds a loop turn per hook, and does not cancel the Python task when its future is dropped
- R3 The body awaits real I/O (HTTP, `spawn_blocking`, timers) between yields, so `resume` is itself a future driven by the caller's runtime
- R4 Route code stays straight-line async (`host.route(OcrOp::ReadDocument).await?`) instead of hand-written states
- R5 Each op fixes its answer type at compile time: a host cannot answer `ReadDocument` with a token, and core never matches a result variant it did not ask for
- R6 A yield the body makes while being resumed is returned by that same poll, so the host driver's inline first poll needs no extra event-loop turn per op
- R7 No task is spawned: `cancel`, or dropping the coroutine, drops the body, and nothing waits forever on an answer that cannot come
- R8 Several yields can be pending at once, since route code hands clones of its `Co` to token providers and hooks
- R9 Stable Rust

# Other implementations and why they do not fit

- Nightly `std::ops::Coroutine`: breaks R9, and its body cannot await futures between yields (R3)
- `genawaiter`: resumes async bodies only with a noop waker, so the body cannot await real I/O (R3)
- `simple_coro`: typestate `Coro` makes answering before resuming a compile-time rule, but its body cannot await arbitrary futures (R3) and its reply type `R` is fixed per coroutine (R5)
- `corosensei` and other stackful coroutines: sync bodies on their own stack, no async I/O inside (R3)
- A hand-written phase enum with an `advance` match (the old `HostPhase`): every await point becomes a state (R4)
- An injected host trait with `async fn`s awaited from core on tokio: core would call the host itself (R1, R2). The host side does use `async fn` hooks, but they are polled from the caller's task by the Python driver, on top of this crate's yields, not instead of them
- `pyo3_async_runtimes::into_future` as the Python boundary: a new Python task per awaitable in a copied context (R2), and dropping the Rust future leaves the Python task running (R7)
- Sans-IO, where core does no I/O and HTTP becomes one more host op: keeps every requirement and makes `resume` a pure step function, but HTTP, streaming, retries and timeouts would move out of core into every bridge. The one real alternative, not taken
- Temporal's Rust workflow SDK (`WorkflowFuture`, `WfContext`) is the closest precedent: an `async fn` polled in place, commands sent over a channel with a oneshot to unblock them. Roles are inverted there (the language SDK owns the program, core answers), and its workflow body may not do real I/O

# Tradeoffs accepted

- A tokio `mpsc` channel plus a `oneshot` per yield instead of compiler-generated states
- Protocol mistakes (resuming before answering, resuming after the end) are runtime `ResumeError`s, not compile errors
- Pending yields come out one per `resume`, in the order they were made, and each reply goes back to the yield that made it (R8)
- An answer sent after its yield stopped waiting (for example the body timed out on it) is discarded, since the body already moved on
