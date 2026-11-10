`litellm-host` defines typed calls, execution hooks, host services, stream delivery and the resumable machine. HTTP and Python drivers interpret the same suspension protocol in their own runtimes

| Responsibility | Shared contract | HTTP | Python |
| --- | --- | --- | --- |
| Input and output conversion | `Protocol::{Request, Response, StreamHead, Chunk, Error}` | Typed input. `ResponseEncoder` and `StreamEncoder` produce HTTP values | `PythonBinding` decodes prepared arguments, encodes public values and maps native errors |
| Host services | `Protocol::HostCall`, `HostServices::call` | `host-native::services::HostCallHandler` answers typed calls. `()` handles protocols without host calls | `PythonHostCalls` invokes retained Python objects. It may share an owner with the binding |
| Active hooks | `Interceptors::{before_provider_request, after_provider_response}` and `hooks::CallHooks<Runtime>` | Request interception and fallible execution callbacks | `PythonCallHooks` also prepares arguments, transforms public responses and receives stream callbacks |
| Passive observation | `observation::ObservationSender` | Queued execution and lifecycle snapshots, retained by the response body | Public Python callbacks remain active hooks with their existing failure policy |
| Runtime driving | `Machine`, `HostRequest::{HostCall, Intercept, Stream}` | Body polling controls demand | Native polling and inline caller-task Python awaits control progress |

`hosted_call(request, observers, execute)` starts with a typed request. Its route closure receives separate `HostServices`, `ChannelInterceptors` and optional observation publisher. It returns `CallOutput`. Hosted-call plumbing alone forwards the returned stream through demand replies. Lower-level `CallMachine` users receive a `CallContext` containing separately named services, interceptors, observers and stream delivery

Core route constructors prepare their dependencies and return a closure accepting the typed request. Python starts that closure only after argument preparation, preflight and decoding succeed. These steps remain inside the driver's terminal and error handling. Decoding may retain objects for subsequent host service calls

Each driver owns terminal dispatch. Hooks can change or fail execution. Passive observers return no result. HTTP observes success after response conversion or stream exhaustion, failure on errors, and cancellation on body drop. Python preserves exception identity and maps native failures once. Explicit Python stream close reports success for delivered chunks. Cancellation stops further callback dispatch

`interceptors.rs` owns `Interceptors` and its request/response payload types. `lifecycle.rs` owns `CallObserver`, `CallEvent`, `ExecutionEvent`, timing, failure origin, and the observation wrappers. Event payloads are generic so a runtime can retain its own response, exception and raw-response references without introducing a language dependency. `snapshot()` projects them into the owned observation contract without retaining runtime objects. Pass interceptors and observers separately at direct route and HTTP entrypoints. Routes publish execution events independently of interception

`hooks.rs` owns the call-stage interface and its runtime-associated types. It contains no Python types or legacy callback policy. A runtime supplies its context and continuation representation through `HookRuntime`

`protocol.rs` owns `Protocol` and suspension messages, including `InterceptRequest` and `StreamDelivery`. `call.rs` owns route outputs and their adaptation into a hosted machine. Rust service handling belongs in `host-native::services`. Coroutine channel handles stay in `machine/context.rs`

Rust handlers answer suspensions through `litellm-host-native::Driver`, which `litellm-host-http` and `litellm_host_native::in_process` share. `in_process::Host` is an assembly of services, interceptors, stream consumer and optional observation publisher. It is not a trait mirroring every suspension. Use `run_hosted` to preserve the distinction between stream completion and detachment

Keep API policy in gateway-inference and python-bridge, and legacy callback policy in callbacks-legacy-python. Python bindings and hooks expose retained references through `PythonOwned`, with idempotent close and GC traversal. Runtime machinery stays in driver, native, handle and runtime modules

Interceptors run inline and can rewrite values or fail execution. Observers consume owned `CallEvent` snapshots from `observation_channel`. Its bounded `ObservationSender` never waits for delivery and counts events dropped when the queue is full or closed. The host owns receiver processing and draining. Pass the same publisher to machine construction and the driver when one receiver should collect execution and lifecycle events. Legacy Python callbacks retain their existing awaited, fallible behavior through the Python adapter

`ExecutionFacts` and `ResultSource` describe execution without pricing or budget policy. `Interceptors::result_ready` delivers these facts through an awaited `InterceptRequest::ResultReady`. Hosts receive them before response transformation or stream delivery. `ExecutionEvent::ResultReady` is the matching lifecycle event and can also be published as a passive snapshot. Accounting must consume the awaited path rather than a lossy observation queue
