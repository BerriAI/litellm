`litellm-host` defines typed calls, execution hooks, host services, stream delivery and the resumable machine. HTTP and Python drivers interpret the same suspension protocol in their own runtimes

| Responsibility | Shared contract | HTTP | Python |
| --- | --- | --- | --- |
| Input and output conversion | `Protocol::{Request, Response, StreamHead, Chunk, Error}` | Typed input; `ResponseEncoder` and `StreamEncoder` produce HTTP values | `PythonBinding` decodes prepared arguments, encodes public values and maps native errors |
| Host services | `Protocol::HostCall`, `HostServices::call` | `HostCallHandler` answers typed calls; `()` handles protocols without host calls | `PythonHostCalls` invokes retained Python objects; it may share an owner with the binding |
| Active hooks | `RouteHooks::{before_provider_request, on_event}` | Request interception and fallible execution callbacks | `PythonCallHooks` also prepares arguments, transforms public responses and receives stream callbacks |
| Passive observation | `lifecycle::CallObserver` | Start and terminal observation, retained by the response body | Public Python callbacks remain active hooks with their existing failure policy |
| Runtime driving | `Machine`, `HostRequest::{HostCall, Hook, Stream}` | Body polling controls demand | Native polling and inline caller-task Python awaits control progress |

`hosted_call(request, execute)` starts with a typed request. Its route closure receives separate `HostServices` and `ChannelHooks`; it returns `CallOutput`. Hosted-call plumbing alone forwards the returned stream through demand replies. Lower-level `CallMachine` users receive a `CallContext` containing separately named services, hooks and stream delivery

Core route constructors prepare their dependencies and return a closure accepting the typed request. Python starts that closure only after argument preparation, preflight and decoding succeed. These steps remain inside the driver's terminal and error handling. Decoding may retain objects for subsequent host service calls

Each driver owns terminal dispatch. Hooks can change or fail execution; passive observers return no result. HTTP observes success after response conversion or stream exhaustion, failure on errors, and cancellation on body drop. Python preserves exception identity and maps native failures once. Explicit Python stream close reports success for delivered chunks; cancellation stops further callback dispatch

Rust handlers answer suspensions through `litellm-host-native::Driver`, which `litellm-host-http` and `litellm_host_native::in_process` share. `in_process::Host` is an assembly of services, hooks, stream consumer and optional observer. It is not a trait mirroring every suspension. Use `run_hosted` to preserve the distinction between stream completion and detachment

Keep API policy in gateway-inference and python-bridge, and legacy callback policy in callbacks-legacy-python. Python bindings and hooks expose retained references through `PythonOwned`, with idempotent close and GC traversal. Runtime machinery stays in driver, native, handle and runtime modules
