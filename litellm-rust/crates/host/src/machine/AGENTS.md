# Resumable execution

`Machine` is the driver-facing contract for a resumable execution. `CallMachine` implements that contract with `litellm_coroutine::Coroutine`, holding the async execution of a route invocation. Resuming continues that same execution, which may request host services, invoke hooks, and deliver many stream chunks

## File ownership

| File | Responsibility |
| --- | --- |
| `mod.rs` | Module declarations and public exports |
| `contract.rs` | `Machine`, its step and interruption futures, and host failure values |
| `coroutine.rs` | `CallMachine`, coroutine state conversion, execution futures, and machine faults |
| `context.rs` | The route's services, hooks, and stream handles, sharing one coroutine channel |

Keep the contract independent of the coroutine implementation. Drivers and wrappers can implement `Machine` without constructing a coroutine. Keep existing public imports through `litellm_host::machine` stable when reorganizing private modules

Credential acquisition contracts and reusable adapters belong in `litellm-auth-types`. A route can use `TokenProviderHandle::from_callback` to request a credential through `HostServices::call`. Keep authentication policy and token-specific traits out of the machine layer

## Execution and replies

The coroutine polls the route future until it completes or yields a `Suspension`. Each suspension carries a typed `Reply` that its driver must answer before resuming, or abandon when interrupting or dropping the execution. A pending network future is an ordinary async wait, not a host suspension

`CallContext` gives the route three separate handles: `HostServices` requests host operations, `ChannelHooks` requests active hooks, and `StreamSender` delivers stream values. Keep their yield-and-reply mechanics in `context.rs`. The actual service and hook implementations belong to the host

`Demand::More` permits stream execution to continue. `Demand::Detached` tells it that the consumer stopped reading. Keep stream forwarding and the distinction between stream exhaustion and detachment in `crate::call::hosted_call`

`CallMachine::interrupt` cancels the coroutine and returns the supplied failure. Dropping `CallMachine` drops its execution future. Preserve both behaviors and do not spawn a producer task or poll stream chunks ahead of consumer demand

## Host responsibilities

Rust drivers live in `litellm-host-native` and `litellm-host-http`. The Python driver lives in `litellm-host-python` and awaits Python hooks in the caller's task. Keep runtime scheduling, encoding, terminal observation, and callback policy in those layers and their adapters

Public behavior tests belong in `crates/host/tests`; private behavior tests stay inline with their owning implementation. Test suspension answers, interruption, resource release, and stream demand through behavior, rather than asserting file layout
