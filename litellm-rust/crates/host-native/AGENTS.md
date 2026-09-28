`litellm-host-native` is the Rust driver for hosted calls. `Driver` owns the machine, a `HostCallHandler` and a `RouteHooks`; `advance()` answers services and hooks inline and returns at completion or at the next stream boundary, holding the `Reply<Demand>` until the consumer calls `advance()` or `detach()` again. Dropping the driver drops the machine and so cancels the call

The consumer decides demand, so the driver never spawns a producer task and never buffers chunks ahead of demand. `litellm-host-http` polls it from the response body; `in_process::run_hosted` polls it on behalf of a `StreamConsumer`. Both observe lifecycle terminals themselves, the driver reports none

Depend on `litellm-host` only. HTTP encoding stays in `litellm-host-http`; `litellm-host-python` drives the machine directly so Python callbacks stay in the caller's asyncio task
