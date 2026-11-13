# invoke

The host side of `Host::invoke`, running one attempt against the deployment router-core picked. Routing retries and fallbacks belong to router-core. The driver can retry an eligible flex failure once on the same deployment at the default tier when `service_tier_fallbacks` is enabled.

- `shaping`: the kwargs an attempt sends to its deployment
- `endpoints`: what each public router method does before, around and after the routed call
- `streaming`: stream wrappers, so a stream that fails before its first token fails the attempt
- `slots`: a deployment's `max_parallel_requests`
- `walk`: the retry and fallback metadata stamped on each attempt
- `prompt_management`: prompt-management groups
- `websocket`: admission for each turn of a Responses websocket

A Rust host replaces this with native endpoint calls.
