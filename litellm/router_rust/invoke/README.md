# invoke

The host side of `Host::invoke`: running one attempt against the one deployment router-core picked. Nothing here selects, retries, falls back or records cooldowns.

- `shaping`: the kwargs an attempt sends to its deployment
- `endpoints`: what each public router method does before, around and after the routed call
- `streaming`: stream wrappers, so a stream that fails before its first token fails the attempt
- `slots`: a deployment's `max_parallel_requests`
- `walk`: the retry and fallback metadata stamped on each attempt
- `prompt_management`: prompt-management groups
- `websocket`: admission for each turn of a Responses websocket

A Rust host replaces this with native endpoint calls.
