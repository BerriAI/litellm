Own the HTTP driver for hosted calls, including response-body demand, cancellation, and lifecycle observation

`serve` and `serve_unary` receive an optional `ObservationSender` separately from active hooks. Observation works with `()` hooks and covers response conversion and body delivery. `run` and `run_unary` return the typed result for a caller that renders later, so their observation covers the call and a stream's delivery but not the conversion

Keep endpoint paths, request parsing, deployment selection, and API-specific response and error formats in gateway-inference

Depend on the neutral host protocol, never on core routes, Python, or gateway crates

Do not spawn producer tasks or buffer chunks ahead of HTTP body demand
