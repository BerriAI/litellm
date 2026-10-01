Own the HTTP driver for hosted calls, including response-body demand, cancellation, and lifecycle observation

`serve` and `serve_unary` receive an optional `ObservationSender` separately from active hooks. Observation works with `()` hooks and covers response conversion and body delivery

Keep endpoint paths, request parsing, deployment selection, and API-specific response and error formats in gateway-inference

Depend on the neutral host protocol, never on core routes, Python, or gateway crates

Do not spawn producer tasks or buffer chunks ahead of HTTP body demand
