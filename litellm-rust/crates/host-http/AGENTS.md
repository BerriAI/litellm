Own the HTTP driver for hosted calls, including response-body demand, cancellation, and lifecycle observation

Keep endpoint paths, request parsing, deployment selection, and API-specific response and error formats in gateway-inference

Depend on the neutral host protocol, never on core routes, Python, or gateway crates

Do not spawn producer tasks or buffer chunks ahead of HTTP body demand
