# Monty security boundary

Never pass host callbacks, mounts, environment variables, credentials, or live SDK objects into Monty

Never import, exec, or eval uploaded source on the gateway. Inspect it with AST only

Never fall back to CPython execution. Missing or unverified workers fail closed

Monty's max_memory is an allocator budget, not an RSS limit. Apply the worker address-space limit separately

Language isolation is not an OS security boundary. An interpreter escape requires separate OS-level containment
