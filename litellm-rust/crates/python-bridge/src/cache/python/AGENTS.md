# Python cache delegation

This directory lets Rust inference use a selected Python cache. `service.rs` implements the injected Rust response-cache service and yields typed cache operations. `host.rs` calls the Python cache's sync or async API and delivers the result back to Rust. `callback.rs` provides Python cache delegation for the Python-facing cache runtime

Keep the shared inference protocol wrapper and adapter selection in the parent module. Receive the configured cache and prepared arguments from the parent module. Do not discover global configuration, choose native backends, or move inference to Python. Core remains independent of Python objects and cache implementation details

Await asynchronous cache operations through the existing host driver in the caller's task. Do not create another asyncio task or event loop. Cancellation must prevent subsequent provider requests and cache writes. Preserve ordinary cache failure handling without swallowing cancellation or other Python base exceptions

Traverse retained Python references for GC and release pending replies when the call closes. Regression tests must require native inference with Python fallback disabled and assert observable hits, provider request counts, cache-key headers, task identity and cancellation
