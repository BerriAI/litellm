# Python cache delegation

This directory lets Rust inference use the configured Python cache. `service.rs` implements the injected Rust response-cache service and yields typed cache operations. `host.rs` asks `litellm.rust_bridge.host.cache` for access and calls its lookup and store functions, sync or async, delivering the result back to Rust

Every call into Python goes through that host module. Do not read `litellm.cache`, call cache methods directly, or re-derive which calls may read or write. Core remains independent of Python objects and cache implementation details

Await asynchronous cache operations through the existing host driver in the caller's task. Do not create another asyncio task or event loop. Cancellation must prevent subsequent provider requests and cache writes. Preserve ordinary cache failure handling without swallowing cancellation or other Python base exceptions

Traverse retained Python references for GC and release pending replies when the call closes. Regression tests must require native inference with Python fallback disabled and assert observable hits, provider request counts, cache-key headers, task identity and cancellation
