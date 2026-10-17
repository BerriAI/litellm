# Python cache delegation

This directory lets Rust inference use the configured Python cache. `service.rs` owns the typed protocol and injected response-cache service. `binding.rs` wraps the route binding, asks `litellm.rust_bridge.host.cache` for access and delivers lookup and store results back to Rust. Keep PyO3 conversion, suspension and Python references in the binding, and cache rules in the Python host module

Every call into Python goes through that host module. Do not read `litellm.cache`, call cache methods directly, or re-derive which calls may read or write. Core remains independent of Python objects and cache implementation details

Preparing the Python adapter must not invoke the lazy native identity builder or create a native cache request or key. Keep the host's hit key unchanged. Python owns key construction, lookup and storage; Rust transports operations and converts responses

Await asynchronous cache lookups through the existing host driver in the caller's task. Do not create another asyncio task or event loop for them. Asynchronous stores are handed to Python's `create_cache_write_task`, so the response returns before the write lands, as on the Python route. Cancellation during a lookup must prevent the provider request and the cache write. A failure to reach Python is a cache failure, never a request failure. Preserve ordinary cache failure handling without swallowing cancellation or other Python base exceptions

Traverse retained Python references for GC and release pending replies when the call closes. Regression tests must require native inference with Python fallback disabled and assert observable hits, provider request counts, cache-key headers, task identity and cancellation
