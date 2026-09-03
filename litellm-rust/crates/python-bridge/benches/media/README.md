# Media request memory benchmark

The refactor keeps existing encoded media in shared buffers through Python extraction, provider transformation and outgoing chunks. Raw audio is encoded in bounded chunks. The benchmark includes request preparation, AWS signing and transmission to a local HTTP sink that hashes each incoming chunk without collecting the body

## Method

Run on macOS arm64 with Rust 1.98.0 and CPython 3.11, release profile, on 2026-09-02. Each scenario uses distinct Python inputs of 1, 16 or 64 MiB at concurrency 1 or 16. Inputs stay alive until all requests finish. Encoded sizes describe existing ASCII base64 strings; raw sizes describe unencoded bytes and produce about 4/3 as many wire bytes

`python` uses Python base64, JSON, botocore signing and concurrent `http.client` calls. It constructs the same Bedrock transcription body as Rust, without the rest of the Python SDK. `buffered` extracts through the existing Python-to-Serde boundary, runs the Rust provider transform and allocates the full outgoing JSON body. `current` is the pre-refactor shared-body implementation saved before this pass. `refactor` uses ByteString, the smaller body module and the explicit transport client

Rust cases call the actual bridge media extractor and Bedrock transcription transform. They use fixed test credentials and signing time. Python uses botocore with the same test credentials and the current signing time. All approaches hash the payload once for signing, then the sink independently hashes received bytes. All four produce identical body lengths and hashes for each scenario

Timing, RSS and allocation measurements run in separate fresh processes. CPU and throughput are medians of three sequential timing runs, interleaved by approach, after builds and test processes finished. CPU includes both the client and local sink threads. Throughput includes extraction, transforms, preparation, signing and sending; it is not remote-provider throughput. Per-stage times are wall-clock medians

RSS includes the interpreter, retained inputs, Rust runtime, HTTP client and sink. `input_rss_mib` is the process high-water RSS after input setup, not live RSS. The Rust allocator counter measures cumulative requested bytes, including reallocations, in the client and sink after setup. Python allocations are measured separately with tracemalloc's peak live traced bytes; these two allocation columns are different quantities and must not be added. Native allocation totals are not a peak-memory metric: bounded encoding can allocate many successive chunks while keeping only a few live

No TLS, remote download, guardrail materialization, retries or response payload optimization is included in these timings. Local transport tests separately cover redirects, retries, disconnects and stalled-consumer cancellation. This benchmark is a focused body pipeline comparison, not a complete LiteLLM deployment memory profile

## Results

At 64 MiB per input and concurrency 16, the retained inputs alone total 1024 MiB

| Input | Approach | Peak RSS, MiB | CPU seconds | Wire MiB/s |
|---|---|---:|---:|---:|
| encoded | python | 2150.9 | 6.14 | 350.2 |
| encoded | buffered | 3111.5 | 6.18 | 283.4 |
| encoded | current | 1062.8 | 8.34 | 225.0 |
| encoded | refactor | 1063.3 | 7.44 | 247.9 |
| raw | python | 3879.5 | 9.40 | 267.9 |
| raw | buffered | 3879.7 | 9.64 | 237.9 |
| raw | current | 1077.1 | 8.96 | 312.6 |
| raw | refactor | 1081.0 | 9.46 | 279.1 |

The full matrix, allocation measurements, stage timings and body hashes are in [results.csv](results.csv)

The refactor counts its emitted chunks to establish Content-Length. Raw media is therefore encoded during length calculation, signing and transmission, one more pass than the pre-refactor implementation. Existing encoded media uses shared slices for all three passes. Encoding and escaping chunks remain at most 64 KiB; their live memory does not scale with the payload

## Reproduction

The existing serialization benchmark also supports one-shot media measurements. The embedded Python interpreter needs botocore and typing_extensions. Set `PYO3_PYTHON` at build time and, when using a virtual environment, make its site-packages available through `PYTHONPATH`

```bash
cd litellm-rust
cargo bench -p litellm-python-bridge --bench serialization -- --media refactor 64 16 raw memory
cargo bench -p litellm-python-bridge --bench serialization -- --media refactor 64 16 raw allocation
cargo bench -p litellm-python-bridge --bench serialization -- --media refactor 64 16 raw timing
```

Approaches are `python`, `buffered`, or a shared-body label such as `refactor`. Input modes are `encoded` and `raw`. Repeat for 1, 16 and 64 MiB, and concurrency 1 and 16. For timing comparisons, build first, identify the executable from Cargo's `compiler-artifact` output, and invoke it directly in fresh processes. Record the executable hash when comparing different work-in-progress versions

The pre-refactor executable was captured from the uncommitted implementation based on staging `e058aa68c4`, before the ByteString and typed-extraction changes. Its source snapshot remained separate while the branch was refactored in place. Executable SHA-256 values used for this report:

Pre-refactor Rust: `4874abe92aeb9e1ed77bef6668c11f4ca545458c2bc189a18404ad61cd804327`

Refactor and Python: `c551eaef47e7a7eceecfed8b4ab76a42d84b8392342be01e8a22f25cce658287`
