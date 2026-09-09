# Initial handoff comparison

Sequential `Vec` baseline and `Bytes` comparison on the same machine, using CPython 3.12.13 with the GIL enabled, Rust 1.98.0, and target `aarch64-apple-darwin`. Both used `pyo3/abi3-py310` without `extension-module`, 20 samples, 1 second warmup, and 3 seconds requested measurement time per case

The baseline was captured after extracting the production reader and Python adapter, while the synthetic source still copied each `Bytes` into a `Vec` during polling. The comparison moved the original `Bytes` instead. Stream construction and payload setup were excluded from both measurements

All numbers below use Criterion mean estimates. Each stream contains 256 chunks. These are handoff overhead measurements from one local experiment, with no allocation instrumentation or provider traffic

Reader byte throughput counts bytes represented by handles without scanning their contents, so it does not measure memory bandwidth

| Workload | Bytes/chunk | Vec µs/stream | Bytes µs/stream | Change | Bytes ns/chunk | Chunks/s | MiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| reader/ready | 64 | 14.79 | 12.31 | -16.8% | 48.1 | 20,799,470 | 1269.50 |
| reader/ready | 256 | 16.89 | 12.25 | -27.5% | 47.9 | 20,895,638 | 5101.47 |
| reader/ready | 4096 | 32.77 | 12.20 | -62.8% | 47.6 | 20,989,739 | 81991.17 |
| reader/ready | 65536 | 188.07 | 12.18 | -93.5% | 47.6 | 21,019,860 | 1313741.26 |
| reader/pending | 64 | 17.43 | 14.79 | -15.1% | 57.8 | 17,305,782 | 1056.26 |
| reader/pending | 256 | 19.75 | 14.73 | -25.4% | 57.5 | 17,383,462 | 4244.01 |
| reader/pending | 4096 | 35.44 | 14.74 | -58.4% | 57.6 | 17,369,188 | 67848.39 |
| reader/pending | 65536 | 189.28 | 14.73 | -92.2% | 57.5 | 17,378,513 | 1086157.06 |
| python_async_for/ready | 64 | 9337.35 | 8975.12 | -3.9% | 35059.1 | 28,523 | 1.74 |
| python_async_for/ready | 256 | 9343.62 | 9131.62 | -2.3% | 35670.4 | 28,034 | 6.84 |
| python_async_for/ready | 4096 | 9520.66 | 9111.25 | -4.3% | 35590.8 | 28,097 | 109.75 |
| python_async_for/ready | 65536 | 10154.28 | 9731.90 | -4.2% | 38015.2 | 26,305 | 1644.08 |
| python_async_for/pending | 64 | 9241.88 | 9464.63 | +2.4% | 36971.2 | 27,048 | 1.65 |
| python_async_for/pending | 256 | 8980.96 | 9156.51 | +2.0% | 35767.6 | 27,958 | 6.83 |
| python_async_for/pending | 4096 | 9353.44 | 9150.96 | -2.2% | 35746.0 | 27,975 | 109.28 |
| python_async_for/pending | 65536 | 10143.89 | 9811.42 | -3.3% | 38325.9 | 26,092 | 1630.75 |

Direct conversion and the benchmark-only legacy copy, measured within the final run with Python already attached:

| Bytes/chunk | Direct ns/chunk | Legacy Vec + PyBytes ns/chunk |
|---:|---:|---:|
| 64 | 7.5 | 22.4 |
| 256 | 8.5 | 24.3 |
| 4096 | 59.0 | 108.6 |
| 65536 | 670.9 | 1779.7 |

Reader time decreased across all cases. Full Python iteration changed between approximately −4% and +2%, including small regressions for pending sources at 64 and 256 bytes. Scheduling and run-to-run variation matter at this boundary; these results do not establish a broad Python streaming speedup

The synthetic source shares a prepared payload allocation across chunks. Cache locality and a single sequential before/after run limit how broadly these numbers apply. The production path now avoids the intermediate Vec allocation and payload copy by construction, but timing alone does not measure allocation counts

The saved `vec` Criterion baseline, confidence intervals, samples, timestamped metadata, and current summary remain under `litellm-rust/target/criterion` on the measurement machine. See [the benchmark guide](README.md) to rerun or save a new comparison
