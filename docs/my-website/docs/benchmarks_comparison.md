# LiteLLM vs Portkey Performance Comparison

**Test Configuration**:

- 4 CPUs, 8 GB RAM per instance
- Load testing with 1k concurrent users, 500 ramp up

---

## Single Instance Results

### Portkey (Single Instance no DB)

| Metric              | Value       |
| ------------------- | ----------- |
| **Total Requests**  | 95,374      |
| **Failed Requests** | 24 (0.025%) |
| **Median Latency**  | 170ms       |
| **95th Percentile** | 6,400ms     |
| **99th Percentile** | 9,500ms     |
| **Average Latency** | 1,076ms     |
| **Current RPS**     | 526.6       |

### LiteLLM (Single Instance with DB)

| Metric               | Value                     |
| -------------------- | ------------------------- |
| **Total Requests**   | 133,364                   |
| **Failed Requests**  | 3 (0.002%)                |
| **Median Latency**   | 490ms                     |
| **95th Percentile**  | 2,700ms                   |
| **99th Percentile**  | 3,300ms                   |
| **Average Latency**  | 1,103ms                   |
| **Current RPS**      | 588.5                     |
| **LiteLLM Overhead** | 40ms (median), 59ms (p95) |

---

## Multi-Instance Results (4x Instances)

### Portkey (no DB)

| Metric              | Value   |
| ------------------- | ------- |
| **Total Requests**  | 293,796 |
| **Failed Requests** | 0       |
| **Median Latency**  | 100ms   |
| **95th Percentile** | 230ms   |
| **99th Percentile** | 500ms   |
| **Average Latency** | 123ms   |
| **Current RPS**     | 1,170.9 |

### LiteLLM (with DB)

| Metric              | Value   |
| ------------------- | ------- |
| **Total Requests**  | 312,405 |
| **Failed Requests** | 0       |
| **Median Latency**  | 120ms   |
| **95th Percentile** | 210ms   |
| **99th Percentile** | 590ms   |
| **Average Latency** | 135ms   |
| **Current RPS**     | 1,132.1 |

---

## Key Takeaways

**Single Instance**:

- LiteLLM: 88% fewer failures (3 vs 24)
- LiteLLM: 58% better p95 latency (2,700ms vs 6,400ms)
- LiteLLM: 65% better p99 latency (3,300ms vs 9,500ms)

**Multi-Instance (4x)**: Both systems perform comparably with zero failures and similar RPS (~1,150 RPS).

## Technical Observations

**Portkey**

* **Memory:** Low memory footprint.
* **CPU Utilization:** Does not fully utilize available CPUs, reaching only ~40% utilization.
* **Stability:** Experienced 3 outages during testing due to I/O timeout connection issues.
* **Performance:** Struggled with connection handling and delivered significantly poorer single-instance performance compared to LiteLLM.
* **Performance Oscillation:** Exhibits fewer latency spikes, resulting in more stable performance from the beginning.

**LiteLLM**

* **Memory:** High memory usage during both initialization and per-request processing.
* **CPU Utilization:** Fully utilizes available CPU resources.
* **Stability:** Experienced 1 outage due to out-of-memory (OOM) errors.
* **Performance:** Demonstrates robust connection handling on single instances, though memory usage remains high.
* **Performance Oscillation:** Achieves low latency the longer it runs, with a few initial spikes.