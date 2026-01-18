# CPU Issue Classification & Reproduction

## 1. Classify the CPU Issue

Select the options that best describes the CPU behavior observed.

- [ ] CPU scales with traffic (RPS-driven)
- [ ] CPU increases without a traffic increase
- [ ] CPU increases after a LiteLLM upgrade

## 2. Can you reproduce the issue?

Before escalating, verify whether the CPU issue can be reproduced in a test environment that mirrors your production setup.  

If reproducible, provide **detailed reproduction steps** along with any relevant requests or configuration used.  
For guidance on the type of information we're looking for, see the [LiteLLM Troubleshooting Guide](../troubleshoot).

## 3. Issue Cannot Be Reproduced

If the CPU issue cannot be reproduced in a test environment that mirrors your production setup, please provide:

1. **Information from Section 1 and 2**  
   - CPU classification (Section 1)  
   - Reproduction attempts and environment details (Section 2)  

2. **Additional context** to help investigate:  
   - **Workload:** A realistic sample of requests processed before and during the spike, including any recent configuration changes.  
   - **Metrics:** CPU usage, P50/P99 latency, memory usage. Please include **screenshots** of the metrics whenever possible.  
   - **Logs / Alerts:** Any relevant logs or alerts captured **before and during the spike**.

> Providing this information allows the team to analyze patterns, correlate spikes with traffic or configuration, and attempt to reproduce the issue internally. Without it, our engineers won't have enough information to look into the problem.
