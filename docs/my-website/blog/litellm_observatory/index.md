---
slug: litellm-observatory
title: "Improve release stability with 24 hour load tests"
date: 2026-02-06T10:00:00
authors:
  - name: Alexsander Hamir
    title: "Performance Engineer, LiteLLM"
    url: https://www.linkedin.com/in/alexsander-baptista/
    image_url: https://github.com/AlexsanderHamir.png
  - name: Krrish Dholakia
    title: "CEO, LiteLLM"
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: "CTO, LiteLLM"
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
description: "How we built a long-running, release-validation system to catch regressions before they reach users."
tags: [testing, observability, reliability, releases]
hide_table_of_contents: false
---

![LiteLLM Observatory](https://raw.githubusercontent.com/AlexsanderHamir/assets/main/Screenshot%202026-01-31%20175355.png)

# Improve release stability with 24 hour load tests

As LiteLLM adoption has grown, so have expectations around reliability, performance, and operational safety. Meeting those expectations requires more than correctness-focused tests, it requires validating how the system behaves over time, under real-world conditions.

This post introduces **LiteLLM Observatory**, a long-running release-validation system we built to catch regressions before they reach users.

---

## Why We Built the Observatory

LiteLLM operates at the intersection of external providers, long-lived network connections, and high-throughput workloads. While our unit and integration tests do an excellent job validating correctness, they are not designed to surface issues that only appear after extended operation.

A subtle lifecycle edge case discovered in v1.81.3 reinforced the need for stronger release validation in this area.

---

## A Real-World Lifecycle Edge Case

In v1.81.3, we shipped a fix for an HTTP client memory leak. The change passed unit and integration tests and behaved correctly in short-lived runs.

The issue that surfaced was not caused by a single incorrect line of logic, but by how multiple components interacted over time:

- A cached `httpx` client was configured with a 1-hour TTL
- When the cache expired, the underlying HTTP connection was closed as expected
- A higher-level client continued to hold a reference to that connection
- Subsequent requests failed with:

```
Cannot send a request, as the client has been closed
```

**Before (with bug):**

| Provider | Requests | Success | Failures | Fail % |
|----------|----------|---------|----------|--------|
| OpenAI   | 720,000  | 432,000 | 288,000  | 40%    |
| Azure    | 692,000  | 415,200 | 276,800  | 40%    |

**After (fixed):**

| Provider | Requests   | Success   | Failures | Fail %  |
|----------|------------|-----------|----------|---------|
| OpenAI   | 1,200,000  | 1,199,988 | 12       | 0.001%  |
| Azure    | 1,150,000  | 1,149,982 | 18       | 0.002%  |

Our focus moving forward is on being the first to detect issues, even when they aren’t covered by unit tests. LiteLLM Observatory is designed to surface latency regressions, OOMs, and failure modes that only appear under real traffic patterns in **our own production deployments** during release validation.


---

### How the Observatory Works

[LiteLLM Observatory](https://github.com/BerriAI/litellm-observatory) is a testing service that runs long-running tests against our LiteLLM deployments. We trigger tests by sending API requests, and results are automatically sent to Slack when tests complete.

#### How Tests Run

1. **Start a Test**: We send a request to the Observatory API with:
   - Which LiteLLM deployment to test (URL and API key)
   - Which test to run (e.g., `TestOAIAzureRelease`)
   - Test settings (which models to test, how long to run, failure thresholds)

2. **Smart Queueing**:
   - The system checks whether we are attempting to run the exact same test more than once
   - If a duplicate test is already running or queued, we receive an error to avoid wasting resources
   - Otherwise, the test is added to a queue and runs when capacity is available (up to 5 tests can run concurrently by default)

3. **Instant Response**: The API responds immediately—we do not wait for the test to finish. Tests may run for hours, but the request itself completes in milliseconds.

4. **Background Execution**:
   - The test runs in the background, issuing requests against our LiteLLM deployment
   - It tracks request success and failure rates over time
   - When the test completes, results are automatically posted to our Slack channel

#### Example: The OpenAI / Azure Reliability Test

The `TestOAIAzureRelease` test is designed to catch a class of bugs that only surface after sustained runtime:

- **Duration**: Runs continuously for 3 hours
- **Behavior**: Cycles through specified models (such as `gpt-4` and `gpt-3.5-turbo`), issuing requests continuously
- **Why 3 Hours**: This helps catch issues where HTTP clients degrade or fail after extended use (for example, a bug observed in LiteLLM v1.81.3)
- **Pass / Fail Criteria**: The test passes if fewer than 1% of requests fail. If the failure rate exceeds 1%, the test fails and we are notified in Slack
- **Key Detail**: The same HTTP client is reused for the entire run, allowing us to detect lifecycle-related bugs that only appear under prolonged reuse

#### When We Use It

- **Before Deployments**: Run tests before promoting a new LiteLLM version to production
- **Routine Validation**: Schedule regular runs (daily or weekly) to catch regressions early
- **Issue Investigation**: Run tests on demand when we suspect a deployment issue
- **Long-Running Failure Detection**: Identify bugs that only appear under sustained load, beyond what short smoke tests can reveal


### Complementing Unit Tests

Unit tests remain a foundational part of our development process. They are fast and precise, but they don’t cover:

- Real provider behavior
- Long-lived network interactions
- Resource lifecycle edge cases
- Time-dependent regressions

LiteLLM Observatory complements unit tests by validating the system as it actually runs in production-like environments.

---

### Looking Ahead

Reliability is an ongoing investment.

LiteLLM Observatory is one of several systems we’re building to continuously raise the bar on release quality and operational safety. As LiteLLM evolves, so will our validation tooling, informed by real-world usage and lessons learned.

We’ll continue to share those improvements openly as we go.

