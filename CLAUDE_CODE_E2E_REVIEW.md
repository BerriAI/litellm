# Interview Review: Claude Code ↔ LiteLLM E2E Integration

## **Overview**
This deliverable implements a production-grade CI/CD workflow for the **LiteLLM ↔ Claude Code** integration. Claude Code is a critical agentic coding tool for LiteLLM users; this suite ensures that any breaking changes to the Anthropic-compatible proxy layer are caught before reaching `main`.

## **Technical Design**
The testing architecture follows a **Research -> Strategy -> Execution** lifecycle, prioritizing reliability and signal quality.

- **Orchestration**: Uses a multi-container Docker Compose setup (`Postgres` + `LiteLLM` + `Claude Code Client`).
- **Isolation**: The Claude Code CLI runs in a security-hardened, non-root container.
- **Verification Strategy**: 
    - **Functional**: Back-to-back requests (V0/V1) ensure state management and connectivity are stable across multiple turns.
    - **Proxy-Side**: Direct validation of `x-litellm-call-id` and `x-litellm-response-cost` headers to confirm the proxy is correctly accounting for agentic usage.
- **Exclusion Policy**: Implemented a 3-day exclusion window for the `latest` Claude Code version to avoid upstream "day-zero" flakiness.

## **Security Posture**
A primary goal was **credential isolation** to prevent "blast radius" exposure during E2E runs:
- **Zero-Secret Client**: The Claude Code container *never* sees the real `ANTHROPIC_API_KEY`. It only holds a transient `LITELLM_MASTER_KEY` valid only for the local test network.
- **Secure Proxy**: LiteLLM acts as the secure vault, holding the upstream credentials and only exposing the necessary API surface to the isolated client.

## **Versions & Determinism**
To ensure CI stability while testing real-world scenarios, we locked and validated the following versions:
- `2.1.100` (Current Primary)
- `2.1.90` (Stable N-1)
- `2.1.80` (Stable N-2)

## **Trade-off Awareness**
- **Anthropic Focus**: We prioritized Native Anthropic over Bedrock/Vertex for the initial delivery. This choice was made to maximize ROI on the most common customer path and ensure 100% fidelity before expanding the provider matrix.
- **Pragmatism**: We intentionally deferred complex "tool-use" assertions in favor of robust "back-to-back" connectivity tests. This provides a cleaner failure signal: if the CLI can't talk to LiteLLM, we know immediately without debugging complex agent logic.

## **How to Run**
The suite is integrated into CircleCI but can be run locally for rapid iteration:
```bash
# Set your key
export ANTHROPIC_API_KEY=sk-...

# Execute the E2E suite
tests/proxy_e2e_anthropic_messages_tests/claude_code/run_claude_code_docker_test.sh
```

## **Deliverables**
- [x] **CircleCI Pipeline**: Automated jobs for multiple Claude Code versions.
- [x] **Back-to-Back Request Suite**: Verification of multi-turn stability.
- [x] **Security-Hardened Docker Environment**: Isolated non-root execution.
- [x] **Proxy Header Validation**: Confirmation of usage tracking and cost logging.
