# EU AI Act Compliance Guide for LiteLLM Deployers

LiteLLM is an AI gateway. Every LLM call in your stack passes through it. That makes it the natural enforcement point for EU AI Act compliance: logging, monitoring, and transparency controls belong at the gateway layer.

This guide maps LiteLLM's existing features to regulatory requirements and identifies what deployers need to add.

## Is your system in scope?

Articles 12, 13, and 14 of the EU AI Act apply only to **high-risk AI systems** as defined in Annex III. Most LiteLLM deployments — general-purpose chatbots, internal productivity tools, code assistants — are not high-risk and are not subject to these obligations.

Your system is likely high-risk if it is used for:
- **Recruitment or HR decisions** (screening CVs, evaluating candidates, task allocation)
- **Credit scoring or insurance pricing**
- **Law enforcement or border control**
- **Critical infrastructure management** (energy, water, transport)
- **Education assessment** (grading, admissions)
- **Access to essential public services**

If your use case does not fall under Annex III, the high-risk obligations (Articles 9-15) do not apply via the Annex III pathway, though risk classification is context-dependent. **Do not self-classify without legal review.** You may still have obligations under **Article 50** (transparency for chatbots and AI systems interacting directly with users) and **GDPR** (if processing personal data). Focus on Article 50 (transparency) and GDPR (data protection) as your baseline obligations. Read those sections below.

## Why the gateway layer matters

For high-risk systems, the EU AI Act requires record-keeping (Article 12), transparency (Article 13), and human oversight (Article 14). These requirements apply to the deployed system, not to individual model providers.

LiteLLM sits between your application and 100+ LLM providers. It already captures:
- Model identifier per request
- Token counts (input, output, total)
- Cost per request
- Latency
- Error types and status codes
- User identity (via custom metadata)

This data is the raw material for compliance. The question is whether it satisfies the specific regulatory requirements.

## What the scanner found

Running [AI Trace Auditor](https://github.com/BipinRimal314/ai-trace-auditor) against the LiteLLM codebase:

- **Files scanned:** 4,861
- **AI providers supported:** 100+ including Anthropic, OpenAI, Google GenAI, AWS Bedrock, GCP Vertex AI, Azure OpenAI, and others
- **Model identifiers:** 112 (across all supported providers)
- **External services:** 12
- **Data flows:** 12

These reflect what LiteLLM *supports*. Your deployment routes to a subset. Document which providers are active.

## Data flow diagram

```mermaid
graph LR
    APP[Your Application] -->|API call| LiteLLM[LiteLLM Gateway]
    LiteLLM -->|routed request| Anthropic([Anthropic API])
    LiteLLM -->|routed request| OpenAI([OpenAI API])
    LiteLLM -->|routed request| Google([Google GenAI])
    LiteLLM -->|routed request| Bedrock([AWS Bedrock])
    LiteLLM -->|routed request| VertexAI([GCP Vertex AI])
    LiteLLM -->|routed request| Azure([Azure OpenAI])
    Anthropic -->|response| LiteLLM
    OpenAI -->|response| LiteLLM
    Google -->|response| LiteLLM
    Bedrock  -->|response| LiteLLM
    VertexAI  -->|response| LiteLLM
    Azure  -->|response| LiteLLM
    LiteLLM -->|response| APP

    classDef processor fill:#60a5fa,stroke:#1e40af,color:#000
    classDef app fill:#a78bfa,stroke:#5b21b6,color:#000
    classDef gateway fill:#4ade80,stroke:#166534,color:#000

    class APP app
    class LiteLLM gateway
    class Anthropic processor
    class OpenAI processor
    class Google processor
    class Bedrock processor
    class VertexAI processor
    class Azure processor
```

Every provider is a **processor** under GDPR: they process data on your behalf. Each requires a Data Processing Agreement (Article 28).

LiteLLM itself, when self-hosted, is under your control (controller). When using LiteLLM's hosted proxy, LiteLLM becomes an additional processor.

## Article 12: Record-keeping

Article 12 requires automatic event recording for the lifetime of high-risk AI systems. Here is how LiteLLM's existing features map:

| Article 12 Requirement | LiteLLM Feature | Status |
|------------------------|----------------|--------|
| Event timestamps | Request/response timestamps in callbacks | **Covered** |
| Model version tracking | `model` field logged per request | **Covered** |
| Input content logging | `messages` logged via callbacks (opt-in) | **Opt-in** |
| Output content logging | `response` logged via callbacks (opt-in) | **Opt-in** |
| Token consumption | `usage.prompt_tokens`, `usage.completion_tokens` | **Covered** |
| Cost tracking | `response_cost` calculated per request | **Covered** |
| Error recording | `exception` type and message in failure callbacks | **Covered** |
| Operation latency | Calculated from request timing | **Covered** |
| User identification | `user` field in request metadata | **Available** |
| Data retention (6+ months) | Depends on your logging backend | **Your responsibility** |
| Temperature/parameters | Logged if passed in request | **Partial** |

LiteLLM covers approximately 70-80% of Article 12 requirements out of the box when callbacks are configured. The gaps are:
1. **Content logging is opt-in** — you must explicitly enable it
2. **Retention is your responsibility** — LiteLLM doesn't store data persistently by default
3. **Request parameters** (temperature, max_tokens, top_p) need to be explicitly included in your logging

### Configuring Article 12-compliant logging

**Option 1: Use a built-in integration** (recommended for most deployments):

```python
import litellm

# Use a built-in backend: "langfuse", "s3", "datadog", "helicone", etc.
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]
```

**Option 2: Custom compliance logger** (when you need full control over what's captured):

```python
import litellm
from litellm.integrations.custom_logger import CustomLogger

class ComplianceLogger(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Article 12 fields to capture:
        # - kwargs["model"], kwargs["messages"] (input content)
        # - response_obj (output content)
        # - kwargs["optional_params"] (temperature, max_tokens)
        # - response_obj.usage (token counts)
        # - kwargs.get("response_cost") (cost per request)
        # - kwargs.get("user") (user identification)
        # - start_time, end_time (timestamps and latency)
        ...

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        # Capture: exception type, message, failing model, timestamp
        ...

litellm.callbacks = [ComplianceLogger()]
```

Whichever option you choose, connect to a persistent backend with a retention policy of at least 6 months (Article 26(6) for deployers; Article 18 requires 10 years for providers).

## Article 13: Transparency

Deployers must inform users that they are interacting with an AI system and provide information about its capabilities and limitations.

LiteLLM's contribution to transparency:
- **Model routing is logged** — you can tell users which model answered their query
- **Cost attribution** — you know which features consume the most AI resources
- **Fallback chains are visible** — when a primary model fails and a fallback serves the response, this is logged

What you need to add:
- User-facing disclosure that AI is involved in generating responses
- Documentation of which models are active and their known limitations
- Information about how routing decisions are made (cost, latency, quality)

## Article 14: Human oversight

Article 14 requires high-risk AI systems to be designed so that natural persons can effectively oversee them: interpret outputs, decide not to use them, and intervene or halt the system. This means human actors in the loop, not automated controls.

LiteLLM's guardrails (content moderation, rate limiting, budget controls, model access controls) are **automated technical controls** that fall under Articles 9 (risk management) and 15 (accuracy/robustness). They are useful infrastructure, but they do not satisfy Article 14 on their own.

What you need to build for Article 14 compliance:

| Requirement | What it means | LiteLLM foundation |
|-------------|---------------|-------------------|
| Human interpretation of outputs | A person can review what the AI produced before it acts | Logged responses via callbacks |
| Decision not to use output | A person can reject an AI recommendation | Requires your application layer |
| Intervention / halt | A person can stop the system mid-operation | Guardrails trigger points can route to human review |
| Escalation procedures | When automated filters flag content, a human reviews | Callback hooks available; escalation logic is yours |

LiteLLM provides the **logging and hook infrastructure** to build human oversight on top of. The oversight logic itself — review queues, approval workflows, kill switches — lives in your application layer.

## GDPR considerations

LiteLLM processes user prompts. If those prompts contain personal data:

1. **Legal basis** (Article 6): Document why you're processing this data
2. **Data Processing Agreements** (Article 28): Required for each LLM provider you route to
3. **Cross-border transfers**: US-based providers (OpenAI, Anthropic) require Standard Contractual Clauses or equivalent safeguards
4. **Data minimization**: Log what you need for compliance, not everything

Generate a GDPR Article 30 Record of Processing Activities:

```bash
pip install ai-trace-auditor
aitrace flow ./your-litellm-deployment -o data-flows.md
```

## Full compliance scan

Generate a complete compliance package:

```bash
aitrace comply ./your-litellm-deployment --split -o compliance/
```

## Recommendations

1. **Enable comprehensive logging** with a persistent backend and 6+ month retention
2. **Audit your traces** periodically: `aitrace audit your-traces.json -r "EU AI Act"`
3. **Document your routing policy** — which models, which fallbacks, which guardrails
4. **Establish DPAs** with every LLM provider you route to
5. **Use self-hosted models** (Ollama, vLLM) for sensitive data to avoid third-party transfers

## Resources

- [EU AI Act full text](https://artificialintelligenceact.eu/)
- [LiteLLM logging documentation](https://docs.litellm.ai/docs/observability/callbacks)
- [LiteLLM guardrails](https://docs.litellm.ai/docs/proxy/guardrails)
- [AI Trace Auditor](https://github.com/BipinRimal314/ai-trace-auditor) — open-source compliance scanning

---

*This guide was generated with assistance from [AI Trace Auditor](https://github.com/BipinRimal314/ai-trace-auditor) and reviewed for accuracy. It is not legal advice. Consult a qualified professional for compliance decisions.*
