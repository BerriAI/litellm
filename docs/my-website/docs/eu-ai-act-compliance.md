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

If your use case does not fall under Annex III, the high-risk obligations (Articles 9-15) are less likely to apply, but risk classification is context-dependent. **Do not self-classify without legal review.** You may still have obligations under **Article 50** (transparency for chatbots and AI systems interacting directly with users) and **GDPR** (if processing personal data). Focus on Article 50 (transparency) and GDPR (data protection) as your baseline obligations. Read those sections below.

If your system is high-risk, the August 2, 2026 deadline for full compliance applies. The rest of this guide addresses high-risk obligations.

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

## Supported providers

LiteLLM integrates with Anthropic, OpenAI, Google GenAI, HuggingFace, Mistral, and many others — 100+ providers total. Your deployment routes to a subset. Document which providers are active in your system, as each has different compliance implications.

## Data flow diagram

```mermaid
graph LR
    APP[Your Application] -->|API call| LiteLLM[LiteLLM Gateway]
    LiteLLM -->|routed request| Anthropic([Anthropic API])
    LiteLLM -->|routed request| OpenAI([OpenAI API])
    LiteLLM -->|routed request| Google([Google GenAI])
    LiteLLM -->|routed request| Bedrock{{AWS Bedrock}}
    LiteLLM -->|routed request| VertexAI{{GCP Vertex AI}}
    LiteLLM -->|routed request| Azure{{Azure OpenAI}}
    Anthropic -->|response| LiteLLM
    OpenAI -->|response| LiteLLM
    Google -->|response| LiteLLM
    Bedrock -->|response| LiteLLM
    VertexAI -->|response| LiteLLM
    Azure -->|response| LiteLLM
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

Providers are typically processors for customer-submitted data, but the exact role depends on each provider's terms of service and processing purpose. Deployers should review each provider's DPA. Each requires a Data Processing Agreement (Article 28).

When you self-host LiteLLM, your organization is the data controller — you determine the purpose and means of processing. LiteLLM as software has no GDPR role; the legal designation applies to the organization operating it. When using LiteLLM's hosted proxy service, the organization operating that service becomes an additional data processor.

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
| Data retention | Depends on your logging backend | **Your responsibility** |
| Temperature/parameters | Logged if passed in request | **Partial** |

Based on the mapping above, LiteLLM's callback system addresses most of the data fields Article 12 references when properly configured. The remaining gaps are:
1. **Content logging is opt-in** — you must explicitly enable it
2. **Retention is your responsibility** — LiteLLM doesn't store data persistently by default. Article 18 requires providers of high-risk systems to retain logs and technical documentation for **10 years** after market placement. Deployers under Article 26(6) must retain logs for a period appropriate to the intended purpose and at least 6 months. Confirm the applicable retention period with legal counsel.
3. **Request parameters** (temperature, max_tokens, top_p) need to be explicitly included in your logging

### Configuring Article 12-compliant logging

Enable comprehensive logging via LiteLLM callbacks:

```python
import litellm

litellm.success_callback = ["your_logging_backend"]
litellm.failure_callback = ["your_logging_backend"]

# Ensure these fields are captured in your callback:
# - model, messages, response, usage, response_cost
# - temperature, max_tokens (from kwargs)
# - user, metadata
# - timestamps, latency
# - error type and message (on failure)
```

Connect to a persistent backend (Langfuse, Helicone, or your own database). Set your retention policy based on your role: providers must retain logs for 10 years (Article 18); deployers for at least 6 months (Article 26(6)). Confirm with legal counsel.

## Article 13: Transparency to deployers

Article 13 requires providers of high-risk AI systems to supply deployers with sufficient information — instructions for use, accuracy metrics, known limitations — to operate the system appropriately. This is provider-to-deployer transparency.

Article 13 compliance requires the provider of the high-risk system to produce and maintain system-level documentation: intended purpose, accuracy and robustness metrics, known risks, and technical measures for monitoring. This is a documentation obligation, not a logging obligation.

LiteLLM's observability features provide **supporting evidence** that can inform Article 13 documentation, but they do not satisfy Article 13 on their own:
- **Model routing logs** — help compile which models are in use and how requests are distributed
- **Cost attribution** — supports resource usage documentation
- **Fallback chain visibility** — provides evidence of system behavior under failure conditions
- **Provider documentation links** — LiteLLM links to upstream model cards, but these describe the LLM providers' models, not your high-risk AI system as a whole

You must independently produce system documentation that covers how your specific deployment uses LiteLLM, its intended purpose, performance characteristics, and residual risks.

## Article 50: End-user transparency

Article 50 requires deployers to inform end users that they are interacting with an AI system. This is deployer-to-user transparency, and it is a separate obligation from Article 13.

What you need to add:
- User-facing disclosure that AI is involved in generating responses
- A mechanism for users to identify when an AI-generated response has been delivered (e.g., clear labeling in the UI)

Note: Article 50 applies to chatbots and systems interacting directly with natural persons. It has a separate scope from the "high-risk" designation under Annex III — it applies even to limited-risk systems.

## Article 14: Human oversight

Article 14 requires that high-risk AI systems be designed so that natural persons can effectively oversee them — including the ability to understand, monitor, interpret, and intervene in the system's operation. LiteLLM's guardrails provide **automated technical safeguards** that support human oversight, but they are not a substitute for it:

| Guardrails Feature | What It Does | Oversight Role |
|-------------------|-------------|----------------|
| Content moderation | Pre-response filtering for harmful content | **Automated safeguard** — reduces the volume of outputs requiring human review, but does not replace human judgment on edge cases |
| Rate limiting | Prevents runaway AI usage | **Automated safeguard** — bounds system behavior, supports the human overseer's ability to maintain control |
| Budget controls | Cost caps per user/team/organization | **Automated safeguard** — prevents uncontrolled resource consumption |
| Model access controls | Restricts which models specific users can access | **Automated safeguard** — enforces organizational policy on model usage |

These automated controls are necessary building blocks, but Article 14 compliance requires **human oversight procedures** on top of them:
- **Escalation procedures** — define what happens when a guardrail triggers (who is notified, what action is taken)
- **Human review pipeline** — for high-stakes decisions, route AI outputs to a qualified person before they take effect
- **Override mechanism** — a human must be able to halt AI responses or override the system's output
- **Competence requirements** — the human overseer must understand the system's capabilities, limitations, and the context of its outputs

The distinction matters: automated safeguards reduce risk, but Article 14 requires a natural person who can exercise judgment and intervene. Configure guardrails as the first line of defense, then build human oversight procedures around them.

## GDPR considerations

LiteLLM processes user prompts. If those prompts contain personal data:

1. **Legal basis** (Article 6): Document why you're processing this data
2. **Data Processing Agreements** (Article 28): Required for each LLM provider you route to
3. **Cross-border transfers**: Providers based outside the EEA — including US-based providers (OpenAI, Anthropic), and any other non-EEA providers you route to — require Standard Contractual Clauses (SCCs) or equivalent safeguards under Chapter V of the GDPR. Review each provider's transfer mechanism individually.
4. **Data minimization**: Log what you need for compliance, not everything

Consider maintaining a GDPR Article 30 Record of Processing Activities that documents each LLM provider relationship, the data categories processed, and the legal basis for processing.

## Recommendations

1. **Enable comprehensive logging** with a persistent backend and retention per Article 18 (10 years for providers) or Article 26(6) (minimum 6 months for deployers)
2. **Audit your traces periodically** against Article 12 requirements
3. **Document your routing policy** — which models, which fallbacks, which guardrails
4. **Establish DPAs** with every LLM provider you route to
5. **Use self-hosted models** (Ollama, vLLM) for sensitive data to avoid third-party transfers

## Resources

- [EU AI Act full text](https://artificialintelligenceact.eu/)
- [LiteLLM logging documentation](https://docs.litellm.ai/docs/observability/callbacks)
- [LiteLLM guardrails](https://docs.litellm.ai/docs/proxy/guardrails)
- [EU AI Office guidance](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)

---

*This is not legal advice. Consult a qualified professional for compliance decisions.*
