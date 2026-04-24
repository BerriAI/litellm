import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Enterprise

LiteLLM Enterprise gives teams the security, governance, and support needed to run LLMs reliably at scale — SSO, audit logs, fine‑grained access control, advanced guardrails, and dedicated professional support.

:::info
- ✨ SSO is free for up to 5 users. Beyond that, an Enterprise license is required. [Get started](https://www.litellm.ai/enterprise)
- Enterprise is designed for teams with **100+ users** or **10+ AI use-cases**. Not sure if it fits? [Talk to us](https://enterprise.litellm.ai/demo).
- Free **7‑day trial** keys available [here](https://www.litellm.ai/enterprise#trial).
:::

[**Procurement available via AWS / Azure Marketplace**](./data_security.md#legalcompliance-faqs)

---

## Why Enterprise

<Image img={require('../img/enterprise_vs_oss_2.png')} />

Enterprise includes everything in open source plus:

- ✅ **Feature prioritization** — influence the roadmap
- ✅ **Custom integrations** built for your stack
- ✅ **Professional support** with a dedicated Slack / Teams channel
- ✅ **SLAs** for production incidents

---

## Feature Overview

### Security & Access Control
- [SSO for Admin UI](./proxy/ui.md#✨-enterprise-features)
- [Audit logs with retention policy](#audit-logs)
- [JWT authentication](./proxy/token_auth.md)
- [Control public / private routes](./proxy/public_routes.md)
- [Secret managers — AWS KMS, Google Secret Manager, Azure Key Vault, HashiCorp Vault](./secret)
- [[Beta] AWS KMS v2 — key decryption](#beta-aws-key-manager--key-decryption)
- IP address access control lists
- Request IP tracking
- [Max request / response size limits](#set-max-request--response-size-on-litellm-proxy)
- [Enforce required params on LLM requests](#required-params-for-llm-requests)
- [Key rotation](./proxy/virtual_keys.md#-key-rotations)

### Logging, Guardrails & Caching per Project
- [Team-based logging](./proxy/team_logging.md) — each team routes to its own Langfuse project or custom callback
- [Disable logging per team](./proxy/team_logging.md#disable-logging-for-a-team) (GDPR compliance)

### Spend Tracking & Data Exports
- [USD budgets for custom tags](./proxy/provider_budget_routing#-tag-budgets)
- [Model budgets for virtual keys](./proxy/users#-virtual-key-model-specific)
- [Export logs to GCS or Azure Blob Storage](./observability/gcs_bucket_integration)
- [`/spend/report` endpoint](./proxy/cost_tracking.md#✨-enterprise-api-endpoints-to-get-spend)

### Guardrails per API Key / Team
- [Secret detection & redaction](#guardrails---secret-detectionredaction)
- [LLM Guard](#content-moderation-with-llm-guard), [LlamaGuard](#content-moderation-with-llamaguard), [Google Text Moderation](#content-moderation-with-google-text-moderation)
- [Blocked user lists](#enable-blocked-user-lists)
- [Banned keyword lists](#enable-banned-keywords-list)

### Projects
- [Project Management API](./proxy/project_management.md) — projects sit between teams and keys for fine-grained access and budget control
- [Project Management UI](./proxy/ui_project_management.md) — create and manage projects directly from the Admin UI

### Custom Branding
- [Custom branding + routes on Swagger](#swagger-docs---custom-routes--branding)
- [Custom email branding](./proxy/email.md#customizing-email-branding)

---

## Deployment Options

### Self-Hosted

Deploy our Docker image or build from our pip package and manage your own infrastructure. We provide a license key and support through a dedicated channel.

Add your license key to the environment:

```env
LITELLM_LICENSE="eyJ..."
```

**No data leaves your environment.**

Self-Managed Enterprise pricing is tailored to your deployment — [contact us to learn more](https://enterprise.litellm.ai/demo).

### Hosted LiteLLM Cloud

LiteLLM operates the proxy so your team can focus on products. **Status: GA** — already used in production. Live reliability: [status.litellm.ai](https://status.litellm.ai/).

- **No infrastructure** to maintain
- **Reliable** — load-tested at 1,000 RPS.
- **Secure** — SOC 2 Type II and ISO 27001 certified.

Supported regions: [see the data security page](./data_security#supported-data-regions-for-litellm-cloud).

---

## Support & SLAs

Professional Support covers LLM / provider integrations, deployment, upgrade management, and provider troubleshooting. We can't fix issues inside your own infrastructure, but we'll guide you through them.

### Response time SLAs

| Severity | Definition | Response SLA | Coverage |
| --- | --- | --- | --- |
| **Sev0** | 100% production traffic failing | **1 hour** | 24×7 |
| **Sev1** | Partial production traffic failing (< 100%) | **6 hours** | 24×7 |
| **Sev2** | Degraded performance, workaround exists | **24 hours** | 7 am – 7 pm PT, Mon–Sat |
| **Sev3** | Setup / configuration issues (e.g., Redis works on our end, not yours) | **24 hours** | 7 am – 7 pm PT, Mon–Sat |
| **Security** | Vulnerability patching | **72 hours** | — |

Custom SLAs are available based on your deployment, traffic profile, and severity definitions.

### What's included

- **Dedicated Slack / Teams channel** with the LiteLLM engineering team
- **Upgrade management** — guidance on version bumps, breaking changes, migration steps
- **LLM provider troubleshooting** — we coordinate with upstream providers on your behalf where possible
- **Architecture reviews** — load balancing, fallbacks, caching, and cost-control design sessions
- **Priority bug fixes** backed by the response SLAs above
- **Feature prioritization** — requested roadmap items are prioritized for Enterprise customers

### What's out of scope

- Issues inside your own infrastructure (networking, Kubernetes, cloud IAM, etc.) — we will guide but not remediate
- Custom code you've written on top of LiteLLM that isn't part of our supported surface
- Third‑party integrations outside of supported providers and callbacks

### How to escalate

1. Post in your dedicated support channel with a Sev level, a short description, and reproduction details or logs.
2. For Sev0 / Sev1, tag the on-call responder — response clocks start at message time.
3. Security issues: email **security@litellm.ai** or use the private channel. Please do **not** post exploit details in public GitHub issues.

---

## Data Security, Legal & Compliance

LiteLLM is designed for environments with strict data and regulatory requirements. Self-hosted deployments keep all data in your own infrastructure — LiteLLM never sees your prompts, responses, API keys, or logs.

### Certifications & attestations

- **SOC 2 Type II** — audited controls for security, availability, and confidentiality
- **ISO 27001** — information security management system certification
- **HIPAA-ready** deployments available on LiteLLM Cloud with a signed BAA

### Data handling

- **Self-Hosted:** all request/response data, prompt content, virtual keys, and spend logs stay inside your VPC. LiteLLM only sees license-validation metadata (license key fingerprint + version heartbeat).
- **LiteLLM Cloud:** data is encrypted at rest (AES‑256) and in transit (TLS 1.2+). Customer data is logically isolated per tenant. Backups are region-pinned.
- **No training on customer data.** LiteLLM does not train models or share customer prompts with any third party outside the LLM provider you've explicitly routed to.

### Data residency

LiteLLM Cloud supports pinning data and processing to specific regions (US, EU, and others). See [supported data regions](./data_security#supported-data-regions-for-litellm-cloud).

### Retention & deletion

- Request and spend logs follow the retention window you configure in the proxy (default: indefinite, configurable per deployment).
- Upon contract termination, customer data on LiteLLM Cloud is deleted within **30 days**, with a signed deletion attestation available on request.

### Subprocessors

A current list of subprocessors used by LiteLLM Cloud is available on request. We notify customers in writing of any material subprocessor changes with at least 30 days notice.

### DPAs, BAAs & contractual documents

The following are available under NDA:

- **Data Processing Agreement (DPA)** — GDPR, CCPA, UK-GDPR aligned
- **Business Associate Agreement (BAA)** — for HIPAA-covered deployments
- **Standard Contractual Clauses (SCCs)** — for cross-border EU data transfers
- **Security questionnaire responses** — SIG Lite, CAIQ available
- **Penetration test summaries** — executive summary of most recent third-party pentest

Request documents via [enterprise.litellm.ai/demo](https://enterprise.litellm.ai/demo).

### Vulnerability disclosure

Report security vulnerabilities privately to **security@litellm.ai**. We commit to:

- Acknowledge receipt within **24 hours**
- Provide an initial assessment within **72 hours**
- Patch confirmed vulnerabilities within the Security SLA above
- Credit reporters in release notes where desired

### Full FAQ

For the complete list of data, legal, and compliance FAQs, see [Data Security / Legal / Compliance](./data_security.md).

---

## Frequently Asked Questions

### How to set up and verify your Enterprise License

1. Add your license key to the environment:

    ```env
    LITELLM_LICENSE="eyJ..."
    ```

2. Restart LiteLLM Proxy.
3. Open `http://<your-proxy-host>:<port>/` — the Swagger page description should show **"Enterprise Edition"**. If not, confirm the key is correct, unexpired, and that the proxy was fully restarted.

### Pricing

Pricing is usage-based. We'll tailor something that works for your team on a call — [contact us](https://enterprise.litellm.ai/demo).

---

## Feature Deep Dives

### Blocking web crawlers

To block web crawlers from indexing the proxy server endpoints, set the `block_robots` setting to `true` in your `litellm_config.yaml` file.

```yaml showLineNumbers title="litellm_config.yaml"
general_settings:
  block_robots: true
```

#### How it works

When this is enabled, the `/robots.txt` endpoint will return a 200 status code with the following content:

```shell showLineNumbers title="robots.txt"
User-agent: *
Disallow: /
```

### Required Params for LLM Requests

Use this when you want to enforce all requests to include certain params. Example you need all requests to include the `user` and `["metadata"]["generation_name"]` params.

<Tabs>

<TabItem value="config" label="Set on Config">

**Step 1** Define all Params you want to enforce on config.yaml

This means `["user"]` and `["metadata"]["generation_name"]` are required in all LLM Requests to LiteLLM

```yaml
general_settings:
  master_key: sk-1234
  enforced_params:
    - user
    - metadata.generation_name
```
</TabItem>

<TabItem value="key" label="Set on Key">

```bash
curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "enforced_params": ["user", "metadata.generation_name"]
}'
```

</TabItem>
</Tabs>

**Step 2 Verify if this works**

<Tabs>

<TabItem value="bad" label="Invalid Request (No `user` passed)">

```shell
curl --location 'http://localhost:4000/chat/completions' \
    --header 'Authorization: Bearer sk-5fmYeaUEbAMpwBNT-QpxyA' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "hi"
        }
    ]
}'
```

Expected Response

```shell
{"error":{"message":"Authentication Error, BadRequest please pass param=user in request body. This is a required param","type":"auth_error","param":"None","code":401}}%
```

</TabItem>

<TabItem value="bad2" label="Invalid Request (No `metadata` passed)">

```shell
curl --location 'http://localhost:4000/chat/completions' \
    --header 'Authorization: Bearer sk-5fmYeaUEbAMpwBNT-QpxyA' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "user": "gm",
    "messages": [
        {
        "role": "user",
        "content": "hi"
        }
    ],
   "metadata": {}
}'
```

Expected Response

```shell
{"error":{"message":"Authentication Error, BadRequest please pass param=[metadata][generation_name] in request body. This is a required param","type":"auth_error","param":"None","code":401}}%
```


</TabItem>
<TabItem value="good" label="Valid Request">

```shell
curl --location 'http://localhost:4000/chat/completions' \
    --header 'Authorization: Bearer sk-5fmYeaUEbAMpwBNT-QpxyA' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "user": "gm",
    "messages": [
        {
        "role": "user",
        "content": "hi"
        }
    ],
   "metadata": {"generation_name": "prod-app"}
}'
```

Expected Response

```shell
{"id":"chatcmpl-9XALnHqkCBMBKrOx7Abg0hURHqYtY","choices":[{"finish_reason":"stop","index":0,"message":{"content":"Hello! How can I assist you today?","role":"assistant"}}],"created":1717691639,"model":"gpt-3.5-turbo-0125","object":"chat.completion","system_fingerprint":null,"usage":{"completion_tokens":9,"prompt_tokens":8,"total_tokens":17}}%
```

</TabItem>
</Tabs>

### Control available public, private routes

See [Control Public & Private Routes](./proxy/public_routes.md) for detailed documentation on configuring public routes, admin-only routes, allowed routes, and wildcard patterns.

## Spend Tracking

#### Viewing Spend per tag

#### `/spend/tags` Request Format
```shell
curl -X GET "http://0.0.0.0:4000/spend/tags" \
-H "Authorization: Bearer sk-1234"
```

#### `/spend/tags` Response Format
```shell
[
  {
    "individual_request_tag": "model-anthropic-claude-v2.1",
    "log_count": 6,
    "total_spend": 0.000672
  },
  {
    "individual_request_tag": "app-ishaan-local",
    "log_count": 4,
    "total_spend": 0.000448
  },
  {
    "individual_request_tag": "app-ishaan-prod",
    "log_count": 2,
    "total_spend": 0.000224
  }
]
```

:::tip
For comprehensive spend tracking features including budgets, alerts, and detailed analytics, check out [Spend Tracking](https://docs.litellm.ai/docs/proxy/cost_tracking).
:::


## Guardrails - Secret Detection/Redaction
❓ Use this to REDACT API Keys, Secrets sent in requests to an LLM.

Example if you want to redact the value of `OPENAI_API_KEY` in the following request

#### Incoming Request

```json
{
    "messages": [
        {
            "role": "user",
            "content": "Hey, how's it going, API_KEY = 'sk_1234567890abcdef'"
        }
    ]
}
```

#### Request after Moderation

```json
{
    "messages": [
        {
            "role": "user",
            "content": "Hey, how's it going, API_KEY = '[REDACTED]'"
        }
    ]
}
```

**Usage**

**Step 1** Add this to your config.yaml

```yaml
litellm_settings:
  callbacks: ["hide_secrets"]
```

**Step 2** Run litellm proxy with `--detailed_debug` to see the server logs

```
litellm --config config.yaml --detailed_debug
```

**Step 3** Test it with request

Send this request
```shell
curl --location 'http://localhost:4000/chat/completions' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "llama3",
    "messages": [
        {
        "role": "user",
        "content": "what is the value of my open ai key? openai_api_key=sk-1234998222"
        }
    ]
}'
```


Expect to see the following warning on your litellm server logs

```shell
LiteLLM Proxy:WARNING: secret_detection.py:88 - Detected and redacted secrets in message: ['Secret Keyword']
```


You can also see the raw request sent from litellm to the API Provider
```json
POST Request Sent from LiteLLM:
curl -X POST \
https://api.groq.com/openai/v1/ \
-H 'Authorization: Bearer gsk_mySVchjY********************************************' \
-d {
  "model": "llama3-8b-8192",
  "messages": [
    {
      "role": "user",
      "content": "what is the time today, openai_api_key=[REDACTED]"
    }
  ],
  "stream": false,
  "extra_body": {}
}
```

### Secret Detection On/Off per API Key

❓ Use this when you need to switch guardrails on/off per API Key

**Step 1** Create Key with `hide_secrets` Off

👉 Set `"permissions": {"hide_secrets": false}` with either `/key/generate` or `/key/update`

This means the `hide_secrets` guardrail is off for all requests from this API Key

<Tabs>
<TabItem value="/key/generate" label="/key/generate">

```shell
curl --location 'http://0.0.0.0:4000/key/generate' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "permissions": {"hide_secrets": false}
}'
```

```shell
# {"permissions":{"hide_secrets":false},"key":"sk-jNm1Zar7XfNdZXp49Z1kSQ"}
```

</TabItem>
<TabItem value="/key/update" label="/key/update">

```shell
curl --location 'http://0.0.0.0:4000/key/update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "key": "sk-jNm1Zar7XfNdZXp49Z1kSQ",
        "permissions": {"hide_secrets": false}
}'
```

```shell
# {"permissions":{"hide_secrets":false},"key":"sk-jNm1Zar7XfNdZXp49Z1kSQ"}
```

</TabItem>
</Tabs>

**Step 2** Test it with new key

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Authorization: Bearer sk-jNm1Zar7XfNdZXp49Z1kSQ' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "llama3",
    "messages": [
        {
        "role": "user",
        "content": "does my openai key look well formatted OpenAI_API_KEY=sk-1234777"
        }
    ]
}'
```

Expect to see `sk-1234777` in your server logs on your callback.

:::info
The `hide_secrets` guardrail check did not run on this request because api key=sk-jNm1Zar7XfNdZXp49Z1kSQ has `"permissions": {"hide_secrets": false}`
:::


## Content Moderation
### Content Moderation with LLM Guard

Set the LLM Guard API Base in your environment

```env
LLM_GUARD_API_BASE = "http://0.0.0.0:8192" # deployed llm guard api
```

Add `llmguard_moderations` as a callback

```yaml
litellm_settings:
    callbacks: ["llmguard_moderations"]
```

Now you can easily test it

- Make a regular /chat/completion call

- Check your proxy logs for any statement with `LLM Guard:`

Expected results:

```
LLM Guard: Received response - {"sanitized_prompt": "hello world", "is_valid": true, "scanners": { "Regex": 0.0 }}
```
#### Turn on/off per key

**1. Update config**
```yaml
litellm_settings:
    callbacks: ["llmguard_moderations"]
    llm_guard_mode: "key-specific"
```

**2. Create new key**

```bash
curl --location 'http://localhost:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "models": ["fake-openai-endpoint"],
    "permissions": {
        "enable_llm_guard_check": true # 👈 KEY CHANGE
    }
}'

# Returns {..'key': 'my-new-key'}
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer my-new-key' \
--data '{"model": "fake-openai-endpoint", "messages": [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "What do you know?"}
    ]
    }'
```

#### Turn on/off per request

**1. Update config**
```yaml
litellm_settings:
    callbacks: ["llmguard_moderations"]
    llm_guard_mode: "request-specific"
```

**2. Create new key**

```bash
curl --location 'http://localhost:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "models": ["fake-openai-endpoint"]
}'

# Returns {..'key': 'my-new-key'}
```

**3. Test it!**

<Tabs>
<TabItem value="openai" label="OpenAI Python v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={
        "metadata": {
            "permissions": {
                "enable_llm_guard_check": True # 👈 KEY CHANGE
            },
        }
    }
)

print(response)
```
</TabItem>
<TabItem value="curl" label="Curl Request">

```bash
curl --location 'http://0.0.0.0:4000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer my-new-key' \
--data '{"model": "fake-openai-endpoint", "messages": [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "What do you know?"}
    ]
    }'
```

</TabItem>
</Tabs>

### Content Moderation with LlamaGuard

Currently works with Sagemaker's LlamaGuard endpoint.

How to enable this in your config.yaml:

```yaml
litellm_settings:
   callbacks: ["llamaguard_moderations"]
   llamaguard_model_name: "sagemaker/jumpstart-dft-meta-textgeneration-llama-guard-7b"
```

Make sure you have the relevant keys in your environment, eg.:

```
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""
```

#### Customize LlamaGuard prompt

To modify the unsafe categories llama guard evaluates against, just create your own version of [this category list](https://github.com/BerriAI/litellm/blob/main/litellm/proxy/llamaguard_prompt.txt)

Point your proxy to it

```yaml
callbacks: ["llamaguard_moderations"]
  llamaguard_model_name: "sagemaker/jumpstart-dft-meta-textgeneration-llama-guard-7b"
  llamaguard_unsafe_content_categories: /path/to/llamaguard_prompt.txt
```



### Content Moderation with Google Text Moderation

Requires your GOOGLE_APPLICATION_CREDENTIALS to be set in your .env (same as VertexAI).

How to enable this in your config.yaml:

```yaml
litellm_settings:
   callbacks: ["google_text_moderation"]
```

#### Set custom confidence thresholds

Google Moderations checks the test against several categories. [Source](https://cloud.google.com/natural-language/docs/moderating-text#safety_attribute_confidence_scores)

#### Set global default confidence threshold

By default this is set to 0.8. But you can override this in your config.yaml.

```yaml
litellm_settings:
    google_moderation_confidence_threshold: 0.4
```

#### Set category-specific confidence threshold

Set a category specific confidence threshold in your config.yaml. If none set, the global default will be used.

```yaml
litellm_settings:
    toxic_confidence_threshold: 0.1
```

Here are the category specific values:

| Category | Setting |
| -------- | -------- |
| "toxic" | toxic_confidence_threshold: 0.1 |
| "insult" | insult_confidence_threshold: 0.1 |
| "profanity" | profanity_confidence_threshold: 0.1 |
| "derogatory" | derogatory_confidence_threshold: 0.1 |
| "sexual" | sexual_confidence_threshold: 0.1 |
| "death_harm_and_tragedy" | death_harm_and_tragedy_threshold: 0.1 |
| "violent" | violent_threshold: 0.1 |
| "firearms_and_weapons" | firearms_and_weapons_threshold: 0.1 |
| "public_safety" | public_safety_threshold: 0.1 |
| "health" | health_threshold: 0.1 |
| "religion_and_belief" | religion_and_belief_threshold: 0.1 |
| "illicit_drugs" | illicit_drugs_threshold: 0.1 |
| "war_and_conflict" | war_and_conflict_threshold: 0.1 |
| "politics" | politics_threshold: 0.1 |
| "finance" | finance_threshold: 0.1 |
| "legal" | legal_threshold: 0.1 |


## Swagger Docs - Custom Routes + Branding

:::info

Requires a LiteLLM Enterprise key to use. Get a free 2-week license [here](https://forms.gle/sTDVprBs18M4V8Le8)

:::

Set LiteLLM Key in your environment

```bash
LITELLM_LICENSE=""
```

#### Customize Title + Description

In your environment, set:

```bash
DOCS_TITLE="TotalGPT"
DOCS_DESCRIPTION="Sample Company Description"
```

#### Customize Routes

Hide admin routes from users.

In your environment, set:

```bash
DOCS_FILTERED="True" # only shows openai routes to user
```

<Image img={require('../img/custom_swagger.png')}  style={{ width: '900px', height: 'auto' }} />


## Enable Blocked User Lists
If any call is made to proxy with this user id, it'll be rejected - use this if you want to let users opt-out of ai features

```yaml
litellm_settings:
     callbacks: ["blocked_user_check"]
     blocked_user_list: ["user_id_1", "user_id_2", ...]  # can also be a .txt filepath e.g. `/relative/path/blocked_list.txt`
```

### How to test

<Tabs>


<TabItem value="openai" label="OpenAI Python v1.0.0+">

Set `user=<user_id>` to the user id of the user who might have opted out.

```python
import openai
client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    user="user_id_1"
)

print(response)
```
</TabItem>

<TabItem value="Curl" label="Curl Request">

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ],
      "user": "user_id_1"
    }
'
```

</TabItem>
</Tabs>

:::info

[Suggest a way to improve this](https://github.com/BerriAI/litellm/issues/new/choose)

:::

### Using via API


**Block all calls for a customer id**

```
curl -X POST "http://0.0.0.0:4000/customer/block" \
-H "Authorization: Bearer sk-1234" \
-D '{
"user_ids": [<user_id>, ...]
}'
```

**Unblock calls for a user id**

```
curl -X POST "http://0.0.0.0:4000/user/unblock" \
-H "Authorization: Bearer sk-1234" \
-D '{
"user_ids": [<user_id>, ...]
}'
```



## Enable Banned Keywords List

```yaml
litellm_settings:
     callbacks: ["banned_keywords"]
     banned_keywords_list: ["hello"] # can also be a .txt file - e.g.: `/relative/path/keywords.txt`
```

### Test this

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": "Hello world!"
        }
      ]
    }
'
```

## Public AI Hub

Share a public page of available models and agents for users

[Learn more](./proxy/ai_hub.md)

<Image img={require('../img/model_hub.png')} style={{ width: '900px', height: 'auto' }}/>


## [BETA] AWS Key Manager - Key Decryption

This is a beta feature, and subject to changes.


**Step 1.** Add `USE_AWS_KMS` to env

```env
USE_AWS_KMS="True"
```

**Step 2.** Add `LITELLM_SECRET_AWS_KMS_` to encrypted keys in env

```env
LITELLM_SECRET_AWS_KMS_DATABASE_URL="AQICAH.."
```

LiteLLM will find this and use the decrypted `DATABASE_URL="postgres://.."` value in runtime.

**Step 3.** Start proxy

```
$ litellm
```

How it works?
- Key Decryption runs before server starts up. [**Code**](https://github.com/BerriAI/litellm/blob/8571cb45e80cc561dc34bc6aa89611eb96b9fe3e/litellm/proxy/proxy_cli.py#L445)
- It adds the decrypted value to the `os.environ` for the python process.

**Note:** Setting an environment variable within a Python script using os.environ will not make that variable accessible via SSH sessions or any other new processes that are started independently of the Python script. Environment variables set this way only affect the current process and its child processes.


## Set Max Request / Response Size on LiteLLM Proxy

Use this if you want to set a maximum request / response size for your proxy server. If a request size is above the size it gets rejected + slack alert triggered

#### Usage
**Step 1.** Set `max_request_size_mb` and `max_response_size_mb`

For this example we set a very low limit on `max_request_size_mb` and expect it to get rejected

:::info
In production we recommend setting a `max_request_size_mb` /  `max_response_size_mb` around `32 MB`

:::

```yaml
model_list:
  - model_name: fake-openai-endpoint
    litellm_params:
      model: openai/fake
      api_key: fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/
general_settings:
  master_key: sk-1234

  # Security controls
  max_request_size_mb: 0.000000001 # 👈 Key Change - Max Request Size in MB. Set this very low for testing
  max_response_size_mb: 100 # 👈 Key Change - Max Response Size in MB
```

**Step 2.** Test it with `/chat/completions` request

```shell
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "fake-openai-endpoint",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

**Expected Response from request**
We expect this to fail since the request size is over `max_request_size_mb`
```shell
{"error":{"message":"Request size is too large. Request size is 0.0001125335693359375 MB. Max size is 1e-09 MB","type":"bad_request_error","param":"content-length","code":400}}
```

---
