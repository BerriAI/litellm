# Enterprise
For companies that need SSO, user management and professional support for LiteLLM Proxy

:::info
Get free 7-day trial key [here](https://www.litellm.ai/#trial)
:::

Deploy managed LiteLLM Proxy within your VPC.

Includes all enterprise features.

[**Procurement available via AWS / Azure Marketplace**](./data_security.md#legalcompliance-faqs)

[**Get 7 day trial key**](https://www.litellm.ai/#trial)


This covers: 
- **Enterprise Features**
    - **Security**
        - ✅ [SSO for Admin UI](./proxy/ui#✨-enterprise-features)
        - ✅ [Audit Logs with retention policy](./proxy/enterprise#audit-logs)
        - ✅ [JWT-Auth](../docs/proxy/token_auth.md)
        - ✅ [Control available public, private routes (Restrict certain endpoints on proxy)](./proxy/enterprise#control-available-public-private-routes)
        - ✅ [**Secret Managers** AWS Key Manager, Google Secret Manager, Azure Key](./secret)
        - ✅ IP address‑based access control lists
        - ✅ Track Request IP Address
        - ✅ [Use LiteLLM keys/authentication on Pass Through Endpoints](./proxy/pass_through#✨-enterprise---use-litellm-keysauthentication-on-pass-through-endpoints)
        - ✅ Set Max Request / File Size on Requests
        - ✅ [Enforce Required Params for LLM Requests (ex. Reject requests missing ["metadata"]["generation_name"])](./proxy/enterprise#enforce-required-params-for-llm-requests)
    - **Customize Logging, Guardrails, Caching per project**
        - ✅ [Team Based Logging](./proxy/team_logging.md) - Allow each team to use their own Langfuse Project / custom callbacks
        - ✅ [Disable Logging for a Team](./proxy/team_logging.md#disable-logging-for-a-team) - Switch off all logging for a team/project (GDPR Compliance)
    - **Controlling Guardrails by Virtual Keys**
    - **Spend Tracking, Budgets & Data Exports**
        - ✅ [Tracking Spend for Custom Tags](./proxy/enterprise#tracking-spend-for-custom-tags)
        - ✅ [Set USD Budgets Spend for Custom Tags](./proxy/provider_budget_routing#-tag-budgets)
        - ✅ [Set Model budgets for Virtual Keys](./proxy/users#-virtual-key-model-specific)
        - ✅ [Exporting LLM Logs to GCS Bucket, Azure Blob Storage](./proxy/bucket#🪣-logging-gcs-s3-buckets)
        - ✅ [API Endpoints to get Spend Reports per Team, API Key, Customer](./proxy/cost_tracking.md#✨-enterprise-api-endpoints-to-get-spend)
    - **Prometheus Metrics**
        - ✅ [Prometheus Metrics - Num Requests, failures, LLM Provider Outages](./proxy/prometheus)
        - ✅ [`x-ratelimit-remaining-requests`, `x-ratelimit-remaining-tokens` for LLM APIs on Prometheus](./proxy/prometheus#✨-enterprise-llm-remaining-requests-and-remaining-tokens)
    - **Custom Branding**
        - ✅ [Custom Branding + Routes on Swagger Docs](./proxy/enterprise#swagger-docs---custom-routes--branding)
        - ✅ [Public Model Hub](../docs/proxy/enterprise.md#public-model-hub)
        - ✅ [Custom Email Branding](../docs/proxy/email.md#customizing-email-branding)
    - **Guardrails**
        - ✅ [Setting team/key based guardrails](./proxy/guardrails/quick_start.md#-control-guardrails-per-project-api-key)
        - ✅ [API endpoint listing available guardrails](./proxy/guardrails/bedrock.md#list-guardrails)
- ✅ **Feature Prioritization**
- ✅ **Custom Integrations**
- ✅ **Professional Support - Dedicated discord + slack**



## Frequently Asked Questions

### What topics does Professional support cover and what SLAs do you offer?

Professional Support can assist with LLM/Provider integrations, deployment, upgrade management, and LLM Provider troubleshooting.  We can’t solve your own infrastructure-related issues but we will guide you to fix them.

- 1 hour for Sev0 issues
- 6 hours for Sev1
- 24h for Sev2-Sev3 between 7am – 7pm PT (Monday through Saturday)

**We can offer custom SLAs** based on your needs and the severity of the issue

### What’s the cost of the Self-Managed Enterprise edition?

Self-Managed Enterprise deployments require our team to understand your exact needs. [Get in touch with us to learn more](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)


### How does deployment with Enterprise License work? 

You just deploy [our docker image](https://docs.litellm.ai/docs/proxy/deploy) and get an enterprise license key to add to your environment to unlock additional functionality (SSO, Prometheus metrics, etc.). 

```env
LITELLM_LICENSE="eyJ..."
```

No data leaves your environment. 