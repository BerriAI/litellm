import Image from '@theme/IdealImage';

# Enterprise
For companies that need SSO, user management and professional support for LiteLLM Proxy

:::info
Get free 7-day trial key [here](https://www.litellm.ai/#trial)
:::

Includes all enterprise features.

<Image img={require('../img/enterprise_vs_oss.png')} />

[**Procurement available via AWS / Azure Marketplace**](./data_security.md#legalcompliance-faqs)


This covers: 
- [**Enterprise Features**](./proxy/enterprise)
- ✅ **Feature Prioritization**
- ✅ **Custom Integrations**
- ✅ **Professional Support - Dedicated discord + slack**


Deployment Options:

**Self-Hosted**
1. Manage Yourself - you can deploy our Docker Image or build a custom image from our pip package, and manage your own infrastructure. In this case, we would give you a license key + provide support via a dedicated support channel. 

2. We Manage - you give us subscription access on your AWS/Azure/GCP account, and we manage the deployment.

**Managed**

You can use our cloud product where we setup a dedicated instance for you. 

## Frequently Asked Questions

### SLA's + Professional Support

Professional Support can assist with LLM/Provider integrations, deployment, upgrade management, and LLM Provider troubleshooting.  We can’t solve your own infrastructure-related issues but we will guide you to fix them.

- 1 hour for Sev0 issues - 100% production traffic is failing
- 6 hours for Sev1 - <100% production traffic is failing
- 24h for Sev2-Sev3 between 7am – 7pm PT (Monday through Saturday) - setup issues e.g. Redis working on our end, but not on your infrastructure.
- 72h SLA for patching vulnerabilities in the software. 

**We can offer custom SLAs** based on your needs and the severity of the issue

### What’s the cost of the Self-Managed Enterprise edition?

Self-Managed Enterprise deployments require our team to understand your exact needs. [Get in touch with us to learn more](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)


### How does deployment with Enterprise License work? 

You just deploy [our docker image](https://docs.litellm.ai/docs/proxy/deploy) and get an enterprise license key to add to your environment to unlock additional functionality (SSO, Prometheus metrics, etc.). 

```env
LITELLM_LICENSE="eyJ..."
```

No data leaves your environment. 

## Data Security / Legal / Compliance FAQs

[Data Security / Legal / Compliance FAQs](./data_security.md)