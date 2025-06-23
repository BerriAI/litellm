import Image from '@theme/IdealImage';

# Enterprise
For companies that need SSO, user management and professional support for LiteLLM Proxy

:::info
Get free 7-day trial key [here](https://www.litellm.ai/#trial)
:::

## Enterprise Features

Includes all enterprise features.

<Image img={require('../img/enterprise_vs_oss.png')} />

[**Procurement available via AWS / Azure Marketplace**](./data_security.md#legalcompliance-faqs)


This covers: 
- [**Enterprise Features**](./proxy/enterprise)
- ✅ **Feature Prioritization**
- ✅ **Custom Integrations**
- ✅ **Professional Support - Dedicated Slack/Teams channel**


## Self-Hosted

Manage Yourself - you can deploy our Docker Image or build a custom image from our pip package, and manage your own infrastructure. In this case, we would give you a license key + provide support via a dedicated support channel. 


### What’s the cost of the Self-Managed Enterprise edition?

Self-Managed Enterprise deployments require our team to understand your exact needs. [Get in touch with us to learn more](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)


### How does deployment with Enterprise License work? 

You just deploy [our docker image](https://docs.litellm.ai/docs/proxy/deploy) and get an enterprise license key to add to your environment to unlock additional functionality (SSO, Prometheus metrics, etc.). 

```env
LITELLM_LICENSE="eyJ..."
```

**No data leaves your environment.** 


## Hosted LiteLLM Proxy

LiteLLM maintains the proxy, so you can focus on your core products. 

We provide a dedicated proxy for your team, and manage the infrastructure. 

### **Status**: GA 

Our proxy is already used in production by customers. 

See our status page for [**live reliability**](https://status.litellm.ai/)

### **Benefits**
- **No Maintenance, No Infra**: We'll maintain the proxy, and spin up any additional infrastructure (e.g.: separate server for spend logs) to make sure you can load balance + track spend across multiple LLM projects. 
- **Reliable**: Our hosted proxy is tested on 1k requests per second, making it reliable for high load.
- **Secure**: LiteLLM is SOC-2 Type 2 and ISO 27001 certified, to make sure your data is as secure as possible.

### Supported data regions for LiteLLM Cloud

You can find [supported data regions litellm here](../docs/data_security#supported-data-regions-for-litellm-cloud)


## Frequently Asked Questions

### SLA's + Professional Support

Professional Support can assist with LLM/Provider integrations, deployment, upgrade management, and LLM Provider troubleshooting.  We can’t solve your own infrastructure-related issues but we will guide you to fix them.

- 1 hour for Sev0 issues - 100% production traffic is failing
- 6 hours for Sev1 - < 100% production traffic is failing
- 24h for Sev2-Sev3 between 7am – 7pm PT (Monday through Saturday) - setup issues e.g. Redis working on our end, but not on your infrastructure.
- 72h SLA for patching vulnerabilities in the software. 

**We can offer custom SLAs** based on your needs and the severity of the issue

## Data Security / Legal / Compliance FAQs

[Data Security / Legal / Compliance FAQs](./data_security.md)


### Pricing

Pricing is based on usage. We can figure out a price that works for your team, on the call. 

[**Contact Us to learn more**](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)



## **Screenshots**

### 1. Create keys

<Image img={require('../img/litellm_hosted_ui_create_key.png')} />

### 2. Add Models

<Image img={require('../img/litellm_hosted_ui_add_models.png')}/>

### 3. Track spend 

<Image img={require('../img/litellm_hosted_usage_dashboard.png')} />


### 4. Configure load balancing 

<Image img={require('../img/litellm_hosted_ui_router.png')} />
