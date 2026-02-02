import Image from '@theme/IdealImage';

# Enterprise

:::info
- ✨ SSO is free for up to 5 users. After that, an enterprise license is required. [Get Started with Enterprise here](https://www.litellm.ai/enterprise)
- Who is Enterprise for? Companies giving access to 100+ users **OR** 10+ AI use-cases. If you're not sure, [get in touch with us](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat) to discuss your needs.
:::

For companies that need SSO, user management and professional support for LiteLLM Proxy

:::info
Get free 7-day trial key [here](https://www.litellm.ai/enterprise#trial)
:::

## Enterprise Features

Includes all enterprise features.

<Image img={require('../img/enterprise_vs_oss_2.png')} />

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

You just deploy [our docker image](https://docs.litellm.ai/docs/proxy/deploy) and get an enterprise license key to add to your environment to unlock additional functionality (SSO, etc.). 

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

### How to set up your Enterprise License (`LITELLM_LICENSE`)

After purchasing an Enterprise license, you will receive a license key (a JWT string starting with `eyJ...`). To activate Enterprise features on your self-hosted LiteLLM Proxy, add the key to your environment:

**Step 1: Add `LITELLM_LICENSE` to your `.env` file**

```env
LITELLM_LICENSE="eyJ..."
```

Or if you're using Docker, pass it as an environment variable:

```bash
docker run -e LITELLM_LICENSE="eyJ..." ghcr.io/berriai/litellm:main-latest
```

Or add it to your `docker-compose.yml`:

```yaml
environment:
  - LITELLM_LICENSE=eyJ...
```

**Step 2: Restart LiteLLM Proxy**

After setting the environment variable, restart your LiteLLM Proxy so it picks up the new license key.

### How to verify your Enterprise License is active

Once LiteLLM Proxy has restarted with the `LITELLM_LICENSE` set, you can verify the license is applied by checking the Swagger API docs:

1. Open your browser and navigate to your LiteLLM Proxy's API docs page:
   ```
   http://<your-proxy-host>:<port>/
   ```
2. Look at the **description text** at the top of the Swagger page.
3. If the license is valid, you will see **"Enterprise Edition"** in the description, for example:
   > *Enterprise Edition*
   >
   > *Proxy Server to call 100+ LLMs in the OpenAI format.*
4. If you do **not** see "Enterprise Edition", the description will simply say:
   > *Proxy Server to call 100+ LLMs in the OpenAI format.*

:::tip
If "Enterprise Edition" is not showing up after setting `LITELLM_LICENSE`:
- Double-check that the license key is correct and has not expired.
- Ensure there are no extra spaces or quotes around the key.
- Confirm that the proxy was fully restarted after setting the variable.
- Check the proxy logs for any license validation errors.
:::

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
