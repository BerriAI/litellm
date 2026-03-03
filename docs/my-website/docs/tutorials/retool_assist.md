import Image from '@theme/IdealImage';

# Retool Assist

This guide walks you through connecting [Retool Assist](https://docs.retool.com/apps/guides/assist/) to LiteLLM Proxy. Retool Assist uses AI to generate and edit apps from within the Retool app IDE. Using LiteLLM with Retool Assist allows you to:

- Access 100+ LLMs through Retool Assist
- Track spend and usage, set budget limits per virtual key
- Control which models Retool Assist can access
- Use your own LLM providers via a unified OpenAI-compatible API

<div style={{ maxWidth: '100%', overflow: 'hidden', paddingBottom: '59.52%', position: 'relative', height: 0 }}>
  <iframe 
    style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', maxWidth: '840px' }}
    src="https://www.youtube.com/embed/aN-Iua5dHGg" 
    frameborder="0" 
    webkitallowfullscreen 
    mozallowfullscreen 
    allowfullscreen
  ></iframe>
</div>

---

:::info
**Hosted Retool requires a public URL.** Retool Cloud runs on Retool's servers, so `localhost` will not work. You must expose your LiteLLM proxy via ngrok, Cloudflare Tunnel, or by deploying to a cloud provider.
:::

## Quick Reference

| Setting | Value |
|---------|-------|
| Provider Schema | OpenAI |
| Base URL | Your ngrok URL (e.g. `https://abc123.ngrok-free.app`) or deployed proxy URL |
| API Key | Your LiteLLM Virtual Key |
| Model | Public model name from LiteLLM (e.g. `openai/gpt-4o-mini`, `openai/gpt-5.2-2025-12-11`) |

---

## Prerequisites

- LiteLLM Proxy running locally or deployed
- [ngrok](https://ngrok.com/download) (or similar tunnel) for local development with hosted Retool
- A [Retool](https://retool.com) account (Cloud or self-hosted)

## 1. Start LiteLLM Proxy

Set up LiteLLM Proxy following the [Getting Started Guide](https://docs.litellm.ai/docs/proxy/docker_quick_start). Ensure your proxy is running on port 4000.

## 2. Expose LiteLLM with a Public URL

<Image img={require('../../img/ngrok_public_url.gif')} />

Retool Cloud runs on Retool's servers. You must expose your local LiteLLM proxy with a public URL.

### Using ngrok

- Install [ngrok](https://ngrok.com/download)
- In a separate terminal, run:
  
```bash
ngrok http 4000
```
- Copy the generated HTTPS URL (e.g. `https://abc123.ngrok-free.app`). This is your **Base URL** for Retool.


### Alternative

If you deploy LiteLLM to Railway, Render, Fly.io, or another cloud provider, use that public URL as your Base URL. See the [Deploy guide](https://docs.litellm.ai/docs/proxy/deploy) for details.

## 3. Generate a Virtual Key

<Image img={require('../../img/litellm_virtual_key.gif')} />

Create a virtual key that Retool Assist will use to authenticate with LiteLLM. The key must have access to the models you want to use (e.g. `openai/*` for all OpenAI models).

### Via LiteLLM UI

- Navigate to [http://localhost:4000/ui](http://localhost:4000/ui)
- Go to **Virtual Keys** → **+ Create New Key**
- Select the models you need (or `openai/*` for all OpenAI models)
- Copy the key

## 4. Add LiteLLM as a Custom Provider in Retool

Inside your Retool dashboard, configure LiteLLM as a custom AI resource:

<Image img={require('../../img/retool_resource_setup.gif')} />

1. Go to **Resources**

2. Under the **AI** category, select **Custom Provider**

3. Fill in the form:
   - **Name:** `LiteLLM`
   - **Description:** (optional) e.g. `LiteLLM Proxy - 100+ LLMs`
   - **Provider Schema:** `OpenAI`
   - **Base URL:** Your ngrok-generated URL (e.g. `https://abc123.ngrok-free.app`) or deployed proxy URL—do not add `/v1` unless Retool requires it
   - **API Key:** Your LiteLLM virtual key from Step 3
4. **Add model names** from your LiteLLM proxy (e.g. `openai/gpt-4o-mini`, `openai/gpt-5.2-2025-12-11`).
5. Click **Create Resource**

<Image img={require('../../img/retool_llm_setup.gif')} />

## 5. Test the Connection

<Image img={require('../../img/retool_litellm_connection.gif')} />

- Open an app in Retool and enable **Assist** (if not already enabled in your organization)
- Use Assist to generate or edit app elements, it will route requests through LiteLLM
- Use the code option from the Sidebar to add a resource query, select the LiteLLM resource, and run it to test the setup.
- Check the LiteLLM **Logs** section to verify requests and track usage

<Image img={require('../../img/retool_litellm_logs.gif')} />

---

## Troubleshooting

### 401 Unauthorized

- Ensure the **API Key** in Retool matches your LiteLLM virtual key exactly
- Verify the key is not expired or blocked in LiteLLM

### 401 "key not allowed to access model"

Your virtual key is restricted to specific models. Generate a new key with `openai/*` or include the model you need (e.g. `openai/gpt-5.2-2025-12-11`) in the key's allowed models list.

### 500 "api_key client option must be set"

LiteLLM could not use your OpenAI API key to call the provider. Ensure `OPENAI_API_KEY` is set in your LiteLLM environment (e.g. in `.env` or `docker-compose.yml`) when using `openai/*` models.

### localhost does not work

Retool Cloud cannot reach `localhost` it points to Retool's servers. Use ngrok or deploy LiteLLM to a public URL.

---

## Additional Resources

- [Virtual Keys](https://docs.litellm.ai/docs/proxy/virtual_keys) – Create and manage API keys
- [Deploy LiteLLM](https://docs.litellm.ai/docs/proxy/deploy) – Production deployment options
- [Retool Assist Documentation](https://docs.retool.com/apps/guides/assist/) – Configure Assist and prompting guides
