# Cursor x LiteLLM

This tutorial walks you through connecting Cursor IDE to your LiteLLM Proxy, enabling you to use any model configured on your proxy directly within Cursor.

## Overview

**Supported Cursor Modes**: Ask, Plan

**Note**: Cursor does not currently support custom API keys for Agent mode. We are working with the Cursor team to add support for this.

## Base URL Format

When configuring Cursor to use LiteLLM, you must use the `/cursor` endpoint:

```
<LITELLM_PROXY_BASE_URL>/cursor
```

**Examples**:
- `https://my-proxy.example.com/cursor`
- `https://litellm-prod.ngrok-free.dev/cursor`
- `http://localhost:4000/cursor`

Cursor automatically appends `/chat/completions` to the base URL, so the full request path becomes `/cursor/chat/completions`.

## Prerequisites

- A running LiteLLM Proxy instance
- Access to the LiteLLM Dashboard
- Cursor IDE installed

## Step-by-Step Guide

### 1. Adding the LiteLLM URL

#### 1.1 Open Cursor Settings

In Cursor IDE, click **"Cursor"** in the top-left menu bar to open the application menu.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/35918062-56a6-4706-a2c3-a5df6cb93d18/ascreenshot.jpeg?tl_px=0,0&br_px=1376,769&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=42,-28)

From the dropdown, hover over **Settings** and click **"Cursor Settings"** to open the settings panel.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/f725f154-588d-448d-a1d7-3c8bffaf3cf3/ascreenshot.jpeg?tl_px=0,0&br_px=1376,769&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=263,73)

#### 1.2 Navigate to Models Settings

The Cursor Settings panel opens to the General tab. Click **"Models"** in the left sidebar to access model and API configuration.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/aaeee682-f4cc-4085-9b9c-1ce22aa14119/ascreenshot.jpeg?tl_px=0,0&br_px=1376,769&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=454,230)

Scroll down to the **API Keys** section. You'll see fields for OpenAI API Key, Override OpenAI Base URL, Anthropic API Key, and other providers.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/76c52bec-e182-49ae-82a3-3fa6a1827665/ascreenshot.jpeg?tl_px=162,347&br_px=1538,1117&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,388)

#### 1.3 Configure the Base URL

Open your LiteLLM Dashboard in a browser. Copy the base URL from the address bar (e.g., `https://your-proxy.ngrok-free.dev`). Press **Cmd+C** (Mac) or **Ctrl+C** (Windows) to copy.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/60d630d6-5c4c-403d-98c9-2d39f303c368/ascreenshot.jpeg?tl_px=0,0&br_px=1728,965&force_format=jpeg&q=100&width=1120.0)

Back in Cursor Settings, click on the **Override OpenAI Base URL** input field to select it.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/be616530-952a-4ed9-9cfe-dea505e7cfa6/ascreenshot.jpeg?tl_px=297,286&br_px=1673,1055&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,277)

Paste your LiteLLM Proxy URL and **append `/cursor`** to the end. The final URL should be:

```
<LITELLM_PROXY_BASE_URL>/cursor
```

For example: `https://your-proxy.ngrok-free.dev/cursor`

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/890f7fd8-8489-45ca-bfe9-40e3ee1d7470/ascreenshot.jpeg?tl_px=0,151&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0)

### 2. Adding the Virtual Key

#### 2.1 Open LiteLLM Dashboard

Now you need to create a Virtual Key in LiteLLM. Open a new browser tab and navigate to your LiteLLM Dashboard.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/6580de2b-3a59-45b2-b7b6-3ab105d87e74/ascreenshot.jpeg?tl_px=138,285&br_px=1515,1054&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,276)

In the LiteLLM Dashboard, click on **"Virtual Keys"** in the left sidebar under the AI GATEWAY section.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/a289f9e4-05cf-4a2c-a3a2-84a14b58a32a/ascreenshot.jpeg?tl_px=352,231&br_px=1728,1000&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=840,276)

#### 2.2 Create a New Key

Click the **"+ Create New Key"** button at the top of the Virtual Keys page.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/1d8156bc-1b12-433f-936d-77f876142e3f/ascreenshot.jpeg?tl_px=0,0&br_px=1376,769&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=240,182)

In the Create Key modal, click on the **"Key Name"** field to enter a descriptive name for your key.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/ee9f46e7-a52f-4db7-88e3-0655f21bf064/ascreenshot.jpeg?tl_px=44,105&br_px=1420,874&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,277)

Type a name for your key (e.g., `cursor-test-key`). This helps you identify the key later in the dashboard.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/0840d626-7f0c-4c20-9c1f-304dee84e046/ascreenshot.jpeg?tl_px=35,166&br_px=1411,935&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,277)

Click on the **"Models"** dropdown to select which models this key should have access to.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/44d816cb-e035-43a0-8d88-adf6150f6f73/ascreenshot.jpeg?tl_px=100,347&br_px=1477,1117&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,322)

Select the model(s) you want to use with Cursor (e.g., `gpt-5.1-openai`). You can select multiple models if needed.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/c45843db-b623-442b-b42b-3145ef3ba986/ascreenshot.jpeg?tl_px=0,151&br_px=1376,920&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=453,277)

Click the **"Create Key"** button to generate your new Virtual Key.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/c9a7ae9f-8b67-4299-926a-0f1d320fb97c/ascreenshot.jpeg?tl_px=352,347&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=736,293)

#### 2.3 Copy the Virtual Key

A modal appears with your new Virtual Key. **Important**: Click **"Copy Virtual Key"** immediately - you won't be able to see this key again after closing the dialog.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/4022504d-fdba-4e17-b16e-bf8e935cbcad/ascreenshot.jpeg?tl_px=0,101&br_px=1376,870&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=512,277)

#### 2.4 Paste the Key in Cursor

Go back to Cursor Settings > Models. Click on the **"OpenAI API Key"** input field.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/b1239362-013b-4913-9d9f-7f0d24583e64/ascreenshot.jpeg?tl_px=280,164&br_px=1656,933&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,277)

Double-click the field to select any existing content, then paste your Virtual Key.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/30eb80f7-916c-462d-8716-0441ce42f0c6/ascreenshot.jpeg?tl_px=352,117&br_px=1728,886&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=839,277)

Verify that the **Override OpenAI Base URL** toggle is enabled (green) and contains your LiteLLM Proxy URL with `/cursor` appended.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/6b50fc92-9219-4868-aac2-a29d0c063e57/ascreenshot.jpeg?tl_px=251,235&br_px=1627,1004&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,276)

### 3. Adding the Model

#### 3.1 View Available Models in Cursor

Scroll up in the Models settings to see the list of available models. You'll see built-in models like GPT-5 Pro, Gemini 2.5 Flash, Claude variants, etc.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/f3c4381e-2589-40e9-9938-db3dc5c4d501/ascreenshot.jpeg?tl_px=178,6&br_px=1554,775&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,277)

Click **"+ Add Custom Model"** to add your LiteLLM model to Cursor.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/4e46538e-a876-44c4-a133-bdae664510f3/ascreenshot.jpeg?tl_px=192,8&br_px=1569,777&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,276)

A text field appears for entering your custom model name. You'll need to get the exact model name from your LiteLLM Dashboard.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/8d4d52ae-8d09-4139-9693-216b9a820feb/ascreenshot.jpeg?tl_px=191,21&br_px=1567,790&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,277)

#### 3.2 Get the Model Name from LiteLLM Dashboard

Go back to the LiteLLM Dashboard. Click **"Models + Endpoints"** in the left sidebar to see all configured models.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/ffb217d8-605f-4160-88ac-9386889042ab/ascreenshot.jpeg?tl_px=0,0&br_px=1376,769&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=90,257)

Click on a model row to view its details. Each model has a unique ID shown in the list.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/a872e7c9-f88a-4132-a8a7-782ef24d321d/ascreenshot.jpeg?tl_px=0,347&br_px=1376,1117&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=240,416)

On the model detail page, find the **"Public Model Name"** (e.g., `gpt-5.1-openai`). Double-click to select it.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/2ee87f64-104a-4b37-8041-c92130a44896/ascreenshot.jpeg?tl_px=0,11&br_px=1376,780&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=331,277)

Press **Cmd+C** (Mac) or **Ctrl+C** (Windows) to copy the Public Model Name.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/20a363d3-ee8a-4d0a-8407-75b45cdb9f12/ascreenshot.jpeg?tl_px=0,0&br_px=1728,965&force_format=jpeg&q=100&width=1120.0)

#### 3.3 Add the Custom Model in Cursor

Go back to Cursor Settings > Models. In the custom model text field, press **Cmd+V** (Mac) or **Ctrl+V** (Windows) to paste the model name.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/240b1172-875f-471e-bb82-2fe7c5f07f43/ascreenshot.jpeg?tl_px=0,0&br_px=1728,965&force_format=jpeg&q=100&width=1120.0)

Click the toggle next to your custom model to **enable** it. The toggle should turn green when enabled.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/5ab35f93-d417-423f-a359-9811ce18e2c3/ascreenshot.jpeg?tl_px=352,26&br_px=1728,795&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=786,277)

### 4. Testing the Setup

#### 4.1 Open a Chat Panel

Press **Cmd+L** (Mac) or **Ctrl+L** (Windows) to open a new Cursor chat panel. This opens the **Ask** mode where you can interact with your model.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/fd9b9bac-89a2-4337-bb12-625d848edf7c/ascreenshot.jpeg?tl_px=0,0&br_px=1728,965&force_format=jpeg&q=100&width=1120.0)

#### 4.2 Select Your Model

Click on the model selector dropdown at the bottom of the chat panel to see available models.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/a48b3e90-b71e-457f-9e8f-19c508e0264b/ascreenshot.jpeg?tl_px=352,347&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=751,542)

Select your custom model (e.g., `gpt-5.1-openai`) from the dropdown list.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/29ef55af-1c93-4ec5-9886-dc867a45d811/ascreenshot.jpeg?tl_px=352,347&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=734,471)

#### 4.3 Send a Test Message

Type a message in the chat input and press **Return/Enter** to send it. Your request will route through LiteLLM Proxy.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/05a5853a-58ed-44bf-a5c2-c14f9003eace/ascreenshot.jpeg?tl_px=0,151&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0)

The model responds through LiteLLM. You can continue the conversation as normal.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/b7d524ab-cbe4-4992-b5ed-0625f4d6e258/ascreenshot.jpeg?tl_px=352,347&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=698,552)

#### 4.4 Verify the Integration

You can ask follow-up questions or start new conversations. All requests go through your LiteLLM Proxy.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/0d867cb9-d86f-4792-ad5a-95b1bbb4c5cc/ascreenshot.jpeg?tl_px=352,347&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=696,526)

Press **Return/Enter** to send another message and verify the integration is working correctly.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/d87ee25b-3c6d-4231-ba00-4d841d0612bc/ascreenshot.jpeg?tl_px=0,151&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0)

Your Cursor IDE is now connected to LiteLLM. All chat requests in **Ask** and **Plan** modes will route through your proxy, giving you centralized logging, budget controls, and access to any model configured on your LiteLLM instance.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/a15d321c-9eb1-4466-ab79-8bc4eadb70cf/ascreenshot.jpeg?tl_px=352,347&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=693,552)

## Configuration Summary

| Setting | Value |
|---------|-------|
| OpenAI API Key | Your LiteLLM Virtual Key (starts with `sk-`) |
| Override OpenAI Base URL | `<LITELLM_PROXY_BASE_URL>/cursor` |
| Custom Model Name | Public Model Name from LiteLLM Dashboard |

## Supported Modes

| Cursor Mode | Supported | Notes |
|-------------|-----------|-------|
| Ask | Yes | Full support via custom API key |
| Plan | Yes | Full support via custom API key |
| Agent | No | Cursor does not support custom keys for Agent mode. We are working with the Cursor team to add support. |

## Benefits of Using LiteLLM with Cursor

- **Centralized Logging**: All requests are logged in your LiteLLM Dashboard
- **Budget Controls**: Set spending limits per key, team, or user
- **Model Flexibility**: Switch between any model configured on your proxy
- **Rate Limiting**: Control request rates to prevent abuse
- **Observability**: Track usage, costs, and performance metrics

## Troubleshooting

### Model not responding
- Verify the Virtual Key has access to the selected model
- Check that the base URL includes `/cursor` at the end (e.g., `https://my-proxy.com/cursor`)
- Ensure your LiteLLM Proxy is running and accessible

### Authentication errors
- Regenerate the Virtual Key if it has expired
- Make sure you copied the full key including the `sk-` prefix

### Connection issues
- Verify your LiteLLM Proxy URL is correct and includes `/cursor`
- Check network connectivity to the proxy
- Review proxy logs for error messages

### Agent mode not working
- This is expected. Cursor does not currently support custom API keys for Agent mode.
- Only **Ask** and **Plan** modes work with custom keys.

---

*[Made with Scribe](https://scribehow.com/shared/Cursor_x_LiteLLM__ahq5Mhi5QH6-TFtSV37PDg)*
