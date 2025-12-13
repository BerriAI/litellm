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

Open Cursor IDE, click **Cursor** in the menu bar, then go to **Settings > Cursor Settings**. Click **Models** in the left sidebar.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/f725f154-588d-448d-a1d7-3c8bffaf3cf3/ascreenshot.jpeg?tl_px=0,0&br_px=1376,769&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=263,73)

Scroll down to the **API Keys** section. Enable **Override OpenAI Base URL** and enter your LiteLLM Proxy URL with `/cursor` appended:

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/890f7fd8-8489-45ca-bfe9-40e3ee1d7470/ascreenshot.jpeg?tl_px=0,151&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0)

### 2. Adding the Virtual Key

In the LiteLLM Dashboard, go to **Virtual Keys** and click **+ Create New Key**.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/1d8156bc-1b12-433f-936d-77f876142e3f/ascreenshot.jpeg?tl_px=0,0&br_px=1376,769&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=240,182)

Enter a key name and select the models you want to access from Cursor.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/c45843db-b623-442b-b42b-3145ef3ba986/ascreenshot.jpeg?tl_px=0,151&br_px=1376,920&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=453,277)

Click **Create Key**, then **Copy Virtual Key** immediately - you won't see this key again.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/4022504d-fdba-4e17-b16e-bf8e935cbcad/ascreenshot.jpeg?tl_px=0,101&br_px=1376,870&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=512,277)

Back in Cursor Settings, paste the key into the **OpenAI API Key** field.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/6b50fc92-9219-4868-aac2-a29d0c063e57/ascreenshot.jpeg?tl_px=251,235&br_px=1627,1004&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,276)

### 3. Adding the Model

In Cursor Settings > Models, click **+ Add Custom Model**.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/4e46538e-a876-44c4-a133-bdae664510f3/ascreenshot.jpeg?tl_px=192,8&br_px=1569,777&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,276)

In the LiteLLM Dashboard, go to **Models + Endpoints** and find the **Public Model Name** for your model.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/2ee87f64-104a-4b37-8041-c92130a44896/ascreenshot.jpeg?tl_px=0,11&br_px=1376,780&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=331,277)

Copy the model name, paste it in Cursor, and enable the toggle.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/5ab35f93-d417-423f-a359-9811ce18e2c3/ascreenshot.jpeg?tl_px=352,26&br_px=1728,795&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=786,277)

### 4. Testing the Setup

Press **Cmd+L** (Mac) or **Ctrl+L** (Windows) to open a chat. Select your model from the dropdown.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/29ef55af-1c93-4ec5-9886-dc867a45d811/ascreenshot.jpeg?tl_px=352,347&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=734,471)

Send a message to verify the connection. All requests now route through LiteLLM.

![](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/05a5853a-58ed-44bf-a5c2-c14f9003eace/ascreenshot.jpeg?tl_px=0,151&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0)

## Configuration Summary

| Setting | Value |
|---------|-------|
| OpenAI API Key | Your LiteLLM Virtual Key (starts with `sk-`) |
| Override OpenAI Base URL | `<LITELLM_PROXY_BASE_URL>/cursor` |
| Custom Model Name | Public Model Name from LiteLLM Dashboard |

## Troubleshooting

### Model not responding
- Verify the Virtual Key has access to the selected model
- Check that the base URL includes `/cursor` at the end
- Ensure your LiteLLM Proxy is running and accessible

### Authentication errors
- Regenerate the Virtual Key if it has expired
- Make sure you copied the full key including the `sk-` prefix

### Agent mode not working
- This is expected. Cursor does not currently support custom API keys for Agent mode.
- Only **Ask** and **Plan** modes work with custom keys.

---

*[Made with Scribe](https://scribehow.com/shared/Cursor_x_LiteLLM__ahq5Mhi5QH6-TFtSV37PDg)*
