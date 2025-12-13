---
sidebar_label: "Cursor IDE"
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Cursor IDE Integration with LiteLLM

This tutorial shows you how to integrate Cursor IDE with LiteLLM Proxy, allowing you to use any LiteLLM-supported model through Cursor's interface with BYOK (Bring Your Own Key) and custom base URL.

## Benefits of using Cursor with LiteLLM


## Setup Guide

Follow these steps to connect Cursor IDE to your LiteLLM Proxy.

### Step 1: Open Cursor Settings

1. Click **Cursor** in the menu bar
2. Select **Cursor Settings** from the dropdown menu

![Click Cursor menu](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/35918062-56a6-4706-a2c3-a5df6cb93d18/ascreenshot.jpeg?tl_px=0,0&br_px=1376,769&force_format=jpeg&q=100&width=1120.0)

![Select Cursor Settings](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/f725f154-588d-448d-a1d7-3c8bffaf3cf3/ascreenshot.jpeg?tl_px=0,0&br_px=1376,769&force_format=jpeg&q=100&width=1120.0)

### Step 2: Navigate to Models Settings

1. In the Cursor Settings panel, click on **Models** in the left sidebar

![Cursor Settings General tab](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/aaeee682-f4cc-4085-9b9c-1ce22aa14119/ascreenshot.jpeg?tl_px=0,0&br_px=1376,769&force_format=jpeg&q=100&width=1120.0)

### Step 3: Configure OpenAI API Key and Base URL

1. Scroll down to the **API Keys** section
2. Enter your LiteLLM Virtual Key in the **OpenAI API Key** field
3. Enable **Override OpenAI Base URL**
4. Set the base URL to your LiteLLM Proxy URL with the `/cursor` endpoint:
   ```
   https://your-litellm-proxy.com/cursor
   ```

![API Keys section with Override OpenAI Base URL](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/76c52bec-e182-49ae-82a3-3fa6a1827665/ascreenshot.jpeg?tl_px=162,347&br_px=1538,1117&force_format=jpeg&q=100&width=1120.0)

![Override OpenAI Base URL configuration](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/be616530-952a-4ed9-9cfe-dea505e7cfa6/ascreenshot.jpeg?tl_px=297,286&br_px=1673,1055&force_format=jpeg&q=100&width=1120.0)

### Step 4: Create a LiteLLM Virtual Key (if you don't have one)

1. Open your LiteLLM Dashboard in a browser
2. Navigate to **Virtual Keys** in the left sidebar
3. Click **+ Create New Key**

![LiteLLM Dashboard - Virtual Keys page](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/1d8156bc-1b12-433f-936d-77f876142e3f/ascreenshot.jpeg?tl_px=0,0&br_px=1376,769&force_format=jpeg&q=100&width=1120.0)

4. Fill in the key details:
   - **Key Name**: Enter a descriptive name (e.g., `cursor-test-key`)
   - **Models**: Select the models you want to use (e.g., `gpt-5.1-openai`)
5. Click **Create Key**

![Create new key form](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/c9a7ae9f-8b67-4299-926a-0f1d320fb97c/ascreenshot.jpeg?tl_px=352,347&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0)

6. Copy the generated Virtual Key (you will not be able to view it again)

![Copy Virtual Key dialog](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/4022504d-fdba-4e17-b16e-bf8e935cbcad/ascreenshot.jpeg?tl_px=0,101&br_px=1376,870&force_format=jpeg&q=100&width=1120.0)

### Step 5: Get Your Model Name from LiteLLM

1. In the LiteLLM Dashboard, navigate to **Models + Endpoints**
2. Find the model you want to use and copy its **Public Model Name**

![Models + Endpoints page](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/ffb217d8-605f-4160-88ac-9386889042ab/ascreenshot.jpeg?tl_px=0,0&br_px=1376,769&force_format=jpeg&q=100&width=1120.0)

![Model details showing Public Model Name](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/2ee87f64-104a-4b37-8041-c92130a44896/ascreenshot.jpeg?tl_px=0,11&br_px=1376,780&force_format=jpeg&q=100&width=1120.0)

### Step 6: Add Custom Model in Cursor

1. Back in Cursor Settings > Models, click **Add Custom Model**
2. Enter the Public Model Name from LiteLLM (e.g., `gpt-5.1-openai`)
3. Enable the model using the toggle

![Cursor Models settings with custom model](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/fd9b9bac-89a2-4337-bb12-625d848edf7c/ascreenshot.jpeg?tl_px=0,0&br_px=1728,965&force_format=jpeg&q=100&width=1120.0)

### Step 7: Start Using Cursor with LiteLLM

1. Open a new chat in Cursor (Cmd+L on Mac, Ctrl+L on Windows/Linux)
2. Select your custom model from the model dropdown
3. Start chatting - your requests will now route through LiteLLM Proxy

![Cursor chat with LiteLLM model selected](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/05a5853a-58ed-44bf-a5c2-c14f9003eace/ascreenshot.jpeg?tl_px=0,151&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0)

## Configuration Summary

| Setting | Value |
|---------|-------|
| OpenAI API Key | Your LiteLLM Virtual Key (e.g., `sk-...`) |
| Override OpenAI Base URL | `https://your-litellm-proxy.com/cursor` |
| Custom Model Name | Public Model Name from LiteLLM Dashboard |

## Troubleshooting

### Connection Issues

- Verify your LiteLLM Proxy is running and accessible
- Ensure the `/cursor` endpoint is included in the base URL
- Check that your Virtual Key has access to the selected models

### Model Not Working

- Confirm the model name in Cursor matches the Public Model Name in LiteLLM exactly
- Verify the model is enabled in your Virtual Key's allowed models list

### Authentication Errors

- Regenerate your Virtual Key if it has expired
- Ensure you copied the full key including the `sk-` prefix

