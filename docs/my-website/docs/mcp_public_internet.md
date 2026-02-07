import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Exposing MCPs on the Public Internet

Control which MCP servers are visible to external callers (e.g., ChatGPT, Claude Desktop) vs. internal-only callers. This is useful when you want a subset of your MCP servers available publicly while keeping sensitive servers restricted to your private network.

## Overview

| Property | Details |
|-------|-------|
| Description | IP-based access control for MCP servers — external callers only see servers marked as public |
| Setting | `available_on_public_internet` on each MCP server |
| Network Config | `mcp_internal_ip_ranges` in `general_settings` |
| Supported Clients | ChatGPT, Claude Desktop, Cursor, OpenAI API, or any MCP client |

## How It Works

When a request arrives at LiteLLM's MCP endpoints, LiteLLM checks the caller's IP address to determine whether they are an **internal** or **external** caller:

1. **Extract the client IP** from the incoming request (supports `X-Forwarded-For` when configured behind a reverse proxy).
2. **Classify the IP** as internal or external by checking it against the configured private IP ranges (defaults to RFC 1918: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.0/8`).
3. **Filter the server list**:
   - **Internal callers** see all MCP servers (public and private).
   - **External callers** only see servers with `available_on_public_internet: true`.

This filtering is applied at every MCP access point: the MCP registry, tool listing, tool calling, dynamic server routes, and OAuth discovery endpoints.

```mermaid
flowchart TD
    A[Incoming MCP Request] --> B[Extract Client IP Address]
    B --> C{Is IP in private ranges?}
    C -->|Yes - Internal caller| D[Return ALL MCP servers]
    C -->|No - External caller| E[Return ONLY servers with<br/>available_on_public_internet = true]
```

## Walkthrough

This walkthrough covers two flows:
1. **Adding a public MCP server** (DeepWiki) and connecting to it from ChatGPT
2. **Making an existing server private** (Exa) and verifying ChatGPT no longer sees it

### Flow 1: Add a Public MCP Server (DeepWiki)

DeepWiki is a free MCP server — a good candidate to expose publicly so AI gateway users can access it from ChatGPT.

#### Step 1: Create the MCP Server

Navigate to the MCP Servers page and click **"+ Add New MCP Server"**.

![Click Add New MCP Server](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/28cc27c2-d980-4255-b552-ebf542ef95be/ascreenshot_30a7e3c043834f1c87b69e6ffc5bba4f_text_export.jpeg)

The create dialog opens. Enter **"DeepWiki"** as the server name.

![Enter server name](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/8c733c38-310a-40ef-8a5b-7af91cc7f74f/ascreenshot_16df83fed5bd4683a22a042e07063cec_text_export.jpeg)

For the transport type dropdown, select **HTTP** since DeepWiki uses the Streamable HTTP transport.

![Select transport type](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/e473f603-d692-40c7-a218-866c2e1cb554/ascreenshot_e93997971f2f44beac6152786889addf_text_export.jpeg)

Now scroll down to the MCP Server URL field.

![Configure server](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/b08d3c1f-9279-45b6-8efb-f73008901da6/ascreenshot_ce0de66f230a41b0a454e76653429021_text_export.jpeg)

Enter the DeepWiki MCP URL: `https://mcp.deepwiki.com/mcp`.

![Enter MCP server URL](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/e59f8285-cfde-4c57-aa79-24244acc9160/ascreenshot_8d575c66dc614a4183212ba282d22b41_text_export.jpeg)

With the name, transport, and URL filled in, the basic server configuration is complete.

![Server URL configured](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/0f1af7ed-760d-4445-bdec-3da706d4eef4/ascreenshot_d7d6db69bc254ded871d14a71188a212_text_export.jpeg)

#### Step 2: Enable "Available on Public Internet"

Before creating, scroll down and expand the **Permission Management / Access Control** section. This is where you control who can see this server.

![Expand Permission Management](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/cc10dea2-6028-4a27-a33b-1b1b7212efb5/ascreenshot_0fdd152b862a4bf39973bc805ce64c57_text_export.jpeg)

Toggle **"Available on Public Internet"** on. This is the key setting — it tells LiteLLM that external callers (like ChatGPT connecting from the public internet) should be able to discover and use this server.

![Toggle Available on Public Internet](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/39c14543-c5ae-4189-8f85-9efc87135820/ascreenshot_9991f54910c24e21bba5c05ea4fa8e28_text_export.jpeg)

With the toggle enabled, click **"Create"** to save the server.

![Click Create](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/843be209-aade-44f4-98da-e55d1644854c/ascreenshot_8cfc90345a5f4d069b397e80d0a6e449_text_export.jpeg)

#### Step 3: Connect from ChatGPT

Now let's verify it works. Open ChatGPT and look for the MCP server icon to add a new connection. The endpoint to use is `<your-litellm-url>/mcp`.

![ChatGPT add MCP server](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/58b5f674-edf4-4156-a5fa-5fdc8ed5d7b9/ascreenshot_36735f7c37394e919793968794614126_text_export.jpeg)

In the dropdown, select **"Add an MCP server"** to configure a new connection.

![ChatGPT MCP server option](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/f89da8af-bc61-44a7-a765-f52733f4970d/ascreenshot_6410a917b782437eb558de3bfcd35ffd_text_export.jpeg)

ChatGPT asks for a server label. Give it a recognizable name like "LiteLLM".

![Enter server label](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/88505afe-07c1-4674-a89c-8035a5d05eb6/ascreenshot_143aefc38ddd4d3f9f5823ca2cc09bc2_text_export.jpeg)

Next, enter the Server URL. This should be your LiteLLM proxy's MCP endpoint — `<your-litellm-url>/mcp`.

![Enter LiteLLM MCP URL](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/9048be4a-7e40-43e7-9789-059fed2741a6/ascreenshot_e81232c17fd148f48f0ae552e9dc2a10_text_export.jpeg)

Paste your LiteLLM URL and confirm it looks correct.

![URL pasted](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/7707e796-e146-47c8-bce0-58e6f4076272/ascreenshot_0710dc58b8ed4d6887856b1388d59329_text_export.jpeg)

ChatGPT also needs authentication. Enter your LiteLLM API key in the authentication field so it can connect to the proxy.

![Enter API key](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/f6cfcb81-021d-4a41-94d7-d4eaf449d025/ascreenshot_d635865abfb64732a7278922f08dbcaa_text_export.jpeg)

Click **"Connect"** to establish the connection.

![Click Connect](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/1146b326-6f0c-4050-9729-af5c88e1bc81/ascreenshot_e19fb857e5394b9a9bf77b075b4fb620_text_export.jpeg)

ChatGPT connects and shows the available tools. Since both DeepWiki and Exa are currently marked as public, ChatGPT can see tools from both servers.

![ChatGPT shows available MCP tools](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/43ac56b7-9933-4762-903a-370fc52c79b5/ascreenshot_39073d6dc3bc4bb6a79d93365a26a4f8_text_export.jpeg)

---

### Flow 2: Make an Existing Server Private (Exa)

Now let's do the reverse — take an existing MCP server (Exa) that's currently public and restrict it to internal access only. After this change, ChatGPT should no longer see Exa's tools.

#### Step 1: Edit the Server

Go to the MCP Servers table and click on the Exa server to open its detail view.

![Exa server overview](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/65844f13-b1ec-4092-b3fd-b1cae3c0c833/ascreenshot_cc8ea435c5e14761a1394ca80fe817c0_text_export.jpeg)

Switch to the **"Settings"** tab to access the edit form.

![Click Settings](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/d5b65271-561e-4d2a-b832-96d32611f6e4/ascreenshot_a200942b17264c1eb7a3ffdb2c2141f5_text_export.jpeg)

The edit form loads with Exa's current configuration.

![Edit server](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/119184f6-f3cd-45b7-9cfa-0ea08de27020/ascreenshot_c39a793da03a4f0fb84b5ee829af9034_text_export.jpeg)

#### Step 2: Toggle Off "Available on Public Internet"

Scroll down and expand the **Permission Management / Access Control** section to find the public internet toggle.

![Expand permissions](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/bf7114cc-8741-4fa0-a39a-fe625482e88a/ascreenshot_8a987649c03e46558a2ec9a6f2f539a4_text_export.jpeg)

Toggle **"Available on Public Internet"** off. This will hide Exa from any caller outside your private network.

![Toggle off public internet](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/f36af5ad-028f-4bb1-aed1-43e38ff9b733/ascreenshot_9128364a049f489bb8483e18e5c88015_text_export.jpeg)

Click **"Save Changes"** to apply. The change takes effect immediately — no proxy restart needed.

![Save changes](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/126a71b3-02e1-4d61-a208-942b92e9ef25/ascreenshot_f349ef69e08044dd8e4903f4286b7b97_text_export.jpeg)

#### Step 3: Verify in ChatGPT

Go back to ChatGPT to confirm Exa is no longer visible. You'll need to reconnect for ChatGPT to re-fetch the tool list.

![ChatGPT verify](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/15518882-8b19-44d3-9bba-245aeb62b4b1/ascreenshot_f98f59c51e6543e1be4f3960ba375fc9_text_export.jpeg)

Open the MCP server settings and select to add or reconnect a server.

![Reconnect to server](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/784d3174-77c0-42e6-a059-4c906db8f72a/ascreenshot_d77db951b83e4b15a00373222712f6b5_text_export.jpeg)

Enter the same LiteLLM MCP URL as before.

![Reconnect URL](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/17ef5fb0-b240-4556-8d20-753d359b7fcf/ascreenshot_583466ce9e8f40d1ba0af8b1e7d04413_text_export.jpeg)

Set the server label.

![Reconnect name](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/d7907637-c957-4a3c-ab4f-1600ca9a70a0/ascreenshot_e429eea43f3f4b3ca4d3ac5a77fbde2d_text_export.jpeg)

Enter your API key for authentication.

![Reconnect key](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/9cfff77a-37aa-4ca6-8032-0b46c50f37e3/ascreenshot_250664183399496b8f5c9f86f576fc0b_text_export.jpeg)

Click **"Connect"** to re-establish the connection.

![Click Connect](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/686f6307-b4ae-448b-ac6c-2c9d7b4f6b57/ascreenshot_3f499d0812af42ab89fed103cc21c249_text_export.jpeg)

This time, only DeepWiki's tools appear — Exa is gone. LiteLLM detected that ChatGPT is calling from a public IP and filtered out Exa since it's no longer marked as public. Internal users on your private network would still see both servers.

![Only DeepWiki tools visible](https://colony-recorder.s3.amazonaws.com/files/2026-02-07/667d79b6-75f9-4799-9315-0c176e7a5e34/ascreenshot_efa43050ac0b4445a09e542fa8f270ff_text_export.jpeg)

## Configuration Reference

### Per-Server Setting

<Tabs>
<TabItem value="ui" label="UI">

Toggle **"Available on Public Internet"** in the Permission Management section when creating or editing an MCP server.

</TabItem>
<TabItem value="config" label="config.yaml">

```yaml title="config.yaml" showLineNumbers
mcp_servers:
  deepwiki:
    url: https://mcp.deepwiki.com/mcp
    available_on_public_internet: true   # visible to external callers

  exa:
    url: https://exa.ai/mcp
    auth_type: api_key
    auth_value: os.environ/EXA_API_KEY
    available_on_public_internet: false  # internal only (default)
```

</TabItem>
<TabItem value="api" label="API">

```bash title="Create a public MCP server" showLineNumbers
curl -X POST <your-litellm-url>/v1/mcp/server \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "server_name": "DeepWiki",
    "url": "https://mcp.deepwiki.com/mcp",
    "transport": "http",
    "available_on_public_internet": true
  }'
```

```bash title="Update an existing server" showLineNumbers
curl -X PUT <your-litellm-url>/v1/mcp/server \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "server_id": "<server-id>",
    "available_on_public_internet": false
  }'
```

</TabItem>
</Tabs>

### Custom Private IP Ranges

By default, LiteLLM treats RFC 1918 private ranges as internal. You can customize this in the **Network Settings** tab under MCP Servers, or via config:

```yaml title="config.yaml" showLineNumbers
general_settings:
  mcp_internal_ip_ranges:
    - "10.0.0.0/8"
    - "172.16.0.0/12"
    - "192.168.0.0/16"
    - "100.64.0.0/10"    # Add your VPN/Tailscale range
```

When empty, the standard private ranges are used (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.0/8`).
