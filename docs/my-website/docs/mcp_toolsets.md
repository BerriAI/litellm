import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# MCP Toolsets

A **Toolset** is a named collection of specific tools drawn from one or more MCP servers. Instead of giving an agent access to every tool on every server, you pick exactly which tools it needs — from whichever servers they live on — and bundle them under a single name.

## How it works

```
                    ┌─────────────────────────────────┐
                    │         MCP Toolset              │
                    │      "devtooling-prod"           │
                    └────────────┬────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
     ┌────────▼────────┐                  ┌────────▼────────┐
     │  CircleCI MCP   │                  │  DeepWiki MCP   │
     │  (10+ tools)    │                  │  (3 tools)      │
     └────────┬────────┘                  └────────┬────────┘
              │                                    │
    ┌─────────┴──────────┐              ┌──────────┴──────────┐
    │ ✓ get_build_logs   │              │ ✓ read_wiki_structure│
    │ ✓ find_flaky_tests │              │ ✓ read_wiki_contents │
    │ ✓ get_pipeline_    │              │ ✗ ask_question       │
    │   status           │              └─────────────────────┘
    │ ✓ run_pipeline     │
    │ ✗ list_followed_   │
    │   projects         │
    └────────────────────┘

        Agent sees exactly 6 tools, nothing more.
```

Instead of 13+ tools across two servers, the agent gets 6 — the ones it actually needs.

**Why this matters:**
- Smaller tool lists → fewer tokens, faster responses, less hallucination
- Combine tools from GitHub + Linear + CircleCI into one named grant
- Assign to keys and teams the same way you assign MCP servers today

---

## Create a toolset

### 1. Go to the MCP page

Navigate to **MCP** in the left sidebar.

![Navigate to MCP](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/1a96c713-6a37-4f96-92f1-07bd58c1973c/ascreenshot_23515f386ccc4597b0633987667fe01f_text_export.jpeg)

### 2. Open the Toolsets tab

Click the **Toolsets** tab on the MCP page.

![Click Toolsets tab](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/65b6986b-595a-4b28-8fdc-a7b36bc76e59/ascreenshot_ca70c18fe7ec415486f96a6b405bf550_text_export.jpeg)

### 3. Click "New Toolset"

![New Toolset button](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/798c55c4-5d6b-4815-a642-70ac9f34f102/ascreenshot_3f144f54a1a944e28454239c837b4e6d_text_export.jpeg)

### 4. Enter a name

Type a name for the toolset. Pick something descriptive — this is what agents will reference.

![Enter toolset name](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/62b412e0-d38f-44c3-99e4-3693f1512f6a/ascreenshot_b678c7c988a04f8b887b0f54c4dd95a7_text_export.jpeg)

![Toolset name field](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/ba5ebc95-cab7-470b-a7c9-21f12b9b01a3/ascreenshot_a602e982a2a44890a83dca64d61c38eb_text_export.jpeg)

### 5. Add the first tool

Select an MCP server from the dropdown, then choose the tool you want to include from that server.

![Select MCP server](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/2aa5bcba-6414-42e3-9813-efb0a9078e32/ascreenshot_58fbff35ba654210a1b4dc5452aa6bd9_text_export.jpeg)

![Choose server from dropdown](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/4fd9cffb-d3ba-461a-8679-89f278bf67ad/ascreenshot_b61e9e85a51b494a8d09fe61198d63e1_text_export.jpeg)

![Select tool from server](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/60718e72-2062-494b-9a23-456992c88cbd/ascreenshot_7a1f8eeab30a4a05ba39c450e5458b78_text_export.jpeg)

### 6. Add tools from a second server

Click **Add Tool**, pick a different MCP server, and select another tool. Repeat for as many tools as you need — they can come from any number of servers.

![Add tool from second server](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/f34e0600-cc74-4b18-8794-88d45f326144/ascreenshot_98834b14ab9343e39fb503e458d72b7c_text_export.jpeg)

![Select second server](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/75150368-2202-4da1-99f1-6f0620e9b133/ascreenshot_f94d0bc08ea147348a9cf021cce7d854_text_export.jpeg)

![Select tool from second server](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/ed2cdf6e-025d-4d50-8b12-ed68745d5c51/ascreenshot_0c1c7f76524b46c5a056fda5e6956e2b_text_export.jpeg)

### 7. Create the toolset

Click **Create Toolset** to save.

![Create Toolset](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/021ca7b3-2d9a-49a0-8758-dae3dc3bcb4d/ascreenshot_14c6434e71114a6091e359a996f20e12_text_export.jpeg)

---

## Use a toolset in the Playground

Once created, your toolset appears alongside MCP servers in the **MCP Servers** dropdown in the Playground — it's selectable the same way.

### 1. Go to the Playground

![Navigate to Playground](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/f9d4aa4c-d98e-4767-b98e-aad2890e97ca/ascreenshot_d84239c441bb4e828f229d0c9e079e3f_text_export.jpeg)

![Click Playground](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/d8a07563-97fe-453a-b974-88da46c87294/ascreenshot_ea494300a536400abb2ea6bf3bdfd5ab_text_export.jpeg)

### 2. Select your toolset from MCP Servers

In the left panel under **MCP Servers**, open the dropdown and pick your toolset. The model will only see the tools you included in it.

![Select MCP servers dropdown](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/ee8cb38c-c4ff-4b4b-844c-22f2e40832ae/ascreenshot_e300fb39cea0434fb5e3986e912a2b8d_text_export.jpeg)

![Open MCP server picker](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/8672070c-5d07-4f63-878c-6fc7dcbc9b65/ascreenshot_326ddd0868224c99a6fa5dab2d144f1f_text_export.jpeg)

![Select toolset](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/955826ad-2bbb-403e-ab26-c1ac03ec2675/ascreenshot_13f837ad53574535986ca7ca5998d34a_text_export.jpeg)

![Toolset selected and active](https://colony-recorder.s3.amazonaws.com/files/2026-03-22/9a59c3b9-1563-4731-838f-1c35d636ddc9/ascreenshot_c05d8fa5f37a4b3093fc46e26f293b4d_text_export.jpeg)

The model now has access to exactly the tools in your toolset and nothing else.

---

## Use a toolset via API

Pass the toolset's route as the `server_url` in your tools list. LiteLLM resolves it server-side — no public URL needed.

<Tabs>
<TabItem value="responses" label="Responses API">

```python
import openai

client = openai.OpenAI(
    api_key="your-litellm-key",
    base_url="http://your-proxy/v1",
)

response = client.responses.create(
    model="gpt-4o",
    input="What CI/CD tools do you have?",
    tools=[
        {
            "type": "mcp",
            "server_label": "devtooling-prod",
            "server_url": "litellm_proxy/mcp/devtooling-prod",
            "require_approval": "never",
        }
    ],
)
print(response.output_text)
```

</TabItem>
<TabItem value="chat" label="Chat Completions API">

```python
import openai

client = openai.OpenAI(
    api_key="your-litellm-key",
    base_url="http://your-proxy/v1",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What CI/CD tools do you have?"}],
    tools=[
        {
            "type": "mcp",
            "server_label": "devtooling-prod",
            "server_url": "litellm_proxy/mcp/devtooling-prod",
            "require_approval": "never",
        }
    ],
)
print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="rest" label="REST">

```bash
curl http://your-proxy/v1/responses \
  -H "Authorization: Bearer your-litellm-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "input": "What CI/CD tools do you have?",
    "tools": [
      {
        "type": "mcp",
        "server_label": "devtooling-prod",
        "server_url": "litellm_proxy/mcp/devtooling-prod",
        "require_approval": "never"
      }
    ]
  }'
```

</TabItem>
</Tabs>

---

## Manage toolsets via API

```bash
# List all toolsets
curl http://your-proxy/v1/mcp/toolset \
  -H "Authorization: Bearer your-litellm-key"

# Create a toolset
curl -X POST http://your-proxy/v1/mcp/toolset \
  -H "Authorization: Bearer your-litellm-key" \
  -H "Content-Type: application/json" \
  -d '{
    "toolset_name": "devtooling-prod",
    "description": "CircleCI + DeepWiki tools for the dev team",
    "tools": [
      {"server_id": "<circleci-server-id>", "tool_name": "get_build_failure_logs"},
      {"server_id": "<circleci-server-id>", "tool_name": "run_pipeline"},
      {"server_id": "<deepwiki-server-id>", "tool_name": "read_wiki_structure"}
    ]
  }'

# Delete a toolset
curl -X DELETE http://your-proxy/v1/mcp/toolset/<toolset_id> \
  -H "Authorization: Bearer your-litellm-key"
```
