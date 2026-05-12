<h1 align="center">
        🚅 LiteLLM
    </h1>
    <p align="center">
        <p align="center">LiteLLM AI Gateway
        </p>
        <p align="center">开源 AI 网关，支持 100+ 大语言模型。可自托管。企业级就绪。以 OpenAI 格式调用任意 LLM。</p>
        <p align="center">
        <a href="https://render.com/deploy?repo=https://github.com/BerriAI/litellm" target="_blank" rel="nofollow"><img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Render"></a>
        <a href="https://railway.com/deploy/RhvhdC?referralCode=7mRv9K&utm_medium=integration&utm_source=template&utm_campaign=generic">
          <img src="https://railway.com/button.svg" alt="Deploy on Railway">
        </a>
        </p>
    </p>
<h4 align="center"><a href="https://docs.litellm.ai/docs/simple_proxy" target="_blank">LiteLLM 代理服务器 (AI 网关)</a> | <a href="https://docs.litellm.ai/docs/enterprise#hosted-litellm-proxy" target="_blank"> 托管代理</a> | <a href="https://litellm.ai/enterprise"target="_blank">企业版</a> | <a href="https://litellm.ai/" target="_blank">官网</a></h4>
<h4 align="center">
    <a href="https://pypi.org/project/litellm/" target="_blank">
        <img src="https://img.shields.io/pypi/v/litellm.svg" alt="PyPI Version">
    </a>
    <a href="https://github.com/BerriAI/litellm" target="_blank">
        <img src="https://img.shields.io/github/stars/BerriAI/litellm.svg?style=social" alt="GitHub Stars">
    </a>
    <a href="https://www.ycombinator.com/companies/berriai">
        <img src="https://img.shields.io/badge/Y%20Combinator-W23-orange?style=flat-square" alt="Y Combinator W23">
    </a>
    <a href="https://wa.link/huol9n">
        <img src="https://img.shields.io/static/v1?label=Chat%20on&message=WhatsApp&color=success&logo=WhatsApp&style=flat-square" alt="Whatsapp">
    </a>
    <a href="https://discord.gg/wuPM9dRgDw">
        <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
    </a>
    <a href="https://www.litellm.ai/support">
        <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Slack&color=black&logo=Slack&style=flat-square" alt="Slack">
    </a>
    <a href="https://codspeed.io/BerriAI/litellm?utm_source=badge">
        <img src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" alt="CodSpeed"/>
    </a>
</h4>

<img width="2688" height="1600" alt="Group 7154 (1)" src="https://github.com/user-attachments/assets/c5ee0412-6fb5-4fb6-ab5b-bafae4209ca6" />

## LiteLLM 的用途

<details open>
<summary><b>大语言模型 (LLMs)</b> - 调用 100+ 大语言模型（Python SDK + AI 网关）</summary>

[**所有支持的端点**](https://docs.litellm.ai/docs/supported_endpoints) - `/chat/completions`, `/responses`, `/embeddings`, `/images`, `/audio`, `/batches`, `/rerank`, `/a2a`, `/messages` 等。

### Python SDK

```shell
pip install litellm
```

```python
from litellm import completion
import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# OpenAI
response = completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hello!"}])

# Anthropic
response = completion(model="anthropic/claude-sonnet-4-20250514", messages=[{"role": "user", "content": "Hello!"}])
```

### AI 网关（代理服务器）

[**快速入门 - 端到端教程**](https://docs.litellm.ai/docs/proxy/docker_quick_start) - 设置虚拟密钥，发起第一次请求

```shell
span
```

```python
import openai

client = openai.OpenAI(api_key="anything", base_url="http://0.0.0.0:4000")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

[**文档：LLM 提供商**](https://docs.litellm.ai/docs/providers)

</details>

<details>
<summary><b>智能体 (Agents)</b> - 调用 A2A 智能体（Python SDK + AI 网关）</summary>

[**支持的提供商**](https://docs.litellm.ai/docs/a2a#add-a2a-agents) - LangGraph, Vertex AI Agent Engine, Azure AI Foundry, Bedrock AgentCore, Pydantic AI

### Python SDK - A2A Protocol

```python
from litellm.a2a_protocol import A2AClient
from a2a.types import SendMessageRequest, MessageSendParams
from uuid import uuid4

client = A2AClient(base_url="http://localhost:10001")

request = SendMessageRequest(
    id=str(uuid4()),
    params=MessageSendParams(
        message={
            "role": "user",
            "parts": [{"kind": "text", "text": "Hello!"}],
            "messageId": uuid4().hex,
        }
    )
)
response = await client.send_message(request)
```

### AI 网关（代理服务器）

**第 1 步。** [将你的智能体添加到 AI 网关](https://docs.litellm.ai/docs/a2a#adding-your-agent)

**第 2 步。** 通过 A2A SDK 调用智能体

```python
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from uuid import uuid4
import httpx

base_url = "http://localhost:4000/a2a/my-agent"  # LiteLLM proxy + agent name
headers = {"Authorization": "Bearer sk-1234"}    # LiteLLM Virtual Key

async with httpx.AsyncClient(headers=headers) as httpx_client:
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
    agent_card = await resolver.get_agent_card()
    client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)

    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(
            message={
                "role": "user",
                "parts": [{"kind": "text", "text": "Hello!"}],
                "messageId": uuid4().hex,
            }
        )
    )
    response = await client.send_message(request)
```

[**文档：A2A 智能体网关**](https://docs.litellm.ai/docs/a2a)

</details>

<details>
<summary><b>MCP 工具</b> - 将 MCP 服务器连接到任意 LLM（Python SDK + AI 网关）</summary>

### Python SDK - MCP 桥接

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from litellm import experimental_mcp_client
import litellm

server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        # 以 OpenAI 格式加载 MCP 工具
        tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")

        # Use with any LiteLLM model
        response = await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's 3 + 5?"}],
            tools=tools
        )
```

### AI 网关 - MCP 网关

**第 1 步。** [将你的 MCP 服务器添加到 AI 网关](https://docs.litellm.ai/docs/mcp#adding-your-mcp)

**第 2 步。** Call MCP tools via `/chat/completions`

```bash
curl -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Summarize the latest open PR"}],
    "tools": [{
      "type": "mcp",
      "server_url": "litellm_proxy/mcp/github",
      "server_label": "github_mcp",
      "require_approval": "never"
    }]
  }'
```

### 在 Cursor IDE 中使用

```json
{
  "mcpServers": {
    "LiteLLM": {
      "url": "http://localhost:4000/mcp/",
      "headers": {
        "x-litellm-api-key": "Bearer sk-1234"
      }
    }
  }
}
```

[**文档：MCP 网关**](https://docs.litellm.ai/docs/mcp)

</details>

---

## 如何使用 LiteLLM

你可以通过代理服务器或 Python SDK 来使用 LiteLLM。两者都为你提供统一的接口来访问多个大语言模型（100+ LLMs）。选择最适合你需求的选项：

<table style={{width: '100%', tableLayout: 'fixed'}}>
<thead>
<tr>
<th style={{width: '14%'}}></th>
<th style={{width: '43%'}}><strong><a href="https://docs.litellm.ai/docs/simple_proxy">LiteLLM AI Gateway</a></strong></th>
<th style={{width: '43%'}}><strong><a href="https://docs.litellm.ai/docs/">LiteLLM Python SDK</a></strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style={{width: '14%'}}><strong>使用场景</strong></td>
<td style={{width: '43%'}}>访问多个 LLM 的中央服务（LLM 网关）</td>
<td style={{width: '43%'}}>在 Python 代码中直接使用 LiteLLM</td>
</tr>
<tr>
<td style={{width: '14%'}}><strong>使用者</strong></td>
<td style={{width: '43%'}}>Gen AI 赋能 / ML 平台团队</td>
<td style={{width: '43%'}}>构建 LLM 项目的开发者</td>
</tr>
<tr>
<td style={{width: '14%'}}><strong>核心功能</strong></td>
<td style={{width: '43%'}}>带身份验证和授权的集中式 API 网关，多租户成本跟踪和按项目/用户的费用管理，按项目定制（日志、护栏、缓存），用于安全访问控制的虚拟密钥，用于监控和管理的管理控制台 UI</td>
<td style={{width: '43%'}}>Direct Python library integration in your codebase, Router with retry/fallback logic across multiple deployments (e.g. Azure/OpenAI) - <a href="https://docs.litellm.ai/docs/routing">Router</a>, application-level load balancing and cost tracking, exception handling with OpenAI-compatible errors, observability callbacks (Lunary, MLflow, Langfuse, etc.)</td>
</tr>
</tbody>
</table>

LiteLLM 性能： **8ms P95 latency** 在 1k RPS 下 (查看[基准测试](https://docs.litellm.ai/docs/benchmarks))

[**跳转到 LiteLLM 代理 (LLM 网关) 文档**](https://docs.litellm.ai/docs/simple_proxy) <br>
[**跳转到支持的 LLM 提供商**](https://docs.litellm.ai/docs/providers)

**稳定版发布：** Use docker images with the `-stable` tag. These have undergone 12 hour load tests, before being published. [More information about the release cycle here](https://docs.litellm.ai/docs/proxy/release_cycle)

支持更多提供商。如果你缺少某个提供商或 LLM 平台，请提交[功能请求](https://github.com/BerriAI/litellm/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml&title=%5BFeature%5D%3A+).

## 开源使用者

<table>
  <tr>
    <td><img height="60" alt="Stripe" src="https://github.com/user-attachments/assets/f7296d4f-9fbd-460d-9d05-e4df31697c4b" /></td>
    <td><img height="60" alt="image" src="https://github.com/user-attachments/assets/436fca71-988b-40bb-b5fe-8450c80fdbd0" /></td>
    <td><img height="60" alt="Google ADK" src="https://github.com/user-attachments/assets/caf270a2-5aee-45c4-8222-41a2070c4f19" /></td>
    <td><img height="60" alt="Greptile" src="https://github.com/user-attachments/assets/0be4bd8a-7cfa-48d3-9090-f415fe948280" /></td>
    <td><img height="60" alt="OpenHands" src="https://github.com/user-attachments/assets/a6150c4c-149e-4cae-888b-8b92be6e003f" /></td>
    <td><h2>Netflix</h2></td>
    <td><img height="60" alt="OpenAI Agents SDK" src="https://github.com/user-attachments/assets/c02f7be0-8c2e-4d27-aea7-7c024bfaebc0" /></td>
  </tr>
</table>

## 支持的提供商 ([网站支持的模型](https://models.litellm.ai/) | [文档](https://docs.litellm.ai/docs/providers))


| Provider                                                                                                                         | `/chat/completions` | `/messages` | `/responses` | `/embeddings` | `/image/generations` | `/audio/transcriptions` | `/audio/speech` | `/moderations` | `/batches` | `/rerank` |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ----------- | ------------ | ------------- | -------------------- | ----------------------- | --------------- | -------------- | ---------- | --------- |
| [Abliteration (`abliteration`)](https://docs.litellm.ai/docs/providers/abliteration)                                             | ✅                  |             |              |               |                      |                         |                 |                |            |           |
| [AI/ML API (`aiml`)](https://docs.litellm.ai/docs/providers/aiml)                                                                | ✅                  | ✅          | ✅           | ✅            | ✅                   |                         |                 |                |            |           |
| [AI21 (`ai21`)](https://docs.litellm.ai/docs/providers/ai21)                                                                     | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [AI21 Chat (`ai21_chat`)](https://docs.litellm.ai/docs/providers/ai21)                                                           | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Aleph Alpha](https://docs.litellm.ai/docs/providers/aleph_alpha)                                                                | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Amazon Nova](https://docs.litellm.ai/docs/providers/amazon_nova)                                                                | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Anthropic (`anthropic`)](https://docs.litellm.ai/docs/providers/anthropic)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                | ✅         |           |
| [Anthropic Text (`anthropic_text`)](https://docs.litellm.ai/docs/providers/anthropic)                                            | ✅                  | ✅          | ✅           |               |                      |                         |                 |                | ✅         |           |
| [Anyscale](https://docs.litellm.ai/docs/providers/anyscale)                                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [AssemblyAI (`assemblyai`)](https://docs.litellm.ai/docs/pass_through/assembly_ai)                                               | ✅                  | ✅          | ✅           |               |                      | ✅                      |                 |                |            |           |
| [Auto Router (`auto_router`)](https://docs.litellm.ai/docs/proxy/auto_routing)                                                   | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [AWS - Bedrock (`bedrock`)](https://docs.litellm.ai/docs/providers/bedrock)                                                      | ✅                  | ✅          | ✅           | ✅            |                      |                         |                 |                |            | ✅        |
| [AWS - Sagemaker (`sagemaker`)](https://docs.litellm.ai/docs/providers/aws_sagemaker)                                            | ✅                  | ✅          | ✅           | ✅            |                      |                         |                 |                |            |           |
| [Azure (`azure`)](https://docs.litellm.ai/docs/providers/azure)                                                                  | ✅                  | ✅          | ✅           | ✅            | ✅                   | ✅                      | ✅              | ✅             | ✅         |           |
| [Azure AI (`azure_ai`)](https://docs.litellm.ai/docs/providers/azure_ai)                                                         | ✅                  | ✅          | ✅           | ✅            | ✅                   | ✅                      | ✅              | ✅             | ✅         |           |
| [Azure Text (`azure_text`)](https://docs.litellm.ai/docs/providers/azure)                                                        | ✅                  | ✅          | ✅           |               |                      | ✅                      | ✅              | ✅             | ✅         |           |
| [Baseten (`baseten`)](https://docs.litellm.ai/docs/providers/baseten)                                                            | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Bytez (`bytez`)](https://docs.litellm.ai/docs/providers/bytez)                                                                  | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Cerebras (`cerebras`)](https://docs.litellm.ai/docs/providers/cerebras)                                                         | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Clarifai (`clarifai`)](https://docs.litellm.ai/docs/providers/clarifai)                                                         | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Cloudflare AI Workers (`cloudflare`)](https://docs.litellm.ai/docs/providers/cloudflare_workers)                                | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Codestral (`codestral`)](https://docs.litellm.ai/docs/providers/codestral)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Cohere (`cohere`)](https://docs.litellm.ai/docs/providers/cohere)                                                               | ✅                  | ✅          | ✅           | ✅            |                      |                         |                 |                |            | ✅        |
| [Cohere Chat (`cohere_chat`)](https://docs.litellm.ai/docs/providers/cohere)                                                     | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [CometAPI (`cometapi`)](https://docs.litellm.ai/docs/providers/cometapi)                                                         | ✅                  | ✅          | ✅           | ✅            |                      |                         |                 |                |            |           |
| [CompactifAI (`compactifai`)](https://docs.litellm.ai/docs/providers/compactifai)                                                | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Custom (`custom`)](https://docs.litellm.ai/docs/providers/custom_llm_server)                                                    | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Custom OpenAI (`custom_openai`)](https://docs.litellm.ai/docs/providers/openai_compatible)                                      | ✅                  | ✅          | ✅           |               |                      | ✅                      | ✅              | ✅             | ✅         |           |
| [Dashscope (`dashscope`)](https://docs.litellm.ai/docs/providers/dashscope)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Databricks (`databricks`)](https://docs.litellm.ai/docs/providers/databricks)                                                   | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [DataRobot (`datarobot`)](https://docs.litellm.ai/docs/providers/datarobot)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Deepgram (`deepgram`)](https://docs.litellm.ai/docs/providers/deepgram)                                                         | ✅                  | ✅          | ✅           |               |                      | ✅                      |                 |                |            |           |
| [DeepInfra (`deepinfra`)](https://docs.litellm.ai/docs/providers/deepinfra)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Deepseek (`deepseek`)](https://docs.litellm.ai/docs/providers/deepseek)                                                         | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [ElevenLabs (`elevenlabs`)](https://docs.litellm.ai/docs/providers/elevenlabs)                                                   | ✅                  | ✅          | ✅           |               |                      | ✅                      | ✅              |                |            |           |
| [Empower (`empower`)](https://docs.litellm.ai/docs/providers/empower)                                                            | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Fal AI (`fal_ai`)](https://docs.litellm.ai/docs/providers/fal_ai)                                                               | ✅                  | ✅          | ✅           |               | ✅                   |                         |                 |                |            |           |
| [Featherless AI (`featherless_ai`)](https://docs.litellm.ai/docs/providers/featherless_ai)                                       | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Fireworks AI (`fireworks_ai`)](https://docs.litellm.ai/docs/providers/fireworks_ai)                                             | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [FriendliAI (`friendliai`)](https://docs.litellm.ai/docs/providers/friendliai)                                                   | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Galadriel (`galadriel`)](https://docs.litellm.ai/docs/providers/galadriel)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [GitHub Copilot (`github_copilot`)](https://docs.litellm.ai/docs/providers/github_copilot)                                       | ✅                  | ✅          | ✅           | ✅            |                      |                         |                 |                |            |           |
| [GitHub Models (`github`)](https://docs.litellm.ai/docs/providers/github)                                                        | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Google - PaLM](https://docs.litellm.ai/docs/providers/palm)                                                                     | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Google - Vertex AI (`vertex_ai`)](https://docs.litellm.ai/docs/providers/vertex)                                                | ✅                  | ✅          | ✅           | ✅            | ✅                   |                         |                 |                |            |           |
| [Google AI Studio - Gemini (`gemini`)](https://docs.litellm.ai/docs/providers/gemini)                                            | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [GradientAI (`gradient_ai`)](https://docs.litellm.ai/docs/providers/gradient_ai)                                                 | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Groq AI (`groq`)](https://docs.litellm.ai/docs/providers/groq)                                                                  | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Heroku (`heroku`)](https://docs.litellm.ai/docs/providers/heroku)                                                               | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Hosted VLLM (`hosted_vllm`)](https://docs.litellm.ai/docs/providers/vllm)                                                       | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Huggingface (`huggingface`)](https://docs.litellm.ai/docs/providers/huggingface)                                                | ✅                  | ✅          | ✅           | ✅            |                      |                         |                 |                |            | ✅        |
| [Hyperbolic (`hyperbolic`)](https://docs.litellm.ai/docs/providers/hyperbolic)                                                   | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [IBM - Watsonx.ai (`watsonx`)](https://docs.litellm.ai/docs/providers/watsonx)                                                   | ✅                  | ✅          | ✅           | ✅            |                      |                         |                 |                |            |           |
| [Infinity (`infinity`)](https://docs.litellm.ai/docs/providers/infinity)                                                         |                     |             |              | ✅            |                      |                         |                 |                |            |           |
| [Jina AI (`jina_ai`)](https://docs.litellm.ai/docs/providers/jina_ai)                                                            |                     |             |              | ✅            |                      |                         |                 |                |            |           |
| [Lambda AI (`lambda_ai`)](https://docs.litellm.ai/docs/providers/lambda_ai)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Lemonade (`lemonade`)](https://docs.litellm.ai/docs/providers/lemonade)                                                         | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [LiteLLM Proxy (`litellm_proxy`)](https://docs.litellm.ai/docs/providers/litellm_proxy)                                          | ✅                  | ✅          | ✅           | ✅            | ✅                   |                         |                 |                |            |           |
| [Llamafile (`llamafile`)](https://docs.litellm.ai/docs/providers/llamafile)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [LM Studio (`lm_studio`)](https://docs.litellm.ai/docs/providers/lm_studio)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Maritalk (`maritalk`)](https://docs.litellm.ai/docs/providers/maritalk)                                                         | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Meta - Llama API (`meta_llama`)](https://docs.litellm.ai/docs/providers/meta_llama)                                             | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Mistral AI API (`mistral`)](https://docs.litellm.ai/docs/providers/mistral)                                                     | ✅                  | ✅          | ✅           | ✅            |                      |                         |                 |                |            |           |
| [Moonshot (`moonshot`)](https://docs.litellm.ai/docs/providers/moonshot)                                                         | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Morph (`morph`)](https://docs.litellm.ai/docs/providers/morph)                                                                  | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Nebius AI Studio (`nebius`)](https://docs.litellm.ai/docs/providers/nebius)                                                     | ✅                  | ✅          | ✅           | ✅            |                      |                         |                 |                |            |           |
| [NLP Cloud (`nlp_cloud`)](https://docs.litellm.ai/docs/providers/nlp_cloud)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Novita AI (`novita`)](https://novita.ai/models/llm?utm_source=github_litellm&utm_medium=github_readme&utm_campaign=github_link) | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Nscale (`nscale`)](https://docs.litellm.ai/docs/providers/nscale)                                                               | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Nvidia NIM (`nvidia_nim`)](https://docs.litellm.ai/docs/providers/nvidia_nim)                                                   | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [OCI (`oci`)](https://docs.litellm.ai/docs/providers/oci)                                                                        | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Ollama (`ollama`)](https://docs.litellm.ai/docs/providers/ollama)                                                               | ✅                  | ✅          | ✅           | ✅            |                      |                         |                 |                |            |           |
| [Ollama Chat (`ollama_chat`)](https://docs.litellm.ai/docs/providers/ollama)                                                     | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Oobabooga (`oobabooga`)](https://docs.litellm.ai/docs/providers/openai_compatible)                                              | ✅                  | ✅          | ✅           |               |                      | ✅                      | ✅              | ✅             | ✅         |           |
| [OpenAI (`openai`)](https://docs.litellm.ai/docs/providers/openai)                                                               | ✅                  | ✅          | ✅           | ✅            | ✅                   | ✅                      | ✅              | ✅             | ✅         |           |
| [OpenAI-like (`openai_like`)](https://docs.litellm.ai/docs/providers/openai_compatible)                                          |                     |             |              | ✅            |                      |                         |                 |                |            |           |
| [OpenRouter (`openrouter`)](https://docs.litellm.ai/docs/providers/openrouter)                                                   | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [OVHCloud AI Endpoints (`ovhcloud`)](https://docs.litellm.ai/docs/providers/ovhcloud)                                            | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Perplexity AI (`perplexity`)](https://docs.litellm.ai/docs/providers/perplexity)                                                | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Petals (`petals`)](https://docs.litellm.ai/docs/providers/petals)                                                               | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Predibase (`predibase`)](https://docs.litellm.ai/docs/providers/predibase)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Recraft (`recraft`)](https://docs.litellm.ai/docs/providers/recraft)                                                            |                     |             |              |               | ✅                   |                         |                 |                |            |           |
| [Replicate (`replicate`)](https://docs.litellm.ai/docs/providers/replicate)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Sagemaker Chat (`sagemaker_chat`)](https://docs.litellm.ai/docs/providers/aws_sagemaker)                                        | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Sambanova (`sambanova`)](https://docs.litellm.ai/docs/providers/sambanova)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Snowflake (`snowflake`)](https://docs.litellm.ai/docs/providers/snowflake)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Text Completion Codestral (`text-completion-codestral`)](https://docs.litellm.ai/docs/providers/codestral)                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Text Completion OpenAI (`text-completion-openai`)](https://docs.litellm.ai/docs/providers/text_completion_openai)               | ✅                  | ✅          | ✅           |               |                      | ✅                      | ✅              | ✅             | ✅         |           |
| [Together AI (`together_ai`)](https://docs.litellm.ai/docs/providers/togetherai)                                                 | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Topaz (`topaz`)](https://docs.litellm.ai/docs/providers/topaz)                                                                  | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Triton (`triton`)](https://docs.litellm.ai/docs/providers/triton-inference-server)                                              | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [V0 (`v0`)](https://docs.litellm.ai/docs/providers/v0)                                                                           | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Vercel AI Gateway (`vercel_ai_gateway`)](https://docs.litellm.ai/docs/providers/vercel_ai_gateway)                              | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [VLLM (`vllm`)](https://docs.litellm.ai/docs/providers/vllm)                                                                     | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Volcengine (`volcengine`)](https://docs.litellm.ai/docs/providers/volcano)                                                      | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Voyage AI (`voyage`)](https://docs.litellm.ai/docs/providers/voyage)                                                            |                     |             |              | ✅            |                      |                         |                 |                |            |           |
| [WandB Inference (`wandb`)](https://docs.litellm.ai/docs/providers/wandb_inference)                                              | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Watsonx Text (`watsonx_text`)](https://docs.litellm.ai/docs/providers/watsonx)                                                  | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [xAI (`xai`)](https://docs.litellm.ai/docs/providers/xai)                                                                        | ✅                  | ✅          | ✅           |               |                      |                         |                 |                |            |           |
| [Xinference (`xinference`)](https://docs.litellm.ai/docs/providers/xinference)                                                   |                     |             |              | ✅            |                      |                         |                 |                |            |           |

[**Read the Docs**](https://docs.litellm.ai/docs/)

## 以开发者模式运行

### 服务

1. 在根目录设置 .env 文件
2. 运行依赖服务 `docker-compose up db prometheus`

### 后端

1. (在根目录)创建虚拟环境 `python -m venv .venv`
2. 激活虚拟环境 `source .venv/bin/activate`
3. 安装依赖 `pip install -e ".[all]"`
4. `pip install prisma`
5. `prisma generate`
6. 启动代理后端 `python litellm/proxy/proxy_cli.py`

### 前端

1. 进入 `ui/litellm-dashboard` 目录
2. 安装依赖 `npm install`
3. Run `npm run dev` 启动控制台

# 验证 Docker 镜像签名

所有发布到 GHCR 的 LiteLLM Docker 镜像都使用 [cosign](https://docs.sigstore.dev/cosign/overview/)。每次发布都使用在 [commit `0112e53`](https://github.com/BerriAI/litellm/commit/0112e53046018d726492c814b3644b7d376029d0) 中引入的同一密钥进行签名。

**使用固定的提交哈希验证（推荐）：**

提交哈希在密码学上是不可变的，因此这是确保你使用原始签名密钥的最安全方式：

```bash
cosign verify \
  --key https://raw.githubusercontent.com/BerriAI/litellm/0112e53046018d726492c814b3644b7d376029d0/cosign.pub \
  ghcr.io/berriai/litellm:<release-tag>
```

**使用发布标签验证（便捷方式）：**

标签在此仓库中受保护，并且解析到相同的密钥。此选项更易读，但依赖于标签保护规则：

```bash
cosign verify \
  --key https://raw.githubusercontent.com/BerriAI/litellm/<release-tag>/cosign.pub \
  ghcr.io/berriai/litellm:<release-tag>
```

将 `<release-tag>` 替换为你正在部署的版本（例如 `v1.83.0-stable`）。

# 企业版

面向需要更高安全性、用户管理和专业支持的公司

[获取企业版许可证](https://litellm.ai/enterprise)
[与创始人交流](https://enterprise.litellm.ai/demo)

企业版涵盖以下内容：

- ✅ **Features under the [LiteLLM Commercial License](https://docs.litellm.ai/docs/proxy/enterprise):**
- ✅ **功能优先开发**
- ✅ **自定义集成**
- ✅ **专业支持 - 专属 Discord + Slack 频道**
- ✅ **自定义 SLA**
- ✅ **通过单点登录 (SSO) 实现安全访问**

# 贡献

我们欢迎对 LiteLLM 的贡献！无论是修复 Bug、添加功能还是改进文档，我们都感谢你的帮助。

## 贡献者快速入门

此操作需要安装 poetry。

```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm
make install-dev    # 安装开发依赖
make format         # 格式化代码
make lint           # 运行所有代码检查
make test-unit      # 运行单元测试
make format-check   # 仅检查格式
```

详细的贡献指南请参见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 代码质量 / 代码检查

LiteLLM 遵循 [Google Python 风格指南](https://google.github.io/styleguide/pyguide.html).

我们的自动化检查包括：

- **Black** - 代码格式化
- **Ruff** - 代码检查和质量
- **MyPy** - 类型检查
- 循环导入检测
- 导入安全性检查

在 PR 合并之前，所有这些检查都必须通过。

# 支持 / 与创始人交流

- [预约演示 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [社区 Discord 💭](https://discord.gg/wuPM9dRgDw)
- [社区 Slack 💭](https://www.litellm.ai/support)
- 我们的邮箱 ✉️ ishaan@berri.ai / krrish@berri.ai

# 为什么构建这个项目

- **追求简洁**：我们的代码在管理和翻译 Azure、OpenAI 和 Cohere 之间的调用时变得极其复杂。

# 贡献者

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->

<!-- prettier-ignore-start -->

<!-- markdownlint-disable -->

<!-- markdownlint-restore -->

<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<a href="https://github.com/BerriAI/litellm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=BerriAI/litellm" />
</a>
