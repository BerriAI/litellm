<h1 align="center">
        🚅 LiteLLM
    </h1>
    <p align="center">
        <p align="center">LiteLLM AI Gateway (AI 大模型统一网关)
        </p>
        <p align="center">面向 100+ 大语言模型的开源 AI 网关。支持私有化部署、企业级就绪，全面统一采用 OpenAI 标准格式调用任意 LLM。</p>
        <p align="center">
        <a href="https://render.com/deploy?repo=https://github.com/BerriAI/litellm" target="_blank" rel="nofollow"><img src="https://render.com/images/deploy-to-render-button.svg" alt="部署至 Render" height="40"></a>
        <a href="https://railway.com/deploy/RhvhdC?referralCode=7mRv9K&utm_medium=integration&utm_source=template&utm_campaign=generic"><img src="https://railway.com/button.svg" alt="部署至 Railway" height="40"></a>
        <a href="https://console.aws.amazon.com/cloudshell/home" target="_blank" rel="nofollow"><img src="./.github/deploy-on-aws.png" alt="部署至 AWS" height="40"></a>
        <a href="https://ssh.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FBerriAI%2Flitellm&cloudshell_workspace=terraform%2Flitellm%2Fgcp%2Fexamples%2Fdefault&cloudshell_tutorial=TUTORIAL.md&cloudshell_image=gcr.io/ds-artifacts-cloudshell/deploystack_custom_image&shellonly=true" target="_blank" rel="nofollow"><img src="./.github/deploy-on-gcp.png" alt="部署至 GCP" height="40"></a>
        </p>
    </p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<h4 align="center"><a href="https://docs.litellm.ai/docs/simple_proxy" target="_blank">LiteLLM 代理服务 (AI 网关)</a> | <a href="https://docs.litellm.ai/docs/enterprise#hosted-litellm-proxy" target="_blank">托管版 Proxy</a> | <a href="https://litellm.ai/enterprise" target="_blank">企业级支持</a> | <a href="https://www.litellm.ai/ai-gateway" target="_blank">官方网站</a></h4>
<h4 align="center">
    <a href="https://pypi.org/project/litellm/" target="_blank">
        <img src="https://img.shields.io/pypi/v/litellm.svg" alt="PyPI 版本">
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

<img alt="LiteLLM AI Gateway 架构" src="https://github.com/user-attachments/assets/c5ee0412-6fb5-4fb6-ab5b-bafae4209ca6" />

---

## 什么是 LiteLLM？

LiteLLM 是一个开源的 **AI 网关 (AI Gateway)**，它为调用 100 多个 LLM 供应商（OpenAI、Anthropic Claude、Google Gemini、AWS Bedrock、Azure OpenAI、DeepSeek 等）提供单一且统一的 OpenAI 规范接口。

既可作为 **Python SDK** 直接嵌入您的应用程序代码中，也可部署为集中式的 **AI 网关代理服务 (Proxy Server)** 供团队与企业全员使用。

[**直达 LiteLLM Proxy (网关服务) 文档**](https://docs.litellm.ai/docs/simple_proxy) <br>
[**直达支持的模型供应商列表**](https://docs.litellm.ai/docs/providers)

---

## 为什么选择 LiteLLM？

跨供应商管理不同大语言模型调用极易陷入繁杂混乱 —— 每个模型都有各自独立的 SDK、认证鉴权方式、请求入参规范以及差异巨大的异常类型。LiteLLM 彻底消除了这些技术摩擦：

- **统一 API 规范** —— 单一接口聚合 100+ LLM，无需在项目里引入与维护数十个各异的 SDK。
- **无缝平替 OpenAI 协议** —— 无需重构业务代码，仅需修改 Base URL 即可自由切换底座模型。
- **生产就绪的企业级网关** —— 开箱即用的虚拟 API Key 管理、多租户成本与花费追踪、安全防护 Guardrails、动态负载均衡与可视化管理后台 Dashboard。
- **极致低延迟性能** —— 在 1000 RPS 并发下 P95 转发延迟仅 **8ms**（[基准评测详情](https://docs.litellm.ai/docs/benchmarks)）。

### 知名开源采纳者 (OSS Adopters)

<table>
  <tr>
    <td><img height="60" alt="Stripe" src="https://github.com/user-attachments/assets/f7296d4f-9fbd-460d-9d05-e4df31697c4b" /></td>
    <td><img height="60" alt="image" src="https://github.com/user-attachments/assets/436fca71-988b-40bb-b5fe-8450c80fdbd0" /></td>
    <td><img height="60" alt="Google ADK" src="https://github.com/user-attachments/assets/caf270a2-5aee-45c4-8222-41a2070c4f19" /></td>
    <td><img height="60" alt="Greptile" src="https://github.com/user-attachments/assets/3db0ae72-0843-4005-a56d-bba1dde2193d" /></td>
    <td><img height="60" alt="OpenHands" src="https://github.com/user-attachments/assets/a6150c4c-149e-4cae-888b-8b92be6e003f" /></td>
    <td><h2>Netflix</h2></td>
    <td><img height="60" alt="OpenAI Agents SDK" src="https://github.com/user-attachments/assets/c02f7be0-8c2e-4d27-aea7-7c024bfaebc0" /></td>
  </tr>
</table>

---

## 核心特性

<details open>
<summary><b>LLM 统一调用</b> - 支持 100+ 模型 (Python SDK + AI 网关)</summary>

[**所有支持的 Endpoint 接口**](https://docs.litellm.ai/docs/supported_endpoints) - `/chat/completions`、`/responses`、`/embeddings`、`/images`、`/audio`、`/batches`、`/rerank`、`/a2a`、`/messages` 等。

### Python SDK 使用方式

```shell
uv add litellm
```

```python
from litellm import completion
import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# 调用 OpenAI 模型
response = completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hello!"}])

# 调用 Anthropic Claude 模型
response = completion(model="anthropic/claude-sonnet-4-20250514", messages=[{"role": "user", "content": "Hello!"}])
```

### AI Gateway (网关代理服务)

[**新手入门 - 端到端教程**](https://docs.litellm.ai/docs/proxy/docker_quick_start) - 配置虚拟密钥并发送第一个请求

```shell
uv tool install 'litellm[proxy]'
litellm --model gpt-4o
```

```python
import openai

client = openai.OpenAI(api_key="anything", base_url="http://0.0.0.0:4000")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

[**文档：支持的模型供应商**](https://docs.litellm.ai/docs/providers)

</details>

<details>
<summary><b>智能体集成</b> - 调度 A2A 协议智能体 (Python SDK + AI 网关)</summary>

[**支持的 Agent 引擎**](https://docs.litellm.ai/docs/a2a#add-a2a-agents) - LangGraph、Vertex AI Agent Engine、Azure AI Foundry、Bedrock AgentCore、Pydantic AI。

### Python SDK - A2A 协议

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

### AI Gateway (网关代理服务)

**步骤 1.** [将您的智能体注册到 AI 网关](https://docs.litellm.ai/docs/a2a#adding-your-agent)（按 Agent 指定 `protocolVersion` 为 `1.0` 或 `0.3`）

**步骤 2.** 通过 A2A SDK 调用该智能体（需要 `a2a-sdk>=1.1.0`）：

```python
import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, SendMessageRequest
from a2a.utils.constants import TransportProtocol
from uuid import uuid4

base_url = "http://localhost:4000/a2a/my-agent"  # LiteLLM 代理地址 + Agent 名称
headers = {"Authorization": "Bearer sk-1234"}    # LiteLLM 生成的虚拟密钥

async with httpx.AsyncClient(headers=headers, timeout=60.0) as http_client:
    resolver = A2ACardResolver(httpx_client=http_client, base_url=base_url)
    agent_card = await resolver.get_agent_card()
    config = ClientConfig(
        httpx_client=http_client,
        streaming=False,
        supported_protocol_bindings=[TransportProtocol.JSONRPC, TransportProtocol.HTTP_JSON],
    )
    client = ClientFactory(config).create(agent_card)

    request = SendMessageRequest(
        message=Message(
            message_id=uuid4().hex,
            role=Role.ROLE_USER,
            parts=[Part(text="Hello!")],
        )
    )
    async for event in client.send_message(request):
        populated = event.ListFields()
        if populated and populated[0][0].name in ("message", "msg"):
            print("".join(getattr(p, "text", "") or "" for p in populated[0][1].parts))
```

[**文档：A2A Agent 网关**](https://docs.litellm.ai/docs/a2a)

</details>

<details>
<summary><b>MCP 工具生态</b> - 将 MCP 服务器桥接至任意 LLM (Python SDK + AI 网关)</summary>

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

        # 将 MCP 工具加载为 OpenAI 标准格式
        tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")

        # 搭配任意 LiteLLM 支持的模型使用
        response = await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's 3 + 5?"}],
            tools=tools
        )
```

### AI Gateway - MCP 网关模式

**步骤 1.** [将您的 MCP 服务器添加到 AI 网关](https://docs.litellm.ai/docs/mcp#adding-your-mcp)

**步骤 2.** 通过 `/chat/completions` 调用 MCP 工具：

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

### 在 Cursor IDE 中配置使用

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

[**文档：MCP 网关集成**](https://docs.litellm.ai/docs/mcp)

</details>

---

## 快速上手与架构选择

您可以根据实际架构需求选择使用 **Proxy Server 网关** 或 **Python SDK**：

<table>
<thead>
<tr>
<th>对比维度</th>
<th><strong><a href="https://docs.litellm.ai/docs/simple_proxy">LiteLLM AI Gateway (代理服务)</a></strong></th>
<th><strong><a href="https://docs.litellm.ai/docs/">LiteLLM Python SDK</a></strong></th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>使用场景</strong></td>
<td>集中式服务（LLM 网关），为整个团队或企业统一调度多种 LLM</td>
<td>直接在 Python 项目代码中作为基础库集成</td>
</tr>
<tr>
<td><strong>适用人群</strong></td>
<td>企业 Gen AI 赋能团队 / 基础架构平台运维 / ML 平台工程师</td>
<td>从事具体大模型应用与智能体开发的工程师</td>
</tr>
<tr>
<td><strong>核心功能</strong></td>
<td>统一鉴权与权限管控、多租户/项目/用户级用量与成本追踪、项目级自定义（审计日志、安全护栏 Guardrails、KV 缓存）、虚拟 API Key 安全分发、管理后台 Web UI</td>
<td>直接作为 Python 依赖库调用、跨供应商（如 Azure/OpenAI）自动重试与故障转移路由 Router、应用级负载均衡、统一捕获 OpenAI 格式异常、可观测性回调集成（Lunary、MLflow、Langfuse 等）</td>
</tr>
</tbody>
</table>

**稳定版本建议**：在生产部署时推荐选用带有 `-stable` 标签的 Docker 镜像，这些镜像在发布前均通过了 12 小时高压力负载测试（[了解发布周期规范](https://docs.litellm.ai/docs/proxy/release_cycle)）。

### 使用 Terraform 在 AWS 或 GCP 上一键生产部署

使用官方发布的 Terraform 模块，以标准组件化架构（网关、后端、UI 独立解耦，托管 Postgres + Redis）一键部署 LiteLLM：

- **AWS 生产环境**：ECS Fargate + Aurora Serverless + ElastiCache + ALB（模块位于 [`terraform/litellm/aws/`](./terraform/litellm/aws/)）。
- **GCP 生产环境**：Cloud Run + Cloud SQL + Memorystore + HTTPS 负载均衡（模块位于 [`terraform/litellm/gcp/`](./terraform/litellm/gcp/)）。

### 本地开发者调试模式

1. 在根目录配置 `.env` 环境变量文件。
2. 启动依赖服务：`docker-compose up db prometheus`。
3. 启动后端代理服务：`make bootstrap && uv run python litellm/proxy/proxy_cli.py`。
4. 启动前端控制台：进入 `ui/litellm-dashboard` 并运行 `npm run dev`。

### 镜像签名与安全验证 (Cosign)

发布到 GHCR 的所有 LiteLLM Docker 镜像均经过 [cosign](https://docs.sigstore.dev/cosign/overview/) 密码学数字签名。

```bash
cosign verify \
  --key https://raw.githubusercontent.com/BerriAI/litellm/0112e53046018d726492c814b3644b7d376029d0/cosign.pub \
  ghcr.io/berriai/litellm:<release-tag>
```

---

## 企业级服务 (Enterprise)

针对需要更严格数据安全、精细化用户管理与企业级 SLA 支持的机构：
[获取企业授权许可证](https://litellm.ai/enterprise) | [预约创始人技术交流](https://enterprise.litellm.ai/demo)

- ✅ **[LiteLLM 商业许可证](https://docs.litellm.ai/docs/proxy/enterprise)全部高级特性**
- ✅ **优先功能研发支持**
- ✅ **定制化内部系统集成**
- ✅ **专属 Slack / Discord 实时技术支持通道**
- ✅ **企业级 SLA 服务保障**
- ✅ **基于 SSO（单点登录）的安全访问控制**

## 参与贡献 (Contributing)

热烈欢迎社区开发者参与贡献！无论修复 Bug、新增模型支持还是完善文档，我们都由衷感谢您的付出。

```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm
make install-dev    # 安装开发依赖
make format         # 格式化代码 (Black/Ruff)
make lint           # 执行全部 Lint 校验
make test-unit      # 运行单元测试
```

代码质量遵循 [Google Python 风格指南](https://google.github.io/styleguide/pyguide.html)，在提交 PR 前请确保通过自动化代码检查。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年8月31日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
