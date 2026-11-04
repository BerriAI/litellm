import { modelUsageExample } from "@/components/AIHub/modelUsageExample";
import type { Plugin } from "@/components/claude_code_plugins/types";
import { formatInstallCommand } from "@/components/claude_code_plugins/helpers";
import { generateCodeSnippet } from "@/components/chat_ui/CodeSnippets";
import { getEndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import { getGlobalLitellmHeaderName } from "@/components/networking";
import type { AgentCard, MCPServerData, ModelGroupInfo } from "@/components/PublicModelHubTableColumns";
import { isDecisionMode } from "@/lib/decisionModels";

import { gatewayUrl, wildcardExample } from "./hubFormat";

const GREETING = "Hello, how are you?";

export const modelSnippet = (model: ModelGroupInfo): string => {
  const name = wildcardExample(model.model_group);
  if (isDecisionMode(model.mode)) {
    return modelUsageExample(model.mode, gatewayUrl(), name, getGlobalLitellmHeaderName());
  }
  const params: Parameters<typeof generateCodeSnippet>[0] = {
    apiKeySource: "custom",
    accessToken: null,
    apiKey: "your_api_key",
    inputMessage: GREETING,
    chatHistory: [{ role: "user", content: GREETING, isImage: false }],
    selectedTags: [],
    selectedVectorStores: [],
    selectedGuardrails: [],
    selectedPolicies: [],
    selectedMCPServers: [],
    endpointType: getEndpointType(model.mode || "chat"),
    selectedModel: name,
    selectedSdk: "openai",
    proxySettings: { PROXY_BASE_URL: gatewayUrl() },
  };
  return generateCodeSnippet(params);
};

export const curlSnippet = (model: ModelGroupInfo): string | null => {
  if (model.mode && model.mode !== "chat") return null;
  return `curl ${gatewayUrl()}/v1/chat/completions \\
  -H "Authorization: Bearer $LITELLM_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${wildcardExample(model.model_group)}",
    "messages": [{"role": "user", "content": "Hello"}]
  }'`;
};

export const agentSnippets = (agent: AgentCard): { title: string; code: string }[] => [
  {
    title: "Retrieve the agent card",
    code: `from a2a.client import A2ACardResolver

resolver = A2ACardResolver(httpx_client=httpx_client, base_url="${agent.url}")
card = await resolver.get_agent_card()`,
  },
  {
    title: "Call the agent",
    code: `from a2a.client import A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

client = A2AClient(httpx_client=httpx_client, agent_card=card)
request = SendMessageRequest(
    id=str(uuid4()),
    params=MessageSendParams(message={
        "role": "user",
        "parts": [{"kind": "text", "text": "how much is 10 USD in INR?"}],
        "messageId": uuid4().hex,
    }),
)
response = await client.send_message(request)`,
  },
];

export const mcpEndpoint = (server: MCPServerData): string => `${gatewayUrl()}/${server.name}/mcp`;

export const mcpSnippet = (server: MCPServerData): string => `from fastmcp import Client

client = Client({
    "mcpServers": {
        "${server.name}": {
            "url": "${mcpEndpoint(server)}",
            "headers": {"x-litellm-api-key": "Bearer your_api_key"},
        }
    }
})

async with client:
    tools = await client.list_tools()`;

export const marketplaceCommand = (): string => `/plugin marketplace add ${gatewayUrl()}/claude-code/marketplace.json`;

export const skillInstallCommand = (skill: Plugin): string => formatInstallCommand(skill);
