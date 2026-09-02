import { describe, it, expect } from "vitest";
import { parseDynamicAgentForForm } from "./agent_type_utils";
import type { AgentCreateInfo } from "@/components/networking";
import type { Agent } from "@/components/agents/types";

const AGENTCORE_ARN = "arn:aws:bedrock-agentcore:eu-central-1:123456789012:runtime/hosted_agent_4vm3i-BaTdfOELAs";

const agentcoreInfo: AgentCreateInfo = {
  agent_type: "bedrock_agentcore",
  agent_type_display_name: "Bedrock AgentCore",
  model_template: "bedrock/agentcore/{agent_runtime_arn}",
  credential_fields: [
    { key: "agent_runtime_arn", label: "Agent Runtime ARN", required: true, include_in_litellm_params: false },
  ],
};

const langflowInfo: AgentCreateInfo = {
  agent_type: "langflow",
  agent_type_display_name: "Langflow",
  model_template: "langflow/{flow_id}",
  credential_fields: [{ key: "flow_id", label: "Flow ID", required: true, include_in_litellm_params: false }],
};

const agentWithModel = (custom_llm_provider: string, model: string): Agent =>
  ({
    agent_id: "agent-3",
    agent_name: "dyn-agent",
    litellm_params: { custom_llm_provider, model },
  }) as unknown as Agent;

describe("parseDynamicAgentForForm", () => {
  it("keeps every slash-separated segment of the AgentCore runtime ARN", () => {
    const agent = agentWithModel("bedrock", `bedrock/agentcore/${AGENTCORE_ARN}`);

    expect(parseDynamicAgentForForm(agent, agentcoreInfo).agent_runtime_arn).toBe(AGENTCORE_ARN);
  });

  it("still reads the placeholder when the stored prefix differs from the template", () => {
    const agent = agentWithModel("langflow", "custom-prefix/flow_1");

    expect(parseDynamicAgentForForm(agent, langflowInfo).flow_id).toBe("flow_1");
  });

  it("leaves the field unset when the model has no placeholder segment", () => {
    const agent = agentWithModel("langflow", "langflow");

    expect(parseDynamicAgentForForm(agent, langflowInfo)).not.toHaveProperty("flow_id");
  });
});
