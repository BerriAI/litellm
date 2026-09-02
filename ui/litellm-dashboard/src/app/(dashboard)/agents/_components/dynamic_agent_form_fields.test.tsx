import { describe, expect, it } from "vitest";
import type { AgentCreateInfo } from "@/components/networking";
import { buildDynamicAgentData } from "./dynamic_agent_form_fields";

const agentcoreInfo: AgentCreateInfo = {
  agent_type: "bedrock_agentcore",
  agent_type_display_name: "Bedrock AgentCore",
  model_template: "bedrock/agentcore/{agent_runtime_arn}",
  credential_fields: [
    { key: "agent_runtime_arn", label: "Agent Runtime ARN" },
    { key: "aws_region_name", label: "AWS Region", include_in_litellm_params: true },
    { key: "aws_access_key_id", label: "AWS Access Key ID", include_in_litellm_params: true },
    { key: "aws_secret_access_key", label: "AWS Secret Access Key", include_in_litellm_params: true },
  ],
};

describe("buildDynamicAgentData", () => {
  it("omits masked credential values the proxy returned so an edit keeps the stored secret", () => {
    const prefilledFromMaskedResponse = {
      agent_name: "agentcore",
      agent_runtime_arn: "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/demo",
      aws_region_name: "us-west-2",
      aws_access_key_id: "AK****01",
      aws_secret_access_key: "SE****34",
    };
    const payload = buildDynamicAgentData(prefilledFromMaskedResponse, agentcoreInfo);

    expect(payload.litellm_params).toEqual({
      model: "bedrock/agentcore/arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/demo",
      agent_runtime_arn: "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/demo",
      aws_region_name: "us-west-2",
    });
  });

  it("sends a newly typed secret so the credential can be rotated", () => {
    const payload = buildDynamicAgentData(
      {
        agent_name: "agentcore",
        agent_runtime_arn: "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/demo",
        aws_secret_access_key: "ROTATED_SECRET_zyxwvu9876",
      },
      agentcoreInfo,
    );

    expect(payload.litellm_params.aws_secret_access_key).toBe("ROTATED_SECRET_zyxwvu9876");
  });
});
