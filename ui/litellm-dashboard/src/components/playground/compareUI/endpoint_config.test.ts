import { describe, expect, it } from "vitest";
import {
  EndpointId,
  ENDPOINT_CONFIGS,
  getAvailableEndpoints,
  getEndpointConfig,
  isAgentEndpoint,
  isModelEndpoint,
  modelOptionsToSelectorOptions,
  agentOptionsToSelectorOptions,
  getSelectionFieldName,
  getComparisonSelection,
  hasValidSelection,
} from "./endpoint_config";
import { Agent } from "../llm_calls/fetch_agents";

describe("endpoint_config", () => {
  it("should export EndpointId constants", () => {
    expect(EndpointId.CHAT_COMPLETIONS).toBe("/v1/chat/completions");
    expect(EndpointId.A2A_AGENTS).toBe("/a2a");
  });

  it("should have endpoint configs for all endpoint IDs", () => {
    expect(ENDPOINT_CONFIGS[EndpointId.CHAT_COMPLETIONS]).toBeDefined();
    expect(ENDPOINT_CONFIGS[EndpointId.A2A_AGENTS]).toBeDefined();
    expect(ENDPOINT_CONFIGS[EndpointId.CHAT_COMPLETIONS].selectorType).toBe("model");
    expect(ENDPOINT_CONFIGS[EndpointId.A2A_AGENTS].selectorType).toBe("agent");
  });

  it("should get available endpoints", () => {
    const endpoints = getAvailableEndpoints();
    expect(endpoints).toHaveLength(2);
    expect(endpoints).toContainEqual({
      value: EndpointId.CHAT_COMPLETIONS,
      label: "/v1/chat/completions",
    });
    expect(endpoints).toContainEqual({
      value: EndpointId.A2A_AGENTS,
      label: "/a2a (Agents)",
    });
  });

  it("should get endpoint config by ID", () => {
    const config = getEndpointConfig(EndpointId.CHAT_COMPLETIONS);
    expect(config.id).toBe(EndpointId.CHAT_COMPLETIONS);
    expect(config.selectorType).toBe("model");
    expect(config.selectorLabel).toBe("Model");
  });

  it("should check if endpoint is agent endpoint", () => {
    expect(isAgentEndpoint(EndpointId.A2A_AGENTS)).toBe(true);
    expect(isAgentEndpoint(EndpointId.CHAT_COMPLETIONS)).toBe(false);
  });

  it("should check if endpoint is model endpoint", () => {
    expect(isModelEndpoint(EndpointId.CHAT_COMPLETIONS)).toBe(true);
    expect(isModelEndpoint(EndpointId.A2A_AGENTS)).toBe(false);
  });

  it("should convert model options to selector options", () => {
    const models = ["gpt-4", "gpt-3.5-turbo", "claude-3"];
    const options = modelOptionsToSelectorOptions(models);
    expect(options).toHaveLength(3);
    expect(options[0]).toEqual({ value: "gpt-4", label: "gpt-4" });
    expect(options[1]).toEqual({ value: "gpt-3.5-turbo", label: "gpt-3.5-turbo" });
    expect(options[2]).toEqual({ value: "claude-3", label: "claude-3" });
  });

  it("should convert agent options to selector options", () => {
    const agents: Agent[] = [
      { agent_id: "agent-1", agent_name: "Agent One" },
      { agent_id: "agent-2", agent_name: "Agent Two" },
      { agent_id: "agent-3", agent_name: undefined as any },
    ];
    const options = agentOptionsToSelectorOptions(agents);
    expect(options).toHaveLength(3);
    expect(options[0]).toEqual({ value: "Agent One", label: "Agent One" });
    expect(options[1]).toEqual({ value: "Agent Two", label: "Agent Two" });
    expect(options[2]).toEqual({ value: undefined, label: "agent-3" });
  });

  it("should get selection field name based on endpoint", () => {
    expect(getSelectionFieldName(EndpointId.CHAT_COMPLETIONS)).toBe("model");
    expect(getSelectionFieldName(EndpointId.A2A_AGENTS)).toBe("agent");
  });

  it("should get comparison selection based on endpoint", () => {
    const comparison = { model: "gpt-4", agent: "agent-1" };
    expect(getComparisonSelection(comparison, EndpointId.CHAT_COMPLETIONS)).toBe("gpt-4");
    expect(getComparisonSelection(comparison, EndpointId.A2A_AGENTS)).toBe("agent-1");
  });

  it("should check if comparison has valid selection", () => {
    const comparisonWithModel = { model: "gpt-4", agent: "" };
    const comparisonWithAgent = { model: "", agent: "agent-1" };
    const comparisonEmpty = { model: "", agent: "" };
    const comparisonWhitespace = { model: "   ", agent: "" };

    expect(hasValidSelection(comparisonWithModel, EndpointId.CHAT_COMPLETIONS)).toBe(true);
    expect(hasValidSelection(comparisonWithAgent, EndpointId.A2A_AGENTS)).toBe(true);
    expect(hasValidSelection(comparisonEmpty, EndpointId.CHAT_COMPLETIONS)).toBe(false);
    expect(hasValidSelection(comparisonWhitespace, EndpointId.CHAT_COMPLETIONS)).toBe(false);
  });
});
