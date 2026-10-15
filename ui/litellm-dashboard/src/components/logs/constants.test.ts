import { describe, expect, it } from "vitest";

import { AGENT_CALL_TYPES, BATCH_CALL_TYPES, isAgentLog, MCP_CALL_TYPES } from "./constants";

describe("isAgentLog", () => {
  it.each(AGENT_CALL_TYPES)("recognizes %s without an agent ID", (callType) => {
    expect(isAgentLog({ call_type: callType })).toBe(true);
  });

  it.each(["acompletion", "anthropic_messages"] as const)("recognizes %s with an agent ID", (callType) => {
    expect(isAgentLog({ call_type: callType, agent_id: "agent-1" })).toBe(true);
  });

  it.each([undefined, null, ""] as const)("does not treat %s as an agent ID", (agentId) => {
    expect(isAgentLog({ call_type: "acompletion", agent_id: agentId })).toBe(false);
  });

  it.each(MCP_CALL_TYPES)("excludes MCP call type %s with an agent ID", (callType) => {
    expect(isAgentLog({ call_type: callType, agent_id: "agent-1" })).toBe(false);
  });

  it.each(BATCH_CALL_TYPES)("excludes batch call type %s with an agent ID", (callType) => {
    expect(isAgentLog({ call_type: callType, agent_id: "agent-1" })).toBe(false);
  });
});
