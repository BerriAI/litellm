import { describe, expect, it } from "vitest";
import { toAgentPublicationRequest } from "./makeAgentPublicMapper";
import { makeAgentPublicSchema } from "./makeAgentPublicSchema";

describe("Agent publication", () => {
  it("keeps the selected identifiers in selection order in the request", () => {
    const agentIds = ["agent-z", "agent-a"];
    expect(toAgentPublicationRequest(makeAgentPublicSchema.parse({ agentIds }))).toEqual({
      agent_ids: agentIds,
    });
  });

  it("rejects an empty selection", () => {
    expect(makeAgentPublicSchema.safeParse({ agentIds: [] }).success).toBe(false);
  });
});
