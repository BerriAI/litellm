import { describe, it, expect } from "vitest";
import { selectionsFromSavedAgentCard, selectionsFromUpstreamCard, skillId } from "./agent_discovery_utils";

const upstreamCard = {
  name: "Upstream Agent",
  description: "Upstream description",
  capabilities: { streaming: true },
  skills: [
    { id: "search", name: "Search", description: "Search the web" },
    { id: "summarize", name: "Summarize", description: "Summarize docs" },
    { id: "chat", name: "Chat", description: "General chat" },
  ],
};

describe("selectionsFromSavedAgentCard", () => {
  it("pre-selects only skills that exist in the saved DB card", () => {
    const savedCard = {
      name: "My Agent",
      description: "Saved description",
      capabilities: { streaming: false },
      skills: [{ id: "search", name: "Search" }],
    };

    const result = selectionsFromSavedAgentCard(upstreamCard, savedCard);

    expect(result.editedName).toBe("My Agent");
    expect(result.editedDescription).toBe("Saved description");
    expect(result.selectedCapabilities.streaming).toBe(false);
    expect(result.selectedSkillIds.has(skillId(upstreamCard.skills![0], 0))).toBe(true);
    expect(result.selectedSkillIds.has(skillId(upstreamCard.skills![1], 1))).toBe(false);
    expect(result.selectedSkillIds.has(skillId(upstreamCard.skills![2], 2))).toBe(false);
  });

  it("matches saved skills by name when id is missing", () => {
    const savedCard = {
      skills: [{ name: "Summarize" }],
    };

    const result = selectionsFromSavedAgentCard(upstreamCard, savedCard);

    expect(result.selectedSkillIds.has(skillId(upstreamCard.skills![1], 1))).toBe(true);
    expect(result.selectedSkillIds.size).toBe(1);
  });
});

describe("selectionsFromUpstreamCard", () => {
  it("selects all upstream skills for create flow", () => {
    const result = selectionsFromUpstreamCard(upstreamCard);
    expect(result.selectedSkillIds.size).toBe(3);
    expect(result.editedName).toBe("Upstream Agent");
  });
});
