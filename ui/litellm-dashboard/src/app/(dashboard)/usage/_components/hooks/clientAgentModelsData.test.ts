import { describe, expect, it } from "vitest";
import type { AgentRow } from "../components/overview/agentCatalog";
import { clientAgentDetail } from "../components/overview/clientAgentDetail";
import { clientAgentModelsFromExport } from "./clientAgentModelsData";

const desktop = "User-Agent: claude-cli/2.1.1 (external, claude-desktop-3p)";
const code = "User-Agent: claude-cli/2.1.1 (external, cli)";
const record = {
  date: "2026-10-22",
  entity_id: desktop,
  model: "model-a",
  spend: 1,
  flat_cost: 0,
  prompt_tokens: 80,
  completion_tokens: 20,
  api_requests: 2,
  successful_requests: 2,
  failed_requests: 0,
  cache_read_input_tokens: 30,
  cache_creation_input_tokens: 10,
};
const agent: AgentRow = {
  id: "claude-desktop",
  label: "Claude Desktop",
  description: "Desktop client",
  kind: "agent",
  logo: null,
  spend: 3,
  tokens: 300,
  requests: 6,
  users: 1,
  tags: [desktop, desktop],
};
const records = [
  { ...record, model: "model-b", spend: 2, prompt_tokens: 160, completion_tokens: 40, api_requests: 4 },
  record,
  { ...record, entity_id: code, spend: 400, model: "code-only" },
  { ...record, entity_id: "User-Agent: claude-cli", spend: 500, model: "bare-rollup" },
];

describe("clientAgentModelsFromExport", () => {
  it("should return exact Desktop models without Code or bare-tag totals", () => {
    const result = clientAgentModelsFromExport(records, agent);

    expect(result.error).toBeNull();
    expect(result.results).toHaveLength(1);
    expect(Object.keys(result.results[0].breakdown.models)).toEqual(["model-a", "model-b"]);
    expect(result.results[0].metrics).toMatchObject({ spend: 3, total_tokens: 300, api_requests: 6 });
    expect(result.results[0].breakdown.models["model-b"].metrics.spend).toBe(2);
    expect(Object.keys(result.results[0].breakdown.entities)).toEqual([desktop]);
    expect(clientAgentDetail(agent, result.results).dailyMatchesSummary).toBe(true);
  });

  it("should preserve recorded input and cache counts for the panel explanation", () => {
    const { results } = clientAgentModelsFromExport(records, agent);
    const expected = { inputTokens: 240, outputTokens: 60, cacheReadTokens: 60, cacheWriteTokens: 20 };
    expect(clientAgentDetail(agent, results).tokenBreakdown).toEqual(expected);
  });

  it("should sum the same model across canonical tags and sort days", () => {
    const secondTag = "User-Agent: claude-desktop/3.0";
    const multiTagAgent = { ...agent, tags: [desktop, secondTag] };
    const rows = [record, { ...record, entity_id: secondTag }, { ...record, date: "2026-10-21" }];
    const { results } = clientAgentModelsFromExport(rows, multiTagAgent);

    expect(results.map((day) => day.date)).toEqual(["2026-10-21", "2026-10-22"]);
    expect(results[1].breakdown.models["model-a"].metrics.total_tokens).toBe(200);
    expect(results[1].metrics.spend).toBe(2);
  });

  it("should label missing model names without losing their recorded usage", () => {
    const oneRow = { ...record, model: null };
    const singleAgent = { ...agent, spend: 1, tokens: 100, requests: 2 };
    const result = clientAgentModelsFromExport([oneRow], singleAgent);

    expect(result.error).toBeNull();
    expect(result.results[0].breakdown.models["Unknown model"].metrics.spend).toBe(1);
  });

  it.each(["cache_read_input_tokens", "prompt_tokens", "api_requests"])(
    "should reject missing %s without inventing zero telemetry",
    (key) => {
      const incomplete = { ...record, [key]: undefined };
      const result = clientAgentModelsFromExport([incomplete], agent);
      expect(result.error).toContain("incomplete");
      expect(result.results).toEqual([]);
    },
  );

  it("should reject duplicate tag, model and date rows rather than double-count them", () => {
    const result = clientAgentModelsFromExport([...records, record], agent);
    expect(result.error).toContain("duplicate");
    expect(result.results).toEqual([]);
  });

  it.each([{ spend: 4 }, { tokens: 400 }, { requests: 8 }])(
    "should withhold models when the client summary changed (%o)",
    (changed) => {
      const result = clientAgentModelsFromExport(records, { ...agent, ...changed });
      expect(result.error).toContain("does not match");
      expect(result.results).toEqual([]);
    },
  );

  it("should never fall back to unrelated models when no exact tags match", () => {
    const otherAgent = { ...agent, tags: ["User-Agent: absent"] };
    const result = clientAgentModelsFromExport(records, otherAgent);
    expect(result.error).not.toBeNull();
    expect(result.results).toEqual([]);
  });
});
