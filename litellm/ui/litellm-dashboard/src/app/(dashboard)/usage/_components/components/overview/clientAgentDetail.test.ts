import { describe, expect, it } from "vitest";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import type { AgentRow } from "./agentCatalog";
import { clientAgentDetail } from "./clientAgentDetail";

const desktopTag = "User-Agent: claude-cli/2.1.1 (external, claude-desktop-3p)";
const codeTag = "User-Agent: claude-cli/2.1.1 (external, cli)";
const agent: AgentRow = {
  id: "claude-desktop",
  label: "Claude Desktop",
  description: "Desktop assistant",
  kind: "agent",
  logo: null,
  spend: 3,
  tokens: 300,
  requests: 3,
  users: 2,
  tags: [desktopTag],
};
const oneRequestAgent = { ...agent, spend: 1, tokens: 100, requests: 1 };
const unknownTokenBreakdown = { inputTokens: null, outputTokens: null, cacheReadTokens: null, cacheWriteTokens: null };

const metrics = (overrides: Partial<SpendMetrics> = {}): SpendMetrics => ({
  spend: 1,
  prompt_tokens: 80,
  completion_tokens: 20,
  total_tokens: 100,
  api_requests: 1,
  successful_requests: 1,
  failed_requests: 0,
  cache_read_input_tokens: 30,
  cache_creation_input_tokens: 10,
  ...overrides,
});

const day = (date: string, entities: Record<string, SpendMetrics>): DailyData => ({
  date,
  metrics: metrics({ spend: 999, total_tokens: 999_999, api_requests: 999 }),
  breakdown: {
    models: { unrelated: { metrics: metrics({ spend: 999 }), metadata: {}, api_key_breakdown: {} } },
    model_groups: {},
    providers: {},
    api_keys: {},
    mcp_servers: {},
    entities: Object.fromEntries(
      Object.entries(entities).map(([tag, value]) => [tag, { metrics: value, metadata: {}, api_key_breakdown: {} }]),
    ),
  },
});

describe("clientAgentDetail", () => {
  it("should preserve the clicked summary before daily data arrives", () => {
    const expected = {
      summary: { spend: 3, tokens: 300, requests: 3 },
      dailyMatchesSummary: false,
      tokenBreakdown: unknownTokenBreakdown,
    };
    expect(clientAgentDetail(agent, [])).toEqual(expected);
  });

  it("should reconcile exact contributing tags once without global traffic", () => {
    const secondTag = "User-Agent: claude-desktop/2.0";
    const result = clientAgentDetail({ ...agent, tags: [desktopTag, secondTag, desktopTag] }, [
      day("2026-10-22", { [desktopTag]: metrics(), [secondTag]: metrics(), [codeTag]: metrics({ spend: 50 }) }),
      day("2026-10-20", { [desktopTag]: metrics(), "User-Agent: claude-cli": metrics({ spend: 500 }) }),
      day("2026-10-21", { [codeTag]: metrics() }),
    ]);

    expect(result.dailyMatchesSummary).toBe(true);
    const expectedTokens = {
      inputTokens: 240,
      outputTokens: 60,
      cacheReadTokens: 90,
      cacheWriteTokens: 30,
    };
    expect(result.tokenBreakdown).toEqual(expectedTokens);
  });

  it("should keep missing cache counts unknown without hiding recorded token counts", () => {
    const result = clientAgentDetail(oneRequestAgent, [
      day("2026-10-22", { [desktopTag]: metrics({ cache_read_input_tokens: undefined }) }),
    ]);

    const expectedTokens = {
      inputTokens: 80,
      outputTokens: 20,
      cacheReadTokens: null,
      cacheWriteTokens: null,
    };
    expect(result.tokenBreakdown).toEqual(expectedTokens);
  });

  it("should distinguish recorded zero caching from missing counts", () => {
    const result = clientAgentDetail(oneRequestAgent, [
      day("2026-10-22", {
        [desktopTag]: metrics({ cache_read_input_tokens: 0, cache_creation_input_tokens: 0 }),
      }),
    ]);

    expect(result.tokenBreakdown.cacheReadTokens).toBe(0);
    expect(result.tokenBreakdown.cacheWriteTokens).toBe(0);
  });

  it("should preserve summary totals but withhold token breakdown when daily snapshots disagree", () => {
    const result = clientAgentDetail(agent, [day("2026-10-22", { [desktopTag]: metrics() })]);

    expect(result.summary).toEqual({ spend: 3, tokens: 300, requests: 3 });
    expect(result.dailyMatchesSummary).toBe(false);
    expect(result.tokenBreakdown).toEqual(unknownTokenBreakdown);
  });

  it("should not turn unrelated tags or ambiguous residuals into zero client activity", () => {
    const unrelated = clientAgentDetail(agent, [day("2026-10-22", { [codeTag]: metrics() })]);
    const residual = clientAgentDetail({ ...agent, tags: [] }, [day("2026-10-22", { [desktopTag]: metrics() })]);

    expect(unrelated.tokenBreakdown).toEqual(unknownTokenBreakdown);
    expect(residual.tokenBreakdown).toEqual(unknownTokenBreakdown);
    expect(residual.summary.spend).toBe(agent.spend);
  });

  it("should allow floating-point rounding without dropping known breakdowns", () => {
    const decimalAgent = { ...agent, spend: 0.3, tokens: 200, requests: 2 };
    const result = clientAgentDetail(decimalAgent, [
      day("2026-10-21", { [desktopTag]: metrics({ spend: 0.1 }) }),
      day("2026-10-22", { [desktopTag]: metrics({ spend: 0.2 }) }),
    ]);

    expect(result.dailyMatchesSummary).toBe(true);
    expect(result.tokenBreakdown.inputTokens).toBe(160);
  });

  it.each([
    { prompt_tokens: 100 },
    { cache_read_input_tokens: 80, cache_creation_input_tokens: 10 },
    { cache_read_input_tokens: Number.NaN },
    { cache_read_input_tokens: Number.POSITIVE_INFINITY },
    { cache_read_input_tokens: -1 },
  ])("should suppress cache counts that do not reconcile (%o)", (invalidCounts) => {
    const result = clientAgentDetail(oneRequestAgent, [day("2026-10-22", { [desktopTag]: metrics(invalidCounts) })]);

    expect(result.tokenBreakdown.cacheReadTokens).toBeNull();
    expect(result.tokenBreakdown.cacheWriteTokens).toBeNull();
  });

  it("should withhold non-finite daily counts while preserving the known summary", () => {
    const result = clientAgentDetail(agent, [
      day("2026-10-22", { [desktopTag]: metrics({ total_tokens: Number.NaN }) }),
    ]);

    expect(result.dailyMatchesSummary).toBe(false);
    expect(result.summary.tokens).toBe(300);
    expect(result.tokenBreakdown.inputTokens).toBeNull();
  });
});
