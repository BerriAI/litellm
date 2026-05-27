import { describe, expect, it } from "vitest";
import type { DailyData } from "../types";
import { SINGLE_DAY_TIME_LABEL, collapseDailyResults, getDailySpendChartData, isSingleDayRange } from "./dailySpendChart";
const bm = { spend:0, prompt_tokens:0, completion_tokens:0, total_tokens:0, api_requests:0, successful_requests:0, failed_requests:0, cache_read_input_tokens:0, cache_creation_input_tokens:0 };
function row(date: string, p: Partial<typeof bm> = {}): DailyData {
  return { date, metrics: { ...bm, ...p }, breakdown: { models:{}, model_groups:{}, mcp_servers:{}, providers:{}, api_keys:{}, entities:{} } };
}
describe("isSingleDayRange", () => {
  it("same day", () => { expect(isSingleDayRange({from:new Date(2026,4,26,0), to:new Date(2026,4,26,23,59,59)})).toBe(true); });
  it("two days", () => { expect(isSingleDayRange({from:new Date(2026,4,26), to:new Date(2026,4,27)})).toBe(false); });
  it("missing", () => {
    expect(isSingleDayRange(undefined)).toBe(false);
    expect(isSingleDayRange(null)).toBe(false);
    expect(isSingleDayRange({from:new Date(2026,4,26), to:undefined})).toBe(false);
  });
  it("midnight cross", () => { expect(isSingleDayRange({from:new Date(2026,2,7,23,30), to:new Date(2026,2,8,0,30)})).toBe(false); });
});
describe("collapseDailyResults", () => {
  it("sums", () => {
    const c = collapseDailyResults([row("2026-05-26",{spend:1.5,total_tokens:10,api_requests:2}), row("2026-05-27",{spend:2.5,total_tokens:30,api_requests:1})], SINGLE_DAY_TIME_LABEL);
    expect(c.date).toBe(SINGLE_DAY_TIME_LABEL); expect(c.metrics.spend).toBe(4); expect(c.metrics.total_tokens).toBe(40); expect(c.metrics.api_requests).toBe(3);
  });
  it("empty", () => {
    const c = collapseDailyResults([] as DailyData[], SINGLE_DAY_TIME_LABEL);
    expect(c.date).toBe(SINGLE_DAY_TIME_LABEL); expect(c.metrics.spend).toBe(0);
  });
  it("missing fields", () => {
    const rows: DailyData[] = [{ date:"2026-05-26", metrics: { ...bm, spend:1, successful_requests: undefined as unknown as number, failed_requests: undefined as unknown as number }, breakdown: { models:{}, model_groups:{}, mcp_servers:{}, providers:{}, api_keys:{}, entities:{} } }];
    expect(() => collapseDailyResults(rows, SINGLE_DAY_TIME_LABEL)).not.toThrow();
  });
});
describe("getDailySpendChartData", () => {
  const single = { from:new Date(2026,4,26,0), to:new Date(2026,4,26,23,59,59) };
  const multi = { from:new Date(2026,4,24,0), to:new Date(2026,4,26,23,59,59) };
  it("collapses single day", () => {
    const out = getDailySpendChartData([row("2026-05-26",{spend:1.5}), row("2026-05-27",{spend:2.5})], single);
    expect(out).toHaveLength(1); expect(out[0].date).toBe(SINGLE_DAY_TIME_LABEL); expect(out[0].metrics.spend).toBe(4);
  });
  it("sorts multi", () => {
    const out = getDailySpendChartData([row("2026-05-25",{spend:2}), row("2026-05-24",{spend:1}), row("2026-05-26",{spend:3})], multi);
    expect(out.map(r=>r.date)).toEqual(["2026-05-24","2026-05-25","2026-05-26"]); expect(out.map(r=>r.metrics.spend)).toEqual([1,2,3]);
  });
  it("empty", () => { expect(getDailySpendChartData([], single)).toEqual([]); expect(getDailySpendChartData([], multi)).toEqual([]); });
  it("no mutate", () => {
    const rows = [row("2026-05-25"), row("2026-05-24")]; const before = rows.map(r=>r.date);
    getDailySpendChartData(rows, multi); expect(rows.map(r=>r.date)).toEqual(before);
  });
  it("undefined date", () => {
    const out = getDailySpendChartData([row("2026-05-25"), row("2026-05-24")], undefined);
    expect(out.map(r=>r.date)).toEqual(["2026-05-24","2026-05-25"]);
  });
});

describe("collapseDailyResults — breakdown merging (LIT-3383 P1)", () => {
  const baseRow = (date: string, opts: { models?: Record<string, number>; entities?: Record<string, number> } = {}): DailyData => ({
    date,
    metrics: { ...bm },
    breakdown: {
      models: Object.fromEntries(
        Object.entries(opts.models ?? {}).map(([k, v]) => [
          k,
          { metrics: { ...bm, spend: v }, metadata: {}, api_key_breakdown: {} },
        ]),
      ),
      model_groups: {},
      mcp_servers: {},
      providers: {},
      api_keys: {},
      entities: Object.fromEntries(
        Object.entries(opts.entities ?? {}).map(([k, v]) => [
          k,
          { metrics: { ...bm, spend: v }, metadata: { alias: k, id: k }, api_key_breakdown: {} },
        ]),
      ),
    },
  });

  it("sums entity spend across all source rows, not just the first", () => {
    const rows = [
      baseRow("2026-05-26", { entities: { e1: 1.5, e2: 0.5 } }),
      baseRow("2026-05-27", { entities: { e1: 1.0, e3: 2.0 } }),
    ];
    const c = collapseDailyResults(rows, SINGLE_DAY_TIME_LABEL);
    expect(c.breakdown.entities.e1.metrics.spend).toBe(2.5);
    expect(c.breakdown.entities.e2.metrics.spend).toBe(0.5);
    expect(c.breakdown.entities.e3.metrics.spend).toBe(2.0);
  });

  it("merges model breakdown across rows", () => {
    const rows = [
      baseRow("2026-05-26", { models: { "gpt-4": 1.0 } }),
      baseRow("2026-05-27", { models: { "gpt-4": 2.0, "claude-3": 1.0 } }),
    ];
    const c = collapseDailyResults(rows, SINGLE_DAY_TIME_LABEL);
    expect(c.breakdown.models["gpt-4"].metrics.spend).toBe(3.0);
    expect(c.breakdown.models["claude-3"].metrics.spend).toBe(1.0);
  });

  it("returns an empty breakdown when all source rows are empty", () => {
    const c = collapseDailyResults([baseRow("2026-05-26")], SINGLE_DAY_TIME_LABEL);
    expect(Object.keys(c.breakdown.entities)).toHaveLength(0);
    expect(Object.keys(c.breakdown.models)).toHaveLength(0);
  });
});
