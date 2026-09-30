import { describe, expect, it } from "vitest";
import { getGlobalTopKeys, getTopModels, getTopAgents } from "./entityUsageAggregations";
import type { DailyData } from "@/components/UsagePage/types";

const partialDay = {
  date: "2026-09-01",
  breakdown: {
    models: { model: { metadata: {} } },
    api_keys: { key: { metadata: { key_alias: "test key" } } },
    entities: { agent: { metadata: { agent_name: "test agent" }, api_key_breakdown: {} } },
  },
} as unknown as DailyData;

describe("partial breakdown metrics", () => {
  it("defaults missing global-key metrics to zero", () => {
    expect(getGlobalTopKeys([partialDay], 10)[0].spend).toBe(0);
  });
  it("defaults missing model metrics to zero", () => {
    expect(getTopModels([partialDay], "models", 10)[0].tokens).toBe(0);
  });
  it("defaults missing agent metrics to zero", () => {
    expect(getTopAgents([partialDay], 10)[0].spend).toBe(0);
  });
  it("still sums populated days after a partial day", () => {
    const goodDay = {
      ...partialDay,
      breakdown: { ...partialDay.breakdown, api_keys: {
        key: { metadata: { key_alias: "test key" }, metrics: { spend: 5 } },
      } },
    } as unknown as DailyData;
    expect(getGlobalTopKeys([partialDay, goodDay], 10)[0].spend).toBe(5);
  });
});
