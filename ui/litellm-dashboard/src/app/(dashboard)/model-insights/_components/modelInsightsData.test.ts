import { describe, expect, it } from "vitest";

import {
  buildTaskTiles,
  buildWeeklySeries,
  DailyMetric,
  formatMetric,
  modelOrder,
  rankModels,
  TaskMetric,
} from "./modelInsightsData";

const row = (over: Partial<DailyMetric>): DailyMetric => ({
  model_group: "a",
  model: "a",
  provider: "openai",
  date: "2026-01-01",
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  ...over,
});

describe("buildWeeklySeries", () => {
  it("sums days into 7-day buckets per model", () => {
    const rows = [
      row({ date: "2026-01-01", requests: 1 }),
      row({ date: "2026-01-07", requests: 2 }),
      row({ date: "2026-01-08", requests: 4 }),
      row({ date: "2026-01-02", model_group: "b", requests: 8 }),
    ];
    expect(buildWeeklySeries(rows, ["a", "b"], "requests")).toEqual([
      { date: "2026-01-01", a: 3, b: 8 },
      { date: "2026-01-08", a: 4, b: 0 },
    ]);
  });

  it("returns nothing for no data", () => {
    expect(buildWeeklySeries([], ["a"], "tokens")).toEqual([]);
  });
});

describe("modelOrder", () => {
  it("orders models by the selected metric, largest first", () => {
    const rows = [row({ model_group: "a", spend: 1, requests: 9 }), row({ model_group: "b", spend: 5, requests: 1 })];
    expect(modelOrder(rows, "spend")).toEqual(["b", "a"]);
    expect(modelOrder(rows, "requests")).toEqual(["a", "b"]);
  });
});

describe("rankModels", () => {
  it("computes share and the change in share between the first and second half", () => {
    const daily = [
      row({ date: "2026-01-01", model_group: "a", requests: 30 }),
      row({ date: "2026-01-01", model_group: "b", requests: 10 }),
      row({ date: "2026-01-02", model_group: "a", requests: 10 }),
      row({ date: "2026-01-02", model_group: "b", requests: 30 }),
    ];
    const totals = [{ ...row({ model_group: "a", requests: 40 }) }, { ...row({ model_group: "b", requests: 40 }) }];
    const ranked = rankModels(totals, daily, "requests");
    expect(ranked.find((m) => m.model_group === "a")).toMatchObject({ share: 50, delta: -50 });
    expect(ranked.find((m) => m.model_group === "b")).toMatchObject({ share: 50, delta: 50 });
  });
});

describe("buildTaskTiles", () => {
  it("labels each task, assigns its category, and names the leading model", () => {
    const rows: TaskMetric[] = [
      { ...row({ model_group: "a", spend: 6 }), task_type: "code_generation" },
      { ...row({ model_group: "b", spend: 2 }), task_type: "code_generation" },
      { ...row({ model_group: "c", spend: 2 }), task_type: "classification" },
    ];
    const tiles = buildTaskTiles(rows, "spend");
    expect(tiles.map((t) => [t.label, t.category, t.share, t.leader])).toEqual([
      ["Code Generation", "Code", 80, "a"],
      ["Classification", "General", 20, "c"],
    ]);
  });
});

describe("formatMetric", () => {
  it("formats spend as currency and counts compactly", () => {
    expect(formatMetric(12.5, "spend")).toBe("$12.50");
    expect(formatMetric(1_500_000, "tokens")).toBe("1.5M");
  });
});
