import { describe, expect, it } from "vitest";
import { categoryChart, categorySummary, requestQuote, type Report } from "./data";

const report: Report = {
  version: "v1",
  categories: [{ id: "build", name: "Build", description: "Build pipelines" }],
  buckets: [
    { day: "2026-01-01", category_id: "build", model: "model-a", spend: 2, requests: 2, unknown_cost_requests: 0 },
    { day: "2026-01-01", category_id: "build", model: "model-b", spend: 3, requests: 1, unknown_cost_requests: 0 },
    {
      day: "2026-01-01",
      category_id: "unclassified",
      model: "model-a",
      spend: 1,
      requests: 4,
      unknown_cost_requests: 0,
    },
    { day: "2026-01-02", category_id: "build", model: "model-a", spend: 4, requests: 1, unknown_cost_requests: 0 },
  ],
};

describe("category spend", () => {
  it("reconciles model totals with category totals including unclassified traffic", () => {
    const rows = categorySummary(report);
    expect(rows.map(({ id, spend, requests }) => ({ id, spend, requests }))).toEqual([
      { id: "build", spend: 9, requests: 4 },
      { id: "unclassified", spend: 1, requests: 4 },
    ]);
    expect(rows[0].models).toEqual([
      { model: "model-a", spend: 6 },
      { model: "model-b", spend: 3 },
    ]);
  });
  it("adds different models into the same category and keeps empty buckets at zero", () => {
    expect(categoryChart(report)).toEqual([
      { day: "2026-01-01", build: 5, unclassified: 1 },
      { day: "2026-01-02", build: 4, unclassified: 0 },
    ]);
  });
  it("uses an actual user quote and preserves incomplete excerpts", () => {
    expect(requestQuote('[{"role":"system","content":"hidden"},{"role":"user","content":"Fix my build"}]')).toBe(
      "Fix my build",
    );
    expect(requestQuote('[{"role":"user"')).toBe('[{"role":"user"');
  });
});
