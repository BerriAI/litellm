import { describe, expect, it } from "vitest";

import {
  UNASSIGNED_COLOR,
  cellReading,
  coverage,
  entryLookup,
  formatLatency,
  formatReading,
  heatIntensity,
  shareOfTask,
  taskColors,
  type TaskMatrixEntry,
} from "./taskInsightsData";

const entry = (overrides: Partial<TaskMatrixEntry>): TaskMatrixEntry => ({
  model_config_id: "m1",
  task_id: "t1",
  request_count: 10,
  avg_cost_usd: 0.02,
  cost_observations: 10,
  p95_latency_ms: 2400,
  latency_observations: 10,
  eval_success_rate: null,
  eval_sample_count: 0,
  ...overrides,
});

describe("cellReading", () => {
  it("treats a cell without cost observations as missing evidence, not as free", () => {
    const reading = cellReading(entry({ avg_cost_usd: null, cost_observations: 0 }), "cost");

    expect(reading).toEqual({ value: null, observations: 0 });
    expect(formatReading(reading, "cost")).toBeNull();
  });

  it("treats a cell without latency observations as missing evidence", () => {
    expect(cellReading(entry({ p95_latency_ms: null, latency_observations: 0 }), "latency").value).toBeNull();
  });

  it("reads the request count for the share metric", () => {
    expect(cellReading(entry({ request_count: 7 }), "share")).toEqual({ value: 7, observations: 7 });
  });

  it("returns no evidence for a deployment that never served the task", () => {
    expect(cellReading(undefined, "share")).toEqual({ value: null, observations: 0 });
  });
});

describe("entryLookup", () => {
  it("finds the entry for one deployment and task without mixing up ids that share a prefix", () => {
    const lookup = entryLookup([entry({ model_config_id: "m1", task_id: "t1" }), entry({ model_config_id: "m1t", task_id: "1" })]);

    expect(lookup("m1", "t1")?.task_id).toBe("t1");
    expect(lookup("m1t", "1")?.model_config_id).toBe("m1t");
    expect(lookup("m2", "t1")).toBeUndefined();
  });
});

describe("shareOfTask", () => {
  it("divides a deployment's requests by every request in the task", () => {
    const entries = [entry({ model_config_id: "a", request_count: 3 }), entry({ model_config_id: "b", request_count: 1 })];

    expect(shareOfTask(entries, "a", "t1")).toBe(0.75);
    expect(shareOfTask(entries, "c", "t1")).toBeNull();
  });
});

describe("heatIntensity", () => {
  it("keeps missing cells unshaded and spans the observed range", () => {
    const values = [1, 5, null];

    expect(heatIntensity(null, values)).toBeNull();
    expect(heatIntensity(1, values)).toBeCloseTo(0.12);
    expect(heatIntensity(5, values)).toBe(1);
  });

  it("shades a single observed value fully", () => {
    expect(heatIntensity(3, [3, null])).toBe(1);
  });
});

describe("taskColors", () => {
  it("colors tasks by rank and greys the unknown and unclassified buckets", () => {
    const colors = taskColors([
      { task_id: "a", kind: "task" },
      { task_id: "b", kind: "task" },
      { task_id: "unknown", kind: "unknown" },
    ]);

    expect(colors.get("a")).not.toBe(colors.get("b"));
    expect(colors.get("unknown")).toBe(UNASSIGNED_COLOR);
  });
});

describe("formatting", () => {
  it("switches latency to seconds from one second up", () => {
    expect(formatLatency(950)).toBe("950 ms");
    expect(formatLatency(2400)).toBe("2.4 s");
    expect(formatLatency(null)).toBeNull();
  });

  it("reports coverage as the classified share of requests", () => {
    const totals = {
      request_count: 10,
      classified_count: 8,
      delivery_count: 12,
      session_count: 4,
      spend: 1,
      cost_observations: 9,
      unattributed_count: 1,
      unknown_count: 1,
      unclassified_count: 1,
      analysis_request_count: 0,
      analysis_spend: null,
    };

    expect(coverage(totals)).toBe(0.8);
    expect(coverage({ ...totals, request_count: 0, classified_count: 0 })).toBeNull();
  });
});
