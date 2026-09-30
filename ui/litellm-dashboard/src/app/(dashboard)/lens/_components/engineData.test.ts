import { describe, expect, it } from "vitest";
import {
  analysisElapsed,
  analysisProgress,
  normalizeFilters,
  sortedFindings,
  type Finding,
  type Job,
} from "./engineData";

const job: Job = {
  id: "scan",
  status: "running",
  stage: "Reading executions",
  created_at: "2026-09-30T12:00:00Z",
  start: "2026-09-29T12:00:00Z",
  end: "2026-09-30T12:00:00Z",
  revision: 1,
  settings: {
    name: "Release reviews",
    model: "analysis",
    checks: [{ id: "failures", instruction: "Find failed outcomes" }],
  },
};

describe("Analysis progress", () => {
  it("measures review progress against the sample, not all eligible runs", () => {
    const expected = { step: 0, done: 7, total: 20, detail: "7 of 20 selected runs reviewed" };
    expect(analysisProgress({ ...job, coverage: { eligible: 1000, selected: 20, screened: 7 } })).toMatchObject(
      expected,
    );
  });

  it("shows actual grouping progress instead of treating reviewed runs as a finished scan", () => {
    const expected = { step: 1, done: 2, total: 4, detail: "2 of 4 observation batches compared" };
    expect(
      analysisProgress({
        ...job,
        stage: "Grouping observations",
        coverage: { screened: 21, grouped_batches: 2, grouping_batches: 4 },
      }),
    ).toMatchObject(expected);
  });

  it("keeps older worker grouping responses indeterminate", () => {
    expect(analysisProgress({ ...job, stage: "Grouping observations", coverage: { screened: 21 } })).toMatchObject({
      step: 1,
      total: 0,
      detail: "Comparing observations across 21 reviewed runs",
    });
  });

  it("shows verified candidate counts separately from run counts", () => {
    expect(
      analysisProgress({
        ...job,
        stage: "Checking original evidence",
        coverage: { screened: 21, investigated: 2, candidates: 5 },
      }),
    ).toMatchObject({
      step: 2,
      done: 2,
      total: 5,
    });
  });

  it("does not show queued work as started", () => {
    expect(analysisProgress({ ...job, status: "queued" })).toMatchObject({
      step: -1,
      total: 0,
      title: "Waiting for a worker",
    });
  });

  it("shows elapsed time and clamps future timestamps during clock skew", () => {
    expect(analysisElapsed(job.created_at, Date.parse("2026-09-30T12:02:13Z"))).toBe("2m 13s");
    expect(analysisElapsed(job.created_at, Date.parse("2026-09-30T11:59:59Z"))).toBe("0s");
  });
});

describe("Lens selection and findings", () => {
  it("preserves literal equals signs in a metadata value", () => {
    expect(normalizeFilters([{ key: " swarm ", value: " research=v2 " }])).toEqual([
      { key: "swarm", value: "research=v2" },
    ]);
  });
  it("rejects an incomplete condition instead of broadening the scan", () => {
    expect(() => normalizeFilters([{ key: "swarm", value: " " }])).toThrow("Choose a key and value");
  });
  it("puts high priority issues ahead of newer low priority findings", () => {
    const base: Finding = {
      id: "low",
      check_id: "check",
      title: "Recovered error",
      description: "The run recovered.",
      evidence: [],
      revision: 1,
      priority: "low",
      first_seen: "2026-09-30T10:00:00Z",
      last_seen: "2026-09-30T12:00:00Z",
    };
    const high: Finding = { ...base, id: "high", priority: "high", last_seen: "2026-09-30T11:00:00Z" };
    expect(sortedFindings([base, high]).map((f) => f.id)).toEqual(["high", "low"]);
  });
});
