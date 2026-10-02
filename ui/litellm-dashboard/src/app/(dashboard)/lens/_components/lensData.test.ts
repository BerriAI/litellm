import { describe, expect, it } from "vitest";
import {
  analysisElapsed,
  workerConnected,
  type LensList,
  analysisProgress,
  normalizeFilters,
  sortedFindings,
  type Finding,
  type Job,
} from "./lensData";

const coverage: Job["coverage"] = {
  eligible: 0,
  selected: 0,
  screened: 0,
  investigated: 0,
  inconclusive: 0,
  grouping_batches: 0,
  grouped_batches: 0,
  candidates: 0,
  partial: 0,
  unassessable: 0,
};

const job: Job = {
  assessments: [],
  coverage,
  attempts: 0,
  error: "",
  cost: 0,
  id: "scan",
  status: "running",
  stage: "Reading executions",
  created_at: "2026-09-30T12:00:00Z",
  start: "2026-09-29T12:00:00Z",
  end: "2026-09-30T12:00:00Z",
  revision: 1,
  settings: {
    context: "",
    source: "traces",
    lookback_hours: 24,
    service: "",
    agent_name: "",
    filters: [],
    enabled: false,
    interval_minutes: 15,
    sample_size: 100,
    sample_percent: 100,
    concurrency: 8,
    team_id: "",
    execution_ids: [],
    monthly_budget: 20,
    name: "Release reviews",
    model: "analysis",
    checks: [{ enabled: true, id: "failures", instruction: "Find failed outcomes" }],
  },
};

describe("Analysis progress", () => {
  it("measures review progress against the sample, not all eligible runs", () => {
    const expected = { step: 0, done: 7, total: 20, detail: "7 of 20 selected runs reviewed" };
    expect(
      analysisProgress({ ...job, coverage: { ...coverage, eligible: 1000, selected: 20, screened: 7 } }),
    ).toMatchObject(expected);
  });

  it("shows actual grouping progress instead of treating reviewed runs as a finished scan", () => {
    const expected = { step: 1, done: 2, total: 4, detail: "2 of 4 observation batches compared" };
    expect(
      analysisProgress({
        ...job,
        stage: "Grouping observations",
        coverage: { ...coverage, screened: 21, grouped_batches: 2, grouping_batches: 4 },
      }),
    ).toMatchObject(expected);
  });

  it("keeps older worker grouping responses indeterminate", () => {
    expect(
      analysisProgress({ ...job, stage: "Grouping observations", coverage: { ...coverage, screened: 21 } }),
    ).toMatchObject({
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
        coverage: { ...coverage, screened: 21, investigated: 2, candidates: 5 },
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
      title: "Queued for your worker",
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
      kind: "issue",
      status: "open",
      reason: "",
      suggestion: "",
      limitation: "",
      occurrences: [],
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

describe("Worker readiness", () => {
  const now = Date.parse("2026-10-01T12:00:00Z");
  const worker: LensList["workers"][number] = {
    id: "worker",
    name: "Worker",
    revoked: false,
    analysis_key_id: "key",
    last_seen: "2026-10-01T11:59:59Z",
    scope: { all_teams: true, api_key_hash: "", team_id: "" },
  };
  it("requires a current heartbeat and assigned billing key", () => {
    expect(workerConnected(worker, now)).toBe(true);
    expect(workerConnected({ ...worker, last_seen: "" }, now)).toBe(false);
    expect(workerConnected({ ...worker, last_seen: "2026-10-01T11:58:00Z" }, now)).toBe(false);
    expect(workerConnected({ ...worker, analysis_key_id: "" }, now)).toBe(false);
    expect(workerConnected({ ...worker, revoked: true }, now)).toBe(false);
  });
});
