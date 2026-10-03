import { describe, expect, it } from "vitest";
import {
  analysisElapsed,
  workerConnected,
  type LensList,
  analysisProgress,
  analysisFraction,
  analysisPace,
  remainingLabel,
  stageDurations,
  normalizeFilters,
  sortedFindings,
  briefMarkdown,
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
  it.each(["Collecting executions", "Reading executions"])(
    "does not turn an unreported run count into zero runs during %s",
    (stage) => {
      expect(analysisProgress({ ...job, stage })).toEqual({
        step: -1,
        title: "Preparing activity",
        done: 0,
        total: 0,
        detail: "Loading the runs selected for this investigation.",
      });
    },
  );

  it("shows the selected count as soon as the worker reports it", () => {
    expect(analysisProgress({ ...job, coverage: { ...coverage, selected: 7 } })).toMatchObject({
      step: 0,
      done: 0,
      total: 7,
      detail: "0 of 7 selected runs reviewed",
    });
  });

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

describe("Analysis pace", () => {
  it("fills the bar left to right across stages without jumping backwards at a stage boundary", () => {
    const endOfReview = analysisFraction({ step: 0, done: 20, total: 20 });
    const startOfGrouping = analysisFraction({ step: 1, done: 0, total: 4 });
    expect(analysisFraction({ step: -1, done: 0, total: 0 })).toBe(0);
    expect(analysisFraction({ step: 0, done: 10, total: 20 })).toBeLessThan(endOfReview);
    expect(startOfGrouping).toBeCloseTo(endOfReview);
    expect(analysisFraction({ step: 2, done: 5, total: 5 })).toBeCloseTo(1);
  });

  it("measures speed within the current stage and projects time left from overall progress", () => {
    const start = Date.parse("2026-09-30T12:00:00Z");
    const pace = analysisPace(
      [
        { at: start, step: 0, done: 0, fraction: 0 },
        { at: start + 30000, step: 0, done: 30, fraction: 0.25 },
      ],
      start + 30000,
    );
    expect(pace.perMinute).toBe(60);
    expect(pace.secondsLeft).toBe(90);
  });

  it("waits for enough samples instead of showing a wild first estimate", () => {
    const start = Date.parse("2026-09-30T12:00:00Z");
    expect(analysisPace([{ at: start, step: 0, done: 1, fraction: 0.01 }], start + 2000)).toEqual({
      perMinute: null,
      secondsLeft: null,
    });
    expect(remainingLabel(null)).toBe("estimating");
  });

  it("lengthens the estimate while progress stalls", () => {
    const start = Date.parse("2026-09-30T12:00:00Z");
    const samples = [
      { at: start, step: 0, done: 0, fraction: 0 },
      { at: start + 10000, step: 0, done: 10, fraction: 0.1 },
    ];
    const moving = analysisPace(samples, start + 10000).secondsLeft ?? 0;
    const stalled = analysisPace(samples, start + 40000).secondsLeft ?? 0;
    expect(stalled).toBeGreaterThan(moving);
  });

  it("still estimates when the worker reports progress less than once a minute", () => {
    const start = Date.parse("2026-09-30T12:00:00Z");
    const pace = analysisPace(
      [
        { at: start, step: 2, done: 0, fraction: 0.8 },
        { at: start + 90000, step: 2, done: 1, fraction: 0.85 },
      ],
      start + 90000,
    );
    expect(pace.secondsLeft).toBeCloseTo(270);
    expect(pace.perMinute).toBeCloseTo(2 / 3);
  });

  it("measures from the last minute rather than the whole run once updates are frequent", () => {
    const start = Date.parse("2026-09-30T12:00:00Z");
    const samples = [
      { at: start, step: 0, done: 0, fraction: 0 },
      { at: start + 120000, step: 0, done: 12, fraction: 0.06 },
      { at: start + 180000, step: 0, done: 72, fraction: 0.36 },
    ];
    expect(analysisPace(samples, start + 180000).perMinute).toBe(60);
  });

  it("rounds remaining time up so the label never promises less than the estimate", () => {
    expect(remainingLabel(61)).toBe("~2m");
    expect(remainingLabel(30)).toBe("<1m");
  });
});

describe("Stage durations", () => {
  const start = Date.parse("2026-09-30T12:00:00Z");
  const createdAt = "2026-09-30T12:00:00Z";

  it("times finished stages from the transitions it saw and the active stage up to now", () => {
    const samples = [
      { at: start + 5000, step: 0, done: 10, fraction: 0.1 },
      { at: start + 124000, step: 1, done: 0, fraction: 0.6 },
    ];
    expect(stageDurations(samples, createdAt, start + 145000)).toEqual([124, 21, null]);
  });

  it("does not guess when a stage started before the page was opened", () => {
    const samples = [{ at: start + 90000, step: 1, done: 2, fraction: 0.7 }];
    expect(stageDurations(samples, createdAt, start + 100000)).toEqual([null, null, null]);
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
  it("turns an issue brief into a pasteable markdown document", () => {
    expect(
      briefMarkdown("PRs were never opened", {
        problem: "The workspace was not a Git repository.",
        user_goal: "Open a PR fixing a typo",
        what_happened: 'Git returned "fatal: not a git repository"',
        test_cases: [
          { input: "Fix the typo and open a PR", expected: "A PR URL is returned" },
          { input: "Rename greet", expected: "The rename is committed" },
        ],
      }),
    ).toBe(
      [
        "# PRs were never opened",
        "## Problem\nThe workspace was not a Git repository.",
        "## User goal\nOpen a PR fixing a typo",
        '## What happened\nGit returned "fatal: not a git repository"',
        "## Test cases\n1. **Input:** Fix the typo and open a PR  \n   **Expect:** A PR URL is returned\n" +
          "2. **Input:** Rename greet  \n   **Expect:** The rename is committed",
      ].join("\n\n"),
    );
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
