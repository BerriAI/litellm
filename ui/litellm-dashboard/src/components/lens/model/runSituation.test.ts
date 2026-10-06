import { describe, expect, it } from "vitest";
import { NEXT_ACTION, runSituation, type SituationInput } from "./runSituation";
import type { Finding, Job, Lens } from "./types";

const settings: Lens["settings"] = {
  context: "",
  source: "traces",
  lookback_hours: 24,
  service: "",
  agent_name: "",
  filters: [],
  interval_minutes: 60,
  sample_size: 100,
  sample_percent: 100,
  concurrency: 8,
  team_id: "",
  execution_ids: [],
  monthly_budget: 20,
  name: "Release reviews",
  model: "analysis",
  enabled: false,
  checks: [],
};
const month = new Date().toISOString().slice(0, 7);
const lens: Lens = {
  version: 0,
  spent: 1,
  id: "lens",
  scope: { all_teams: true, api_key_hash: "", team_id: "" },
  settings,
  revision: 1,
  created_at: "2026-09-30T10:00:00Z",
  next_run_at: "2026-09-30T10:00:00Z",
  budget_month: month,
  findings: [],
  jobs: [],
};
const job: Job = {
  id: "run",
  assessments: [],
  steps: [],
  reviews: [],
  reviewed: 0,
  reading: [],
  activities: [],
  trigger: "manual",
  attempts: 1,
  error: "",
  cost: 0,
  coverage: {
    eligible: 10,
    selected: 10,
    screened: 10,
    investigated: 0,
    inconclusive: 0,
    grouping_batches: 0,
    grouped_batches: 0,
    candidates: 0,
    partial: 0,
    unassessable: 0,
    failed_tasks: 0,
  },
  status: "completed",
  stage: "Complete",
  created_at: "2026-09-30T10:00:00Z",
  start: "2026-09-29T10:00:00Z",
  end: "2026-09-30T10:00:00Z",
  settings,
  revision: 1,
};
const finding = (kind: Finding["kind"], status: Finding["status"]): Finding => ({
  id: `${kind}-${status}`,
  check_id: "check",
  title: "",
  description: "",
  reason: "",
  suggestion: "",
  limitation: "",
  kind,
  priority: "high",
  status,
  revision: 1,
  first_seen: job.created_at,
  last_seen: job.created_at,
  occurrences: [],
  evidence: [],
});

const input = (overrides: Partial<SituationInput> & { jobPatch?: Partial<Job> }): SituationInput => {
  const { jobPatch, ...rest } = overrides;
  return { lens, job: { ...job, ...jobPatch }, findings: [], connected: true, ...rest };
};
const failed = { status: "failed", error: "Model request failed" } as const;
const watching = { ...lens, settings: { ...settings, enabled: true } };

describe("runSituation", () => {
  it.each([
    ["never", input({ job: undefined }), "run"],
    ["queued", input({ jobPatch: { status: "queued" } }), "stop"],
    ["running", input({ jobPatch: { status: "running" } }), "stop"],
    ["budget", input({ jobPatch: failed, lens: { ...lens, spent: 20 } }), "raiseBudget"],
    ["budget", input({ jobPatch: { ...failed, error: "Monthly lens budget reached" } }), "raiseBudget"],
    ["offline", input({ jobPatch: failed, connected: false }), "connectWorker"],
    ["failed", input({ jobPatch: failed }), "retry"],
    ["failed", input({ jobPatch: { ...failed, error: "Could not reserve analysis budget" } }), "retry"],
    ["cancelled", input({ jobPatch: { status: "cancelled" } }), "retry"],
    ["partial", input({ jobPatch: { error: "Result validation failed after 3 retries" } }), "retry"],
    ["unknown", input({ findings: null }), null],
    ["issues", input({ lens: watching, findings: [finding("issue", "open")] }), "reviewIssues"],
    ["watching", input({ lens: watching, findings: [finding("issue", "resolved")] }), null],
    ["clean", input({ findings: [finding("pattern", "open")] }), "monitor"],
  ] as const)("reads %s and offers %s", (expected, situationInput, action) => {
    const situation = runSituation(situationInput);
    expect(situation).toBe(expected);
    expect(NEXT_ACTION[situation]).toBe(action);
  });

  it("puts a running job ahead of a spent budget, so Stop stays reachable", () => {
    expect(runSituation(input({ jobPatch: { status: "running" }, lens: { ...lens, spent: 99 } }))).toBe("running");
  });

  it("ignores spend from an earlier month", () => {
    expect(runSituation(input({ jobPatch: failed, lens: { ...lens, spent: 99, budget_month: "1999-01" } }))).toBe(
      "failed",
    );
  });
});
