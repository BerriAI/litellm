import { activeJob, investigationActivity, listPollInterval, nextCheckStatus, workerConnected } from "./status";
import { describe, expect, it } from "vitest";

import type { Job, Lens, LensList } from "./types";

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
  steps: [],
  trigger: "schedule",
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

const lens: Lens = {
  version: 0,
  spent: 0,
  id: "lens",
  scope: { all_teams: true, api_key_hash: "", team_id: "" },
  settings: { ...job.settings, enabled: false },
  revision: 1,
  created_at: job.created_at,
  next_run_at: job.created_at,
  budget_month: "2026-09",
  findings: [],
  jobs: [{ ...job, status: "completed" }],
};

it("shows the actual next schedule and avoids a stale countdown during active scans", () => {
  const now = Date.parse("2026-09-30T10:00:00Z");
  const monitoring = {
    ...lens,
    settings: { ...lens.settings, enabled: true },
    next_run_at: "2026-09-30T10:12:00Z",
  };
  expect(nextCheckStatus(monitoring, now)).toContain("in 12 minutes");
  expect(nextCheckStatus(monitoring, now + 12 * 60000)).toBe("Due now · waiting for an analyzer");
  expect(nextCheckStatus({ ...monitoring, jobs: [{ ...lens.jobs[0], status: "running" }] }, now)).toBe(
    "Next check scheduled after this scan finishes",
  );
  expect(nextCheckStatus({ ...monitoring, jobs: [{ ...lens.jobs[0], status: "queued" }] }, now)).toBe(
    "Waiting for an analyzer",
  );
  expect(nextCheckStatus(lens, now)).toBeNull();
});

it("summarizes activity across investigations, preferring a running scan over a queued one", () => {
  const withStatus = (status: Job["status"]): Lens => ({ ...lens, jobs: [{ ...job, status }] });
  expect(investigationActivity([])).toBe("idle");
  expect(investigationActivity([lens, withStatus("failed")])).toBe("idle");
  expect(investigationActivity([lens, withStatus("queued")])).toBe("queued");
  expect(investigationActivity([withStatus("queued"), withStatus("running")])).toBe("running");
  expect(investigationActivity([{ ...lens, jobs: [lens.jobs[0], { ...job, status: "running" }] }])).toBe("running");
});

describe("List polling cadence", () => {
  const now = Date.parse("2026-10-01T12:00:00Z");
  const connectedWorker: LensList["workers"][number] = {
    id: "worker",
    name: "Worker",
    revoked: false,
    analysis_key_id: "key",
    last_seen: "2026-10-01T11:59:59Z",
    scope: { all_teams: true, api_key_hash: "", team_id: "" },
  };
  const staleWorker = { ...connectedWorker, last_seen: "2026-10-01T11:00:00Z" };
  const list = (workers: LensList["workers"], lenses: Lens[] = [lens]): LensList => ({
    lenses,
    workers,
    tracing_enabled: true,
  });

  it("polls slowly when nothing is running and Settings is closed", () => {
    expect(listPollInterval(list([staleWorker]), false, now)).toBe(10000);
    expect(listPollInterval(undefined, false, now)).toBe(10000);
  });

  it("polls fast while Settings waits for a worker and slows down once one connects", () => {
    expect(listPollInterval(undefined, true, now)).toBe(2000);
    expect(listPollInterval(list([staleWorker]), true, now)).toBe(2000);
    expect(listPollInterval(list([connectedWorker]), true, now)).toBe(10000);
  });

  it("speeds up as soon as any investigation is queued or running, whatever tab is open", () => {
    const queued: Lens = { ...lens, jobs: [{ ...job, status: "queued" }] };
    const running: Lens = { ...lens, jobs: [lens.jobs[0], { ...job, status: "running" }] };
    expect(listPollInterval(list([connectedWorker], [lens, queued]), false, now)).toBe(2000);
    expect(listPollInterval(list([connectedWorker], [running]), false, now)).toBe(2000);
    expect(listPollInterval(list([connectedWorker], [lens]), false, now)).toBe(10000);
  });
});

describe("activeJob", () => {
  it("picks the queued or running job and ignores finished ones", () => {
    const done = { ...job, id: "done", status: "completed" as const };
    expect(activeJob([done, job])).toBe(job);
    expect(activeJob([done, { ...job, status: "queued" }])?.status).toBe("queued");
    expect(activeJob([done])).toBeUndefined();
  });
});
