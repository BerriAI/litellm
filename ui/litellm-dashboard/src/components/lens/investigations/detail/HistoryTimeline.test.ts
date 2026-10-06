import { describe, expect, it } from "vitest";

import type { Job } from "../../model/types";
import { checkColumns } from "./HistoryTimeline";

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
  failed_tasks: 0,
};

const job: Job = {
  assessments: [],
  steps: [],
  reviews: [],
  reviewed: 0,
  reading: [],
  activities: [],
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

const assessment = (id: string, issues: string[]): Job["assessments"][number] => ({
  execution_id: id,
  cannot_assess: false,
  issue_checks: issues,
  pattern_checks: [],
});

const check = (id: string, status: Job["status"], screened: number, assessments: Job["assessments"]): Job => ({
  ...job,
  id,
  status,
  coverage: { ...coverage, screened },
  assessments,
});

describe("checkColumns", () => {
  const newest = check("newest", "completed", 3, [
    assessment("a", ["failures"]),
    assessment("b", []),
    assessment("c", ["failures", "other"]),
  ]);
  const oldest = check("oldest", "failed", 7, []);

  it("pads the left and puts the newest check on the right", () => {
    const columns = checkColumns([newest, oldest], 4);
    expect(columns.map((c) => c.job?.id ?? null)).toEqual([null, null, "oldest", "newest"]);
    expect(columns[0]).toMatchObject({ total: 0, failed: 0 });
  });

  it("counts reviewed traces and the traces with at least one issue", () => {
    expect(checkColumns([newest], 1)[0]).toMatchObject({ total: 3, failed: 2, opacity: 1 });
  });

  it("fades checks that did not complete", () => {
    const [failed] = checkColumns([oldest], 1);
    const [cancelled] = checkColumns([{ ...oldest, status: "cancelled" }], 1);
    expect(failed.opacity).toBeLessThan(1);
    expect(cancelled.opacity).toBe(failed.opacity);
  });

  it("keeps only the most recent checks that fit", () => {
    expect(checkColumns([newest, oldest], 1).map((c) => c.job?.id)).toEqual(["newest"]);
  });
});
