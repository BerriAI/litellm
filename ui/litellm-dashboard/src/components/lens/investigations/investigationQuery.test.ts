import { describe, expect, it } from "vitest";

import { fieldValues } from "@/components/shared/search/evaluate";
import type { Job, Lens } from "../model/types";
import { filterInvestigations, INVESTIGATION_INDEX } from "./investigationQuery";

function makeLens(
  id: string,
  settings: Partial<Lens["settings"]>,
  jobs: readonly (Pick<Job, "status"> & Partial<Pick<Job, "error">>)[] = [],
): Lens {
  return {
    version: 0,
    spent: 0,
    id,
    scope: { all_teams: true, api_key_hash: "", team_id: "" },
    settings: {
      context: "",
      source: "traces",
      lookback_hours: 24,
      service: "",
      agent_name: "",
      filters: [],
      interval_minutes: 15,
      sample_size: 100,
      sample_percent: 100,
      concurrency: 8,
      team_id: "",
      execution_ids: [],
      monthly_budget: 20,
      name: id,
      model: "analysis",
      enabled: true,
      checks: [],
      ...settings,
    },
    revision: 1,
    created_at: "2026-09-30T10:00:00Z",
    next_run_at: "2026-09-30T10:00:00Z",
    budget_month: "2026-09",
    findings: [],
    jobs: jobs as Job[],
  };
}

const lenses = [
  makeLens("Refund audit", { agent_name: "billing-agent" }, [{ status: "failed" }, { status: "completed" }]),
  makeLens("Release reviews", { service: "reviewer", enabled: false }, [{ status: "completed" }]),
  makeLens("Lead scoring", { filters: [{ key: "team", value: "sales" }] }),
];

const names = (query: string) => filterInvestigations(lenses, query).map((lens) => lens.settings.name);

describe("filterInvestigations", () => {
  it("matches free text against the name and the scope", () => {
    expect(names("REFUND")).toEqual(["Refund audit"]);
    expect(names("sales")).toEqual(["Lead scoring"]);
    expect(names("reviewer")).toEqual(["Release reviews"]);
  });

  it("filters by the latest run's status, treating no runs as never", () => {
    expect(names("status:failed")).toEqual(["Refund audit"]);
    expect(names("status:completed")).toEqual(["Release reviews"]);
    expect(names("status:never")).toEqual(["Lead scoring"]);
  });

  it("filters by schedule and by agent, falling back to the service", () => {
    expect(names("schedule:paused")).toEqual(["Release reviews"]);
    expect(names("-schedule:paused")).toEqual(["Refund audit", "Lead scoring"]);
    expect(names("agent:billing-agent")).toEqual(["Refund audit"]);
    expect(names("agent:reviewer")).toEqual(["Release reviews"]);
  });

  it("finds partial runs separately from completed and failed runs", () => {
    const partial = makeLens("Partial review", {}, [{ status: "completed", error: "One task failed" }]);
    expect(filterInvestigations([...lenses, partial], "status:partial")).toEqual([partial]);
    expect(fieldValues(INVESTIGATION_INDEX, [partial], "status")).toEqual(["partial"]);
  });
});

describe("INVESTIGATION_INDEX values", () => {
  it("offers only the agents and statuses present", () => {
    expect(fieldValues(INVESTIGATION_INDEX, lenses, "agent")).toEqual(["billing-agent", "reviewer"]);
    expect(fieldValues(INVESTIGATION_INDEX, lenses, "status")).toEqual(["completed", "failed", "never"]);
  });
});
