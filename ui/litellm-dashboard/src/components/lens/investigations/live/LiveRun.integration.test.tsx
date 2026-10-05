import { fireEvent, screen, within } from "@testing-library/react";
import { beforeEach, expect, it } from "vitest";

import { renderWithLens, stubGateway } from "@/../tests/lens-test-utils";
import { testQueryClient } from "@/../tests/test-utils";
import { createLensDemoData } from "../../data/demo/fixtures";
import type { Activity, Job, Review } from "../../model/types";
import { LiveRun } from "./LiveRun";

beforeEach(() => {
  testQueryClient.clear();
  window.localStorage.clear();
  stubGateway().get.mockResolvedValue({});
});

function job(overrides: Partial<Job> = {}): Job {
  return {
    ...createLensDemoData().lenses[0].jobs[0],
    status: "running",
    stage: "Checking original evidence",
    activities: [],
    reading: [],
    reviewed: 450,
    ...overrides,
  };
}

function review(): Review {
  return {
    execution_id: "execution-1",
    trace_id: "trace-1",
    agent: "research-agent",
    name: "research",
    spans: [],
    reasoning: "The grep call failed before the terminal recovered.",
    verdicts: [{ check_id: "tools", kind: "issue", summary: "Grep argument mismatch" }],
    cannot_assess: false,
    model: "analysis",
    duration_ms: 100,
    at: "2026-10-03T16:00:00Z",
    tool_calls: [
      { name: "read", calls: 2 },
      { name: "python", calls: 1 },
    ],
  };
}

function activity(overrides: Partial<Activity> = {}): Activity {
  return {
    id: "candidate-1",
    phase: "investigate",
    label: "Grep compatibility",
    execution_ids: ["execution-1", "execution-2"],
    started_at: "2026-10-03T16:00:00Z",
    operations: ["python"],
    tool_calls: [
      { name: "read", calls: 2 },
      { name: "search", calls: 1 },
    ],
    finished: false,
    ...overrides,
  };
}

it("shows real candidate and grouping activity, durable tool counts, and preliminary review scope", async () => {
  const reviewed = review();
  const grouping: Partial<Activity> = {
    id: "group-2",
    phase: "group",
    label: "Group 2",
    execution_ids: [],
    operations: ["model"],
    tool_calls: [],
  };
  const active = job({ activities: [activity(), activity(grouping)] });
  const view = (current: Job) => <LiveRun job={current} reviews={[reviewed]} name="Tool quality" />;
  const { rerender } = renderWithLens(view(active));
  const strip = within(screen.getByRole("region", { name: "Live trace results" }));
  expect(strip.getByRole("status")).toHaveTextContent("Checking original evidence");
  fireEvent.click(strip.getByRole("button", { name: "View run" }));
  const drawer = within(await screen.findByRole("dialog"));
  const work = within(drawer.getByRole("region", { name: "Current work" }));
  expect(work.getByText("2 active")).toBeVisible();
  expect(work.getByText("Investigating candidate · 2 traces")).toBeVisible();
  expect(work.getByText("Running Python")).toBeVisible();
  expect(work.getByText("Grouping observations")).toBeVisible();
  expect(work.getByText("Tool calls: Read × 2 · Search × 1")).toBeVisible();
  expect(drawer.queryByRole("region", { name: "Now reading" })).not.toBeInTheDocument();
  expect(drawer.getByRole("heading", { name: "Preliminary observations" })).toBeVisible();
  expect(drawer.getByText("1 displayed of 450 reviewed")).toBeVisible();
  expect(drawer.getByText(/Final findings are shown in Findings/)).toBeVisible();

  const completed: Job = { ...active, status: "completed", stage: "Completed", findings: [], activities: [] };
  rerender(view(completed));
  expect(screen.queryByRole("region", { name: "Current work" })).not.toBeInTheDocument();
  expect(strip.getByText(/0 issues found/)).toBeVisible();
  const traces = within(drawer.getByRole("list", { name: "Reviewed traces" }));
  fireEvent.click(traces.getByRole("button", { name: /research-agent/ }));
  expect(traces.getByText("Tool calls: Read × 2 · Python × 1")).toBeVisible();
  expect(drawer.getByRole("heading", { name: "Preliminary observations" })).toBeVisible();
});

it("keeps older workers' reading lanes but stops calling grouping work reading", async () => {
  const active = job({
    stage: "Reading executions",
    reviewed: 0,
    reading: [
      { execution_id: "old", trace_id: "trace-old", agent: "older-worker", started_at: "2026-10-03T16:00:00Z" },
    ],
  });
  const view = (current: Job) => <LiveRun job={current} reviews={[]} name="Compatibility" />;
  const { rerender } = renderWithLens(view(active));
  fireEvent.click(screen.getByRole("button", { name: "View run" }));
  const drawer = within(await screen.findByRole("dialog"));
  expect(within(drawer.getByRole("region", { name: "Now reading" })).getByText("older-worker")).toBeVisible();
  rerender(view({ ...active, stage: "Grouping observations", reading: [] }));
  expect(drawer.queryByRole("region", { name: "Now reading" })).not.toBeInTheDocument();
  expect(drawer.getByRole("status")).toHaveTextContent("Grouping observations");
});
