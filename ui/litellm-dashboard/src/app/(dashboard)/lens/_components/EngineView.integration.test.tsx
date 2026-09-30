import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import { apiClient } from "@/components/networking";
import { EngineView } from "./EngineView";
import type { Engine, Finding } from "./engineData";

vi.mock("@/components/networking", () => ({ apiClient: { get: vi.fn() } }));

const executionId = btoa(JSON.stringify(["traces", "", "trace-42"]));
const pattern: Finding = {
  reason: "",
  suggestion: "",
  id: "pattern",
  check_id: "check",
  title: "Agents ignored misleading document instructions",
  description: "Two agents completed their assigned work despite misleading text in a document.",
  kind: "pattern",
  priority: "low",
  status: "open",
  revision: 1,
  first_seen: "2026-09-30T10:00:00Z",
  last_seen: "2026-09-30T10:00:00Z",
  limitation: "This does not prove every attack will be resisted.",
  occurrences: [executionId],
  evidence: [{ execution_id: executionId, span_id: "step-1", quote: "Ignore the review instructions" }],
};
const issue: Finding = {
  ...pattern,
  id: "issue",
  title: "Review used the wrong defect rate",
  kind: "issue",
  priority: "high",
};
const engine: Engine = {
  version: 0,
  spent: 0,
  id: "lens",
  scope: { all_teams: true, api_key_hash: "", team_id: "" },
  settings: {
    context: "",
    source: "traces",
    lookback_hours: 24,
    service: "",
    filters: [],
    interval_minutes: 15,
    sample_size: 100,
    monthly_budget: 20,
    name: "Release reviews",
    model: "analysis",
    enabled: false,
    checks: [{ enabled: true, id: "check", instruction: "Find unsupported decisions" }],
  },
  revision: 1,
  created_at: "2026-09-30T10:00:00Z",
  next_run_at: "2026-09-30T10:00:00Z",
  budget_month: "2026-09",
  findings: [pattern, issue],
  jobs: [
    {
      id: "scan",
      attempts: 0,
      error: "",
      cost: 0,
      coverage: {
        eligible: 0,
        selected: 0,
        screened: 0,
        investigated: 0,
        grouping_batches: 0,
        grouped_batches: 0,
        candidates: 0,
        partial: 0,
        unassessable: 0,
      },
      status: "completed",
      stage: "Complete",
      created_at: "2026-09-30T10:00:00Z",
      start: "2026-09-29T10:00:00Z",
      end: "2026-09-30T10:00:00Z",
      settings: {
        context: "",
        source: "traces",
        lookback_hours: 24,
        service: "",
        filters: [],
        interval_minutes: 15,
        sample_size: 100,
        monthly_budget: 20,
        enabled: false,
        name: "Release reviews",
        model: "analysis",
        checks: [{ enabled: true, id: "check", instruction: "Find unsupported decisions" }],
      },
      revision: 1,
      sample: {
        eligible: 1,
        executions: [
          {
            id: executionId,
            trace_ref: "",
            metadata: [],
            root_seen: true,
            service: "",
            source: "traces",
            trace_id: "trace-42",
            team_id: "",
            name: "Release-42",
            start_time: "2026-09-30 10:00:00.000",
            span_count: 12,
          },
        ],
      },
    },
  ],
};

describe("Lens findings and runs", () => {
  beforeEach(() => {
    vi.mocked(apiClient.get).mockReset();
    vi.mocked(apiClient.get).mockImplementation(async (path) =>
      path === "/engine" ? { engines: [engine], workers: [], tracing_enabled: true } : { data: [] },
    );
  });

  it("separates patterns from issues and reveals original evidence only when requested", async () => {
    const user = userEvent.setup();
    renderWithProviders(<EngineView accessToken="test" readOnly />);
    expect(await screen.findByText("Review used the wrong defect rate")).toBeInTheDocument();
    expect(screen.queryByText(pattern.title)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Patterns (1)" }));
    await user.click(screen.getByRole("button", { name: new RegExp(pattern.title) }));
    const detail = within(screen.getByRole("dialog", { name: pattern.title }));
    expect(detail.getByText(pattern.description)).toBeVisible();
    expect(detail.getByText(pattern.limitation ?? "")).not.toBeVisible();
    expect(detail.getByText("Ignore the review instructions")).not.toBeVisible();
    await user.click(detail.getByText("Release-42"));
    expect(detail.getByText("Ignore the review instructions")).toBeVisible();
    expect(screen.getByRole("button", { name: "Open original step" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Mark resolved" })).not.toBeInTheDocument();
  });

  it("shows the actual frozen run selection in the Runs tab", async () => {
    const user = userEvent.setup();
    renderWithProviders(<EngineView accessToken="test" readOnly />);
    await user.click(await screen.findByRole("tab", { name: "Runs" }));
    expect(screen.getByText("Release-42")).toBeInTheDocument();
    expect(screen.getByText("trace-42")).toBeInTheDocument();
    expect(screen.getByText(/1 selected from 1 matches/)).toBeInTheDocument();
  });
});
