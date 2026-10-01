import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { apiClient } from "@/components/networking";
import EnginePage from "../page";
import { EngineView } from "./EngineView";
import { nextCheckStatus, type Engine, type EngineList, type Finding } from "./engineData";

const useAuthorizedMock = vi.hoisted(() =>
  vi.fn<() => { accessToken: string; userRole: string; isViewOnly: boolean }>(),
);

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: useAuthorizedMock }));
vi.mock("@/components/networking", () => ({ apiClient: { get: vi.fn(), post: vi.fn() }, proxyBaseUrl: "" }));

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
    testQueryClient.clear();
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

describe("Lens getting started", () => {
  const empty: EngineList = { engines: [], workers: [], tracing_enabled: false };
  const worker: EngineList["workers"][number] = {
    id: "analyzer",
    name: "Lens analyzer",
    last_seen: new Date().toISOString(),
    revoked: false,
    scope: engine.scope,
  };

  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(apiClient.get).mockReset();
    vi.mocked(apiClient.post).mockReset();
    vi.mocked(apiClient.get).mockImplementation(async (path) => (path === "/engine" ? empty : { data: [] }));
    vi.mocked(apiClient.post).mockResolvedValue({ eligible: 0, executions: [] });
    useAuthorizedMock.mockReset().mockReturnValue({ accessToken: "test", userRole: "Admin", isViewOnly: false });
  });

  it("explains the prerequisites and opens the existing analyzer and lens setup dialogs", async () => {
    const user = userEvent.setup();
    renderWithProviders(<EnginePage />);
    const guide = within(await screen.findByRole("region", { name: "Turn agent activity into answers" }));
    expect(guide.getByRole("heading", { name: "Record activity" })).toBeVisible();
    expect(guide.getByText("Tracing not configured")).toBeVisible();
    expect(guide.getByText("Analyzer not connected")).toBeVisible();
    expect(guide.getByText("Where do agents get stuck or repeat the same work?")).toBeVisible();
    expect(guide.getByRole("link", { name: "Open logs" })).toHaveAttribute("href", "/ui/logs");

    await user.click(guide.getByRole("button", { name: "Connect analyzer" }));
    const analyzer = within(screen.getByRole("dialog", { name: "Set up Lens analysis" }));
    expect(analyzer.getByRole("textbox", { name: "Your LiteLLM deployment URL" })).toBeVisible();
    expect(analyzer.getByRole("button", { name: "Generate setup command" })).toBeEnabled();
    await user.click(analyzer.getByRole("button", { name: "Close" }));

    await user.click(guide.getByRole("button", { name: "Create a lens" }));
    const setup = within(screen.getByRole("dialog", { name: "Set up a lens" }));
    expect(setup.getByRole("textbox", { name: "Name" })).toBeVisible();
    expect(setup.getByRole("combobox", { name: "Activity type" })).toBeVisible();
    expect(await setup.findByText("0 matching runs")).toBeVisible();
  });

  it.each([
    { name: "a connected analyzer", revoked: false, age: 0, connected: true },
    { name: "an offline analyzer", revoked: false, age: 180000, connected: false },
    { name: "a revoked analyzer", revoked: true, age: 0, connected: false },
  ])("reports configured tracing and the current status of $name", async ({ revoked, age, connected }) => {
    vi.mocked(apiClient.get).mockImplementation(async (path) =>
      path === "/engine"
        ? {
            ...empty,
            tracing_enabled: true,
            workers: [{ ...worker, revoked, last_seen: new Date(Date.now() - age).toISOString() }],
          }
        : { data: [] },
    );
    renderWithProviders(<EngineView accessToken="test" />);
    expect(await screen.findByText("Tracing configured")).toBeVisible();
    expect(screen.getByText(connected ? "Analyzer connected" : "Analyzer not connected")).toBeVisible();
    expect(screen.getByRole("button", { name: connected ? "Manage analyzer" : "Connect analyzer" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Create a lens" })).toBeEnabled();
  });

  it.each([
    { userRole: "Admin", isViewOnly: true },
    { userRole: "Internal User", isViewOnly: false },
  ])(
    "gives $userRole with isViewOnly=$isViewOnly an admin handoff without write controls",
    async ({ userRole, isViewOnly }) => {
      useAuthorizedMock.mockReturnValue({ accessToken: "test", userRole, isViewOnly });
      renderWithProviders(<EnginePage />);
      expect(await screen.findByText(/Ask a proxy admin to connect an analyzer and create a lens/)).toBeVisible();
      expect(screen.getByText("No lenses available yet")).toBeVisible();
      expect(screen.getByRole("link", { name: "Open logs" })).toBeVisible();
      expect(screen.queryByRole("button")).not.toBeInTheDocument();
    },
  );

  it("waits for a successful response before showing the getting started guide", () => {
    vi.mocked(apiClient.get).mockImplementation((path) =>
      path === "/engine" ? new Promise(() => {}) : Promise.resolve({ data: [] }),
    );
    renderWithProviders(<EngineView accessToken="test" />);
    expect(screen.getByRole("status")).toHaveTextContent("Loading lenses…");
    expect(screen.queryByRole("region", { name: "Turn agent activity into answers" })).not.toBeInTheDocument();
  });

  it("shows a retryable error instead of treating a failed request as an empty list", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.get).mockImplementation(async (path) => {
      if (path === "/engine") throw new Error("Could not load lenses");
      return { data: [] };
    });
    renderWithProviders(<EngineView accessToken="test" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load lenses");
    expect(screen.queryByRole("region", { name: "Turn agent activity into answers" })).not.toBeInTheDocument();
    vi.mocked(apiClient.get).mockImplementation(async (path) => (path === "/engine" ? empty : { data: [] }));
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("region", { name: "Turn agent activity into answers" })).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

it("shows the actual next schedule and avoids a stale countdown during active scans", () => {
  const now = Date.parse("2026-09-30T10:00:00Z");
  const monitoring = {
    ...engine,
    settings: { ...engine.settings, enabled: true },
    next_run_at: "2026-09-30T10:12:00Z",
  };
  expect(nextCheckStatus(monitoring, now)).toContain("in 12 minutes");
  expect(nextCheckStatus(monitoring, now + 12 * 60000)).toBe("Due now · waiting for an analyzer");
  expect(nextCheckStatus({ ...monitoring, jobs: [{ ...engine.jobs[0], status: "running" }] }, now)).toBe(
    "Next check scheduled after this scan finishes",
  );
  expect(nextCheckStatus({ ...monitoring, jobs: [{ ...engine.jobs[0], status: "queued" }] }, now)).toBe(
    "Waiting for an analyzer",
  );
  expect(nextCheckStatus(engine, now)).toBeNull();
});
