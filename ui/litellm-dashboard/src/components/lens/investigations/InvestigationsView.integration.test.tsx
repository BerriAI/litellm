import { act, screen, within, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders as renderProviders, testQueryClient } from "@/../tests/test-utils";
import { ApiError } from "@/lib/http/client";
import { apiClient } from "@/components/networking";
import { lensKeys } from "../api/queries";
import { InvestigationsView } from "./InvestigationsView";
import { briefMarkdown } from "../model/findings";
import { runTime } from "../model/format";
import { type Lens, type Finding } from "../model/types";

function renderWithProviders(ui: React.ReactElement, options?: Parameters<typeof renderProviders>[1]) {
  return renderProviders(ui, { searchParams: window.location.search, ...options });
}

vi.mock("@/components/networking", () => ({
  apiClient: { get: vi.fn(), post: vi.fn(), request: vi.fn() },
  proxyBaseUrl: "",
}));

beforeEach(() => {
  window.history.replaceState({}, "", "/lens/?lens=lens");
  vi.mocked(apiClient.post).mockReset();
  vi.mocked(apiClient.post).mockResolvedValue({ eligible: 0, selected: 0, executions: [] });
});

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
  evidence: [
    { execution_id: executionId, span_id: "step-1", quote: "Ignore the review instructions", role: "support" },
  ],
};
const issue: Finding = {
  ...pattern,
  id: "issue",
  title: "Review used the wrong defect rate",
  kind: "issue",
  priority: "high",
};
const lens: Lens = {
  version: 0,
  spent: 0,
  id: "lens",
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
      findings: [pattern, issue],
      assessments: [],
      attempts: 0,
      error: "",
      cost: 0,
      coverage: {
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
        agent_name: "",
        filters: [],
        interval_minutes: 15,
        sample_size: 100,
        sample_percent: 100,
        concurrency: 8,
        team_id: "",
        execution_ids: [],
        monthly_budget: 20,
        enabled: false,
        name: "Release reviews",
        model: "analysis",
        checks: [{ enabled: true, id: "check", instruction: "Find unsupported decisions" }],
      },
      revision: 1,
      sample: {
        eligible: 1,
        selected: 1,
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
    vi.mocked(apiClient.get).mockImplementation(async (path) => {
      if (path === "/lens") return { lenses: [lens], workers: [], tracing_enabled: true };
      if (path === "/lens/lens/runs") return lens.jobs;
      return { data: [] };
    });
  });

  it("separates patterns from issues and reveals original evidence only when requested", async () => {
    const user = userEvent.setup();
    renderWithProviders(<InvestigationsView accessToken="test" readOnly />);
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

  const brief = {
    problem: "The workspace was not a Git repository, so the agent could not commit.",
    user_goal: "Open a pull request fixing a typo",
    what_happened: 'Git returned "fatal: not a git repository"',
    test_cases: [{ input: "Fix the typo and open a PR", expected: "A PR URL is returned" }],
  };

  async function openIssue(finding: Finding) {
    testQueryClient.clear();
    const jobs = lens.jobs.map((job) => ({ ...job, findings: [finding] }));
    vi.mocked(apiClient.get).mockImplementation(async (path) => {
      if (path === "/lens")
        return { lenses: [{ ...lens, findings: [finding], jobs }], workers: [], tracing_enabled: true };
      if (path === "/lens/lens/runs") return jobs;
      return { data: [] };
    });
    const user = userEvent.setup();
    renderWithProviders(<InvestigationsView accessToken="test" readOnly />);
    await user.click(await screen.findByRole("button", { name: new RegExp(finding.title) }));
    return { user, detail: within(screen.getByRole("dialog", { name: finding.title })) };
  }

  it.each(["Claude Code", "Codex"])("renders the issue brief and copies its markdown for %s", async (agent) => {
    const { user, detail } = await openIssue({ ...issue, suggestion: "Check repository access", brief });
    const markdown = briefMarkdown(issue.title, brief);
    expect(detail.getByRole("heading", { level: 1, name: issue.title })).toBeVisible();
    for (const section of ["Problem", "User goal", "What happened", "Test cases"]) {
      expect(detail.getByRole("heading", { level: 2, name: section })).toBeVisible();
    }
    expect(detail.getByText(brief.problem)).toBeVisible();
    expect(detail.getByRole("listitem")).toHaveTextContent(
      `Input: ${brief.test_cases[0].input} Expect: ${brief.test_cases[0].expected}`,
    );
    expect(detail.queryByText("## Problem", { exact: false })).not.toBeInTheDocument();
    expect(detail.queryByText("Check repository access")).not.toBeInTheDocument();
    await user.click(detail.getByRole("button", { name: `Copy for ${agent}` }));
    expect(await navigator.clipboard.readText()).toBe(markdown);
  });

  it("keeps the summary and suggestion for findings recorded before briefs existed", async () => {
    const { detail } = await openIssue({ ...issue, suggestion: "Check repository access" });
    expect(detail.getByText(issue.description)).toBeVisible();
    expect(detail.getByText("Check repository access")).toBeVisible();
    expect(detail.queryByRole("button", { name: "Copy for Claude Code" })).not.toBeInTheDocument();
  });

  it("shows the actual frozen run selection in the Runs tab", async () => {
    const user = userEvent.setup();
    renderWithProviders(<InvestigationsView accessToken="test" readOnly />);
    await user.click(await screen.findByRole("tab", { name: "Traces" }));
    expect(screen.getByText("Release-42")).toBeInTheDocument();
    expect(screen.getByTitle("trace-42")).toHaveTextContent("Release-42");
    expect(screen.getByText(/1 selected from 1 matching runs/)).toBeInTheDocument();
  });
});

it("runs saved settings immediately without opening setup", async () => {
  testQueryClient.clear();
  vi.mocked(apiClient.get).mockImplementation(async (path) => {
    if (path === "/lens")
      return {
        lenses: [lens],
        tracing_enabled: true,
        workers: [
          {
            id: "worker",
            name: "Worker",
            revoked: false,
            analysis_key_id: "a".repeat(64),
            scope: lens.scope,
            last_seen: new Date().toISOString(),
          },
        ],
      };
    if (path === "/lens/lens/runs") return lens.jobs;
    if (path === "/lens/activity/available") return { traces: true, requests: false };
    return { data: [] };
  });
  vi.mocked(apiClient.post).mockResolvedValue(lens);
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView accessToken="test" />);
  await user.click(await screen.findByRole("button", { name: "Run now" }));
  expect(apiClient.post).toHaveBeenCalledWith("/lens/lens/runs", { accessToken: "test", body: {} });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

it("offers the interactive demo without starting an investigation", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  vi.mocked(apiClient.get).mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [], workers: [], tracing_enabled: false };
    return { traces: false, requests: false };
  });
  const onDemo = vi.fn();
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView accessToken="test" onDemo={onDemo} />);
  await user.click(await screen.findByRole("button", { name: "Preview sample" }));
  expect(onDemo).toHaveBeenCalledOnce();
  expect(apiClient.post).not.toHaveBeenCalled();
});

it("guides a first-time administrator into worker connection and lens setup", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  vi.mocked(apiClient.get).mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [], workers: [], tracing_enabled: true };
    if (path === "/lens/agents") return [];
    return { traces: true, requests: false, data: [] };
  });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView accessToken="test" onDemo={vi.fn()} />);
  const guide = within(await screen.findByRole("region", { name: "Find what needs attention" }));
  expect(apiClient.get).toHaveBeenCalledWith("/lens/activity/available", { accessToken: "test" });
  expect(await guide.findByRole("link", { name: "View traces" })).toHaveAttribute(
    "href",
    expect.stringMatching(/^\/ui\/lens\/?\?tab=traces$/),
  );
  expect(await guide.findByRole("button", { name: "Preview sample" })).toBeVisible();
  await user.click(guide.getByRole("button", { name: "Connect worker" }));
  const connection = within(await screen.findByRole("dialog", { name: "Connect a worker" }));
  expect(connection.getByRole("button", { name: "Get install command" })).toBeVisible();
  await user.click(connection.getByRole("button", { name: "Close" }));
  expect(guide.getByRole("button", { name: "New investigation" })).toBeDisabled();
  await act(async () => {
    testQueryClient.setQueryData(lensKeys.list("test"), {
      lenses: [],
      tracing_enabled: true,
      workers: [
        {
          id: "worker",
          name: "Worker",
          revoked: false,
          analysis_key_id: "a".repeat(64),
          scope: lens.scope,
          last_seen: new Date().toISOString(),
        },
      ],
    });
  });
  await waitFor(() => expect(guide.getByRole("button", { name: "New investigation" })).toBeEnabled());
  expect(guide.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
  await user.click(guide.getByRole("button", { name: "New investigation" }));
  expect(await screen.findByRole("dialog", { name: "Which activity should we investigate?" })).toBeVisible();
});

it("opens the saved results of an older batch", async () => {
  testQueryClient.clear();
  const older = {
    ...lens.jobs[0],
    id: "older",
    created_at: "2026-09-29T10:00:00Z",
    finished_at: "2026-09-29T10:02:13Z",
    findings: [{ ...issue, title: "Earlier batch finding" }],
  };
  vi.mocked(apiClient.get).mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [lens], workers: [], tracing_enabled: true };
    if (path === "/lens/lens/runs") return [lens.jobs[0], older];
    if (path === "/lens/lens/runs/older") return older;
    return { data: [] };
  });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView accessToken="test" readOnly />);
  await screen.findByRole("option", { name: `${runTime(older.created_at)} · completed` });
  await user.selectOptions(screen.getByRole("combobox", { name: "Investigation run" }), "older");
  expect(await screen.findByText("Earlier batch finding")).toBeVisible();
  expect(screen.queryByText(issue.title)).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Run details" }));
  expect(screen.getByText(/Took 2m 13s/)).toBeVisible();
  expect(screen.getByText("Activity window")).toBeVisible();
  await user.keyboard("{Escape}");
  await user.click(screen.getByRole("tab", { name: "History" }));
  expect(within(screen.getByRole("tabpanel", { name: "History" })).getByText(/Took 2m 13s/)).toBeVisible();
});

it("reads request content from the beginning after its abbreviated preview", async () => {
  testQueryClient.clear();
  const requestId = btoa(JSON.stringify(["requests", "", "request-1"]));
  const job = {
    ...lens.jobs[0],
    sample: {
      eligible: 1,
      executions: [{ ...lens.jobs[0].sample!.executions[0], id: requestId, source: "requests" as const }],
    },
  };
  vi.mocked(apiClient.get).mockImplementation(async (path, options) => {
    if (path === "/lens") return { lenses: [{ ...lens, jobs: [job] }], workers: [], tracing_enabled: true };
    if (path === "/lens/lens/runs") return [job];
    if (!path.includes("/executions/")) return { data: [] };
    const offset = options?.query?.offset ?? 0;
    return {
      parts: [
        {
          span_id: "request",
          content: offset === 0 ? "Abbreviated preview" : `Original at ${offset}`,
          truncated: true,
        },
      ],
    };
  });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView accessToken="test" readOnly />);
  await user.click(await screen.findByRole("tab", { name: "Traces" }));
  await user.click(screen.getByRole("button", { name: /Release-42/ }));
  expect(await screen.findByText("Abbreviated preview")).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Next section" }));
  expect(await screen.findByText("Original at 1")).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Next section" }));
  expect(await screen.findByText("Original at 8001")).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Previous section" }));
  expect(await screen.findByText("Original at 1")).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Previous section" }));
  expect(await screen.findByText("Abbreviated preview")).toBeVisible();
});

it.each([false, true])(
  "directs a new user to traces when tracing_enabled=%s and there are no traces",
  async (enabled) => {
    window.history.replaceState({}, "", "/lens/");
    testQueryClient.clear();
    vi.mocked(apiClient.get).mockImplementation(async (path) =>
      path === "/lens" ? { lenses: [], workers: [], tracing_enabled: enabled } : { data: [] },
    );
    renderWithProviders(<InvestigationsView accessToken="test" />);
    expect(await screen.findByRole("heading", { name: "Find what needs attention" })).toBeVisible();
    expect(screen.getByRole("link", { name: "Set up traces" })).toHaveAttribute(
      "href",
      expect.stringMatching(/^\/ui\/lens\/?\?tab=traces$/),
    );
    expect(screen.getByRole("button", { name: "New investigation" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Connect worker" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Set up analysis" })).not.toBeInTheDocument();
  },
);

it("enables first-lens setup when a trace arrives without leaving Investigations", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  const traceCheck = vi.fn().mockResolvedValue({ traces: false, requests: false });
  vi.mocked(apiClient.get).mockImplementation(async (path) =>
    path === "/lens" ? { lenses: [], workers: [], tracing_enabled: true } : traceCheck(),
  );
  vi.useFakeTimers();
  try {
    const view = renderWithProviders(<InvestigationsView accessToken="test" />);
    await act(async () => vi.advanceTimersByTimeAsync(50));
    expect(screen.getByRole("link", { name: "Set up traces" })).toBeVisible();

    traceCheck.mockResolvedValue({ traces: true, requests: false });
    await act(async () => vi.advanceTimersByTimeAsync(5000));
    expect(screen.getByRole("button", { name: "Connect worker" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "New investigation" })).toBeDisabled();
    expect(screen.queryByRole("link", { name: "Set up traces" })).not.toBeInTheDocument();

    const completedChecks = traceCheck.mock.calls.length;
    await act(async () => vi.advanceTimersByTimeAsync(5000 * 2));
    expect(traceCheck.mock.calls.length).toBeGreaterThan(completedChecks);
    view.unmount();
  } finally {
    vi.useRealTimers();
  }
});

it("allows retrying a failed trace readiness check without treating it as an empty account", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  const traceCheck = vi
    .fn()
    .mockRejectedValueOnce(new ApiError("Trace storage unavailable", 503, {}))
    .mockResolvedValue({ data: [] });
  vi.mocked(apiClient.get).mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [], workers: [], tracing_enabled: true };
    if (path === "/lens/activity/available") return traceCheck();
    return { data: [] };
  });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView accessToken="test" />);
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Could not check recorded activity. Trace storage unavailable",
  );
  expect(screen.getByRole("button", { name: "Connect worker" })).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByRole("link", { name: "Set up traces" })).toBeVisible();
});

it("keeps saved investigations accessible when tracing is disabled", async () => {
  testQueryClient.clear();
  vi.mocked(apiClient.get).mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [lens], workers: [], tracing_enabled: false };
    if (path === "/lens/lens/runs") return lens.jobs;
    return { data: [] };
  });
  renderWithProviders(<InvestigationsView accessToken="test" readOnly />);
  expect(await screen.findByText(issue.title)).toBeVisible();
  expect(screen.queryByRole("link", { name: "Set up traces" })).not.toBeInTheDocument();
});

it("allows request-only accounts to connect a worker without requiring agent traces", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  vi.mocked(apiClient.get).mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [], workers: [], tracing_enabled: true };
    if (path === "/lens/activity/available") return { traces: false, requests: true };
    return { data: [] };
  });
  renderWithProviders(<InvestigationsView accessToken="test" />);
  expect(await screen.findByText("Request logs received")).toBeVisible();
  expect(screen.getByRole("button", { name: "Connect worker" })).toBeEnabled();
  expect(screen.getByRole("button", { name: "New investigation" })).toBeDisabled();
});

it("closes editing when browser navigation leaves the investigation", async () => {
  testQueryClient.clear();
  vi.mocked(apiClient.get).mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [lens], workers: [], tracing_enabled: true };
    if (path === "/lens/lens/runs") return lens.jobs;
    if (path === "/lens/agents") return [];
    return { data: [] };
  });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView accessToken="test" />);
  await user.click(await screen.findByRole("button", { name: "Investigation actions" }));
  await user.click(await screen.findByRole("menuitem", { name: "Edit investigation" }));
  expect(await screen.findByRole("dialog")).toBeVisible();
  await act(async () => {
    window.history.replaceState({}, "", "/lens/");
    window.dispatchEvent(new PopStateEvent("popstate"));
  });
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(apiClient.request).not.toHaveBeenCalled();
});
