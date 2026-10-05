import { act, fireEvent, screen, within, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { testQueryClient } from "@/../tests/test-utils";
import { renderWithLens, stubGateway } from "@/../tests/lens-test-utils";
import { ApiError } from "@/lib/http/client";
import { lensKeys } from "../data/queries";
import { InvestigationsView } from "./InvestigationsView";
import { briefMarkdown } from "../model/findings";
import { findingKey } from "../model/inbox";
import { runTime } from "../model/format";
import { type Lens, type Finding } from "../model/types";

function renderWithProviders(ui: React.ReactElement, options?: Parameters<typeof renderWithLens>[1]) {
  return renderWithLens(ui, { searchParams: window.location.search, ...options });
}

vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  proxyBaseUrl: "",
  getProxyBaseUrl: () => "",
}));

let proxy = stubGateway();
const sentBody = (handler: typeof proxy.post, path: string) =>
  handler.mock.calls.filter(([called]) => called === path).map(([, request]) => request.body);

beforeEach(() => {
  window.history.replaceState({}, "", "/lens/?lens=lens");
  proxy = stubGateway();
  proxy.post.mockResolvedValue({ eligible: 0, selected: 0, executions: [] });
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
      steps: [],
      trigger: "schedule",
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
    proxy.get.mockReset();
    proxy.get.mockImplementation(async (path) => {
      if (path === "/lens") return { lenses: [lens], workers: [], tracing_enabled: true };
      if (path === "/lens/lens/runs") return lens.jobs;
      return { data: [] };
    });
  });

  it("separates patterns from issues and reveals original evidence only when requested", async () => {
    const user = userEvent.setup();
    renderWithProviders(<InvestigationsView readOnly />);
    const investigation = within(await screen.findByRole("complementary", { name: "Investigation details" }));
    expect(investigation.getByText("Review used the wrong defect rate")).toBeInTheDocument();
    expect(investigation.queryByText(pattern.title)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Patterns (1)" }));
    await user.click(screen.getByRole("button", { name: new RegExp(pattern.title) }));
    const detail = within(screen.getByRole("complementary", { name: "Finding details" }));
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
    proxy.get.mockImplementation(async (path) => {
      if (path === "/lens")
        return { lenses: [{ ...lens, findings: [finding], jobs }], workers: [], tracing_enabled: true };
      if (path === "/lens/lens/runs") return jobs;
      return { data: [] };
    });
    const user = userEvent.setup();
    renderWithProviders(<InvestigationsView readOnly />);
    await user.click(await screen.findByRole("button", { name: new RegExp(finding.title) }));
    return { user, detail: within(screen.getByRole("complementary", { name: "Finding details" })) };
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
    renderWithProviders(<InvestigationsView readOnly />);
    await user.click(await screen.findByRole("tab", { name: "Agent traces" }));
    expect(screen.getByText("Release-42")).toBeInTheDocument();
    expect(screen.getByTitle("trace-42")).toHaveTextContent("Release-42");
    expect(screen.getByText(/1 selected from 1 matching runs/)).toBeInTheDocument();
  });

  it("closes the open run when the keyboard switches to another investigation run", async () => {
    testQueryClient.clear();
    const older = { ...lens.jobs[0], id: "older", created_at: "2026-09-29T10:00:00Z" };
    proxy.get.mockImplementation(async (path) => {
      if (path === "/lens") return { lenses: [lens], workers: [], tracing_enabled: true };
      if (path === "/lens/lens/runs") return [...lens.jobs, older];
      return { data: [] };
    });
    const onUrlUpdate = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(<InvestigationsView readOnly />, {
      searchParams: `?lens=lens&section=runs&evidence=${executionId}`,
      onUrlUpdate,
    });
    expect(await screen.findByTestId("run-panel")).toBeInTheDocument();
    const picker = screen.getByRole("combobox", { name: "Investigation run" });
    await waitFor(() => expect(within(picker).getAllByRole("option")).toHaveLength(4));
    fireEvent.change(picker, { target: { value: "older" } });
    await waitFor(() => expect(screen.queryByTestId("run-panel")).not.toBeInTheDocument());
    const url = new URLSearchParams(String(onUrlUpdate.mock.lastCall?.[0].queryString ?? ""));
    expect(url.get("run")).toBe("older");
    expect(url.has("evidence")).toBe(false);
  });
});

it("runs with saved settings from Run now without opening setup, then accepts an agent and window", async () => {
  testQueryClient.clear();
  proxy.get.mockImplementation(async (path) => {
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
  proxy.post.mockResolvedValue(lens);
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView />);
  await user.click(await screen.findByRole("button", { name: "Run now" }));
  const choices = await screen.findByRole("dialog", { name: "Run now" });
  expect(within(choices).getByRole("button", { name: "Since last run" })).toHaveAttribute("aria-pressed", "true");
  await user.click(within(choices).getByRole("button", { name: "Run now" }));
  expect(sentBody(proxy.post, "/lens/lens/runs")).toEqual([{}]);
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

  proxy.post.mockClear();
  await user.click(screen.getByRole("button", { name: "Run now" }));
  const custom = await screen.findByRole("dialog", { name: "Run now" });
  fireEvent.change(within(custom).getByRole("combobox", { name: "Agent" }), { target: { value: "billing" } });
  await user.click(within(custom).getByRole("button", { name: "Last 24h" }));
  await user.click(within(custom).getByRole("button", { name: "Run now" }));
  expect(sentBody(proxy.post, "/lens/lens/runs")).toEqual([{ agent_name: "billing", lookback_hours: 24 }]);
});

it("guides a first-time administrator into worker connection and lens setup", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [], workers: [], tracing_enabled: true };
    if (path === "/lens/agents") return [];
    return { traces: true, requests: false, data: [] };
  });
  const user = userEvent.setup();
  const connect = vi.fn();
  const create = vi.fn();
  renderWithProviders(<InvestigationsView />, {
    onboarding: { connect, create },
  });
  const guide = within(await screen.findByRole("region", { name: "Get Lens running" }));
  expect(proxy.get).toHaveBeenCalledWith(
    "/lens/activity/available",
    expect.objectContaining({ authorization: "Bearer test" }),
  );
  expect(guide.queryByRole("button", { name: /Send your first trace/ })).not.toBeInTheDocument();
  expect(guide.queryByRole("button", { name: /Enable tracing on the gateway/ })).not.toBeInTheDocument();
  expect(guide.getByRole("button", { name: /Connect a worker/ })).toHaveAttribute("aria-expanded", "true");
  expect(guide.queryByRole("button", { name: "View traces" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
  await user.click(guide.getByRole("button", { name: "Connect worker" }));
  expect(connect).toHaveBeenCalledOnce();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
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
  await user.click(await guide.findByRole("button", { name: "Continue to investigation" }));
  expect(create).toHaveBeenCalledOnce();
  expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
  await user.click(guide.getByRole("button", { name: /Run your first investigation/ }));
  expect(guide.getByRole("button", { name: "New investigation" })).toBeEnabled();
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
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [lens], workers: [], tracing_enabled: true };
    if (path === "/lens/lens/runs") return [lens.jobs[0], older];
    if (path === "/lens/lens/runs/older") return older;
    return { data: [] };
  });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView readOnly />);
  await screen.findByRole("option", { name: `${runTime(older.created_at)} · completed` });
  await user.selectOptions(screen.getByRole("combobox", { name: "Investigation run" }), "older");
  const investigation = within(screen.getByRole("complementary", { name: "Investigation details" }));
  expect(await investigation.findByText("Earlier batch finding")).toBeVisible();
  expect(investigation.queryByText(issue.title)).not.toBeInTheDocument();
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
  proxy.get.mockImplementation(async (path, options) => {
    if (path === "/lens") return { lenses: [{ ...lens, jobs: [job] }], workers: [], tracing_enabled: true };
    if (path === "/lens/lens/runs") return [job];
    if (!path.includes("/executions/")) return { data: [] };
    const offset = Number(options.query.offset ?? 0);
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
  renderWithProviders(<InvestigationsView readOnly />);
  await user.click(await screen.findByRole("tab", { name: "Agent traces" }));
  const row = screen.getByRole("button", { name: /Release-42/ });
  await user.click(row);
  const panel = await screen.findByRole("complementary", { name: "Run details" });
  expect(row).toHaveAttribute("aria-selected", "true");
  expect(panel).toHaveTextContent("1 / 1");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
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
    proxy.get.mockImplementation(async (path) =>
      path === "/lens" ? { lenses: [], workers: [], tracing_enabled: enabled } : { data: [] },
    );
    const user = userEvent.setup();
    renderWithProviders(<InvestigationsView />);
    const guide = within(await screen.findByRole("region", { name: "Get Lens running" }));
    expect(guide.getByRole("button", { name: /Send your first trace/ })).toHaveAttribute("aria-expanded", "true");
    expect(guide.getByRole("button", { name: "Check for traces" })).toBeVisible();
    await user.click(guide.getByRole("button", { name: /Connect a worker/ }));
    expect(guide.getByRole("button", { name: "Connect worker" })).toBeDisabled();
    await user.click(guide.getByRole("button", { name: /Run your first investigation/ }));
    expect(guide.getByRole("button", { name: "New investigation" })).toBeDisabled();
  },
);

it("enables first-lens setup when a trace arrives without leaving Investigations", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  const traceCheck = vi.fn().mockResolvedValue({ traces: false, requests: false });
  proxy.get.mockImplementation(async (path) =>
    path === "/lens" ? { lenses: [], workers: [], tracing_enabled: true } : traceCheck(),
  );
  vi.useFakeTimers();
  try {
    const view = renderWithProviders(<InvestigationsView />);
    await act(async () => vi.advanceTimersByTimeAsync(50));
    expect(screen.queryByText(/Your first trace is ready/)).not.toBeInTheDocument();

    traceCheck.mockResolvedValue({ traces: true, requests: false });
    await act(async () => vi.advanceTimersByTimeAsync(5000));
    expect(screen.queryByRole("button", { name: /Send your first trace/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Connect worker" })).toBeEnabled();

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
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [], workers: [], tracing_enabled: true };
    if (path === "/lens/activity/available") return traceCheck();
    return { data: [] };
  });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not check setup. Trace storage unavailable");
  expect(screen.getByRole("region", { name: "Get Lens running" })).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Retry" }));
  await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  expect(traceCheck).toHaveBeenCalledTimes(2);
});

it("shows a centered failure with a retry when investigations cannot load, then recovers", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  const list = vi
    .fn()
    .mockRejectedValueOnce(new ApiError("Proxy timed out", 504, {}))
    .mockResolvedValue({ lenses: [], workers: [], tracing_enabled: true });
  proxy.get.mockImplementation(async (path) => (path === "/lens" ? list() : { data: [] }));
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView />);
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("Couldn't load investigations");
  expect(alert).toHaveTextContent("Proxy timed out");
  await user.click(within(alert).getByRole("button", { name: "Try again" }));
  expect(await screen.findByRole("region", { name: "Get Lens running" })).toBeVisible();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("keeps saved investigations accessible when tracing is disabled", async () => {
  testQueryClient.clear();
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [lens], workers: [], tracing_enabled: false };
    if (path === "/lens/lens/runs") return lens.jobs;
    return { data: [] };
  });
  renderWithProviders(<InvestigationsView readOnly />);
  expect(await screen.findByRole("row", { name: issue.title })).toBeVisible();
  expect(screen.queryByRole("region", { name: "Get Lens running" })).not.toBeInTheDocument();
});

it("allows request-only accounts to connect a worker without requiring agent traces", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [], workers: [], tracing_enabled: true };
    if (path === "/lens/activity/available") return { traces: false, requests: true };
    return { data: [] };
  });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView />);
  expect(await screen.findByRole("button", { name: "Connect worker" })).toBeEnabled();
  expect(screen.queryByRole("button", { name: /Send your first trace/ })).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /Run your first investigation/ }));
  expect(screen.getByRole("button", { name: "New investigation" })).toBeDisabled();
});

it("reopens the inline editor from a shared link and drops it from the URL on cancel", async () => {
  testQueryClient.clear();
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [lens], workers: [], tracing_enabled: true };
    if (path === "/lens/lens/runs") return lens.jobs;
    if (path === "/lens/agents") return [];
    return { data: [] };
  });
  const onUrlUpdate = vi.fn();
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView />, {
    searchParams: `?lens=${lens.id}&dialog=edit`,
    onUrlUpdate,
  });
  const editor = await screen.findByRole("region", { name: "Edit investigation" });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(within(editor).getByDisplayValue(lens.settings.name)).toBeVisible();
  expect(screen.queryByRole("heading", { level: 2, name: lens.settings.name })).not.toBeInTheDocument();
  await user.click(within(editor).getByRole("button", { name: "Cancel" }));
  await waitFor(() => expect(screen.queryByRole("region", { name: "Edit investigation" })).not.toBeInTheDocument());
  expect(await screen.findByRole("heading", { level: 2, name: lens.settings.name })).toBeVisible();
  const url = new URLSearchParams(String(onUrlUpdate.mock.lastCall?.[0].queryString ?? ""));
  expect(url.has("dialog")).toBe(false);
  expect(url.get("lens")).toBe(lens.id);
  expect(proxy.put).not.toHaveBeenCalled();
  expect(sentBody(proxy.post, "/lens")).toEqual([]);
});

it("reopens a finding and a results section from shared links", async () => {
  testQueryClient.clear();
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [lens], workers: [], tracing_enabled: true };
    if (path === "/lens/lens/runs") return lens.jobs;
    return { data: [] };
  });
  const onUrlUpdate = vi.fn();
  const user = userEvent.setup();
  const { unmount } = renderWithProviders(<InvestigationsView />, {
    searchParams: `?issue=${encodeURIComponent(findingKey(lens, issue))}`,
    onUrlUpdate,
  });
  const panel = await screen.findByRole("complementary", { name: "Finding details" });
  expect(within(panel).getByRole("heading", { name: issue.title })).toBeVisible();
  expect(screen.getByRole("row", { name: issue.title })).toHaveAttribute("aria-selected", "true");
  await user.click(within(panel).getByRole("button", { name: "Close finding (Esc)" }));
  await waitFor(() =>
    expect(new URLSearchParams(String(onUrlUpdate.mock.lastCall?.[0].queryString)).has("issue")).toBe(false),
  );
  expect(screen.getByRole("row", { name: issue.title })).toHaveAttribute("aria-selected", "false");
  unmount();

  renderWithProviders(<InvestigationsView />, { searchParams: `?lens=${lens.id}&section=checks` });
  expect(await screen.findByRole("tab", { name: "Criteria", selected: true })).toBeVisible();
});

it("steps across findings and investigations with J and K, skipping hidden findings, and closes with Escape", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  const twinIssue: Finding = { ...issue, id: "twin-issue", title: "Twin review used the wrong defect rate" };
  const twin: Lens = {
    ...lens,
    id: "twin",
    settings: { ...lens.settings, name: "Twin reviews" },
    findings: [twinIssue],
    jobs: lens.jobs.map((job) => ({ ...job, findings: [twinIssue] })),
  };
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [lens, twin], tracing_enabled: true, workers: [] };
    if (path === "/lens/activity/available") return { traces: true, requests: false };
    if (path.endsWith("/runs")) return [];
    return { data: [] };
  });
  const onUrlUpdate = vi.fn();
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView />, { onUrlUpdate });
  const lastIssue = () => new URLSearchParams(String(onUrlUpdate.mock.lastCall?.[0].queryString ?? "")).get("issue");
  await user.click(await screen.findByRole("button", { name: `Hide findings for ${twin.settings.name}` }));
  expect(screen.queryByRole("row", { name: twinIssue.title })).not.toBeInTheDocument();

  await user.click(screen.getByRole("row", { name: issue.title }));
  const panel = await screen.findByRole("complementary", { name: "Finding details" });
  expect(panel).toHaveTextContent("2 / 3");
  await waitFor(() => expect(lastIssue()).toBe(findingKey(lens, issue)));

  await user.keyboard("j");
  await waitFor(() => expect(lastIssue()).toBeNull());
  expect(screen.getByRole("row", { name: twin.settings.name })).toHaveAttribute("aria-selected", "true");
  expect(screen.queryByRole("row", { name: twinIssue.title })).not.toBeInTheDocument();
  const twinPanel = screen.getByRole("complementary", { name: "Investigation details" });
  expect(within(twinPanel).getByRole("heading", { level: 2, name: twin.settings.name })).toBeVisible();
  expect(twinPanel).toHaveTextContent("3 / 3");
  expect(screen.getByRole("button", { name: "Next investigation (J)" })).toBeDisabled();

  await user.keyboard("k");
  await waitFor(() => expect(lastIssue()).toBe(findingKey(lens, issue)));
  await user.keyboard("{Escape}");
  await waitFor(() => expect(lastIssue()).toBeNull());
  expect(screen.getByRole("row", { name: issue.title })).toHaveAttribute("aria-selected", "false");
});

it("opens an investigation beside the list and walks from it into its findings with J and K", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [lens], tracing_enabled: true, workers: [] };
    if (path === "/lens/activity/available") return { traces: true, requests: false };
    if (path.endsWith("/runs")) return [];
    return { data: [] };
  });
  const onUrlUpdate = vi.fn();
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView />, { onUrlUpdate });
  const lastUrl = () => new URLSearchParams(String(onUrlUpdate.mock.lastCall?.[0].queryString ?? ""));
  await user.click(await screen.findByRole("row", { name: lens.settings.name }));
  const panel = await screen.findByRole("complementary", { name: "Investigation details" });
  expect(within(panel).getByRole("heading", { level: 2, name: lens.settings.name })).toBeVisible();
  expect(screen.getByRole("table", { name: "Investigations" })).toBeVisible();
  expect(panel).toHaveTextContent("1 / 2");
  await waitFor(() => expect(lastUrl().get("lens")).toBe(lens.id));

  await user.keyboard("j");
  await waitFor(() => expect(lastUrl().get("issue")).toBe(findingKey(lens, issue)));
  expect(lastUrl().has("lens")).toBe(false);
  expect(screen.getByRole("complementary", { name: "Finding details" })).toHaveTextContent("2 / 2");
  expect(screen.getByRole("row", { name: issue.title })).toHaveAttribute("aria-selected", "true");

  await user.keyboard("k");
  await waitFor(() => expect(lastUrl().get("lens")).toBe(lens.id));
  expect(lastUrl().has("issue")).toBe(false);
  expect(screen.getByRole("complementary", { name: "Investigation details" })).toHaveTextContent("1 / 2");
});

it("lists each finding under the investigation that owns it and resolves only that copy", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  const twin: Lens = { ...lens, id: "twin", settings: { ...lens.settings, name: "Twin reviews" } };
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [lens, twin], tracing_enabled: true, workers: [] };
    if (path === "/lens/activity/available") return { traces: true, requests: false };
    if (path.endsWith("/runs")) return [];
    return { data: [] };
  });
  proxy.patch.mockResolvedValue(undefined);
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView />);
  const rows = await screen.findAllByRole("row", { name: issue.title });
  expect(rows).toHaveLength(2);
  await user.click(await screen.findByRole("button", { name: `Hide findings for ${lens.settings.name}` }));
  const remaining = screen.getByRole("row", { name: issue.title });
  expect(remaining.previousElementSibling).toBe(screen.getByRole("row", { name: twin.settings.name }));
  await user.click(remaining);
  await user.click(await screen.findByRole("button", { name: "Mark resolved" }));
  await waitFor(() => expect(proxy.patch).toHaveBeenCalledTimes(1));
  expect(proxy.patch.mock.calls[0][0]).toBe("/lens/twin/findings/issue");
});

it("lists investigations without edit or run controls for read-only viewers", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [lens], tracing_enabled: true, workers: [] };
    if (path === "/lens/activity/available") return { traces: true, requests: false };
    if (path === "/lens/lens/runs") return [];
    if (path === "/lens/agents") return [];
    return { data: [] };
  });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView readOnly />);
  const row = await screen.findByRole("row", { name: lens.settings.name });
  expect(within(row).queryByRole("button", { name: /now/ })).not.toBeInTheDocument();
  await user.click(row);
  expect(await screen.findByRole("heading", { level: 2, name: lens.settings.name })).toBeVisible();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

it("opens investigations from the keyboard without treating nested edit keys as row activation", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [lens], tracing_enabled: true, workers: [] };
    if (path === "/lens/activity/available") return { traces: true, requests: false };
    if (path === "/lens/lens/runs") return [];
    if (path === "/lens/agents") return [];
    return { data: [] };
  });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView />);
  const row = await screen.findByRole("row", { name: lens.settings.name });
  row.focus();
  expect(row).toHaveFocus();
  await user.keyboard("{Enter}");
  expect(await screen.findByRole("heading", { level: 2, name: lens.settings.name })).toBeVisible();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

  expect(screen.getByRole("row", { name: lens.settings.name })).toHaveAttribute("aria-selected", "true");
  await user.click(screen.getByRole("button", { name: "Close investigation (Esc)" }));
  const editButton = await screen.findByRole("button", { name: `Edit ${lens.settings.name}` });
  editButton.focus();
  expect(editButton).toHaveFocus();
  await user.keyboard("{Enter}");
  const editor = await screen.findByRole("region", { name: "Edit investigation" });
  expect(within(editor).getByDisplayValue(lens.settings.name)).toBeVisible();
  expect(screen.queryByRole("heading", { level: 2, name: lens.settings.name })).not.toBeInTheDocument();
  expect(screen.queryByRole("row", { name: lens.settings.name })).not.toBeInTheDocument();
});

it("opens a failed investigation's details from its row and edits only from the pencil", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  const job = {
    ...lens.jobs[0],
    id: "failed-run",
    status: "failed" as const,
    stage: "Failed",
    error: "boom",
    findings: [],
  };
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [{ ...lens, jobs: [job] }], workers: [], tracing_enabled: true };
    if (path === "/lens/activity/available") return { traces: true, requests: false };
    if (path === "/lens/lens/runs") return [job];
    if (path === "/lens/lens/runs/failed-run") return job;
    if (path === "/lens/agents") return [];
    return { data: [] };
  });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView />);
  await user.click(await screen.findByRole("button", { name: `Edit ${lens.settings.name}` }));
  const editor = await screen.findByRole("region", { name: "Edit investigation" });
  expect(within(editor).getByDisplayValue(lens.settings.name)).toBeVisible();
  await user.click(within(editor).getByRole("button", { name: "Cancel" }));
  await user.click(await screen.findByRole("row", { name: lens.settings.name }));
  expect(within(await screen.findByRole("alert")).getByLabelText("Investigation error")).toHaveTextContent("boom");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

it("shows the actual saved failure and run context without opening backend logs", async () => {
  testQueryClient.clear();
  const error =
    "Grouping observations failed: Clusters response invalid after 2 attempts.\n" +
    "candidates.0.check_id: Field required [missing]";
  const job = { ...lens.jobs[0], id: "failed-run", status: "failed" as const, stage: "Failed", error, findings: [] };
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [{ ...lens, jobs: [job] }], workers: [], tracing_enabled: true };
    if (path === "/lens/lens/runs") return [job];
    if (path === "/lens/lens/runs/failed-run") return job;
    return { data: [] };
  });
  renderWithProviders(<InvestigationsView readOnly />);
  const failure = within(await screen.findByRole("alert"));
  expect(failure.getByLabelText("Investigation error")).toHaveTextContent(error.replaceAll("\n", " "));
  expect(failure.getByText("failed-run")).toBeVisible();
  expect(failure.getByText(job.settings.model)).toBeVisible();
  expect(failure.queryByText(/find the error in proxy and worker logs/)).not.toBeInTheDocument();
});

it("keeps a finding open to retry when its update fails", async () => {
  window.history.replaceState({}, "", "/lens/");
  testQueryClient.clear();
  const twin: Lens = { ...lens, id: "twin", settings: { ...lens.settings, name: "Twin reviews" } };
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [lens, twin], tracing_enabled: true, workers: [] };
    if (path === "/lens/activity/available") return { traces: true, requests: false };
    if (path.endsWith("/runs")) return [];
    return { data: [] };
  });
  proxy.patch.mockReset();
  proxy.patch.mockImplementation(async (path) => {
    if (String(path).startsWith("/lens/twin/")) throw new Error("Twin reviews could not be updated");
  });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView />);
  await user.click(await screen.findByRole("button", { name: `Hide findings for ${lens.settings.name}` }));
  await user.click(screen.getByRole("row", { name: issue.title }));
  await user.click(await screen.findByRole("button", { name: "Mark resolved" }));
  expect(await screen.findByText("Twin reviews could not be updated")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Mark resolved" })).toBeVisible();
});

it("pauses monitoring from the detail menu by saving the investigation with monitoring off", async () => {
  testQueryClient.clear();
  const watching: Lens = { ...lens, settings: { ...lens.settings, enabled: true } };
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [watching], workers: [], tracing_enabled: true };
    if (path === "/lens/lens/runs") return watching.jobs;
    return { data: [] };
  });
  proxy.put.mockResolvedValue({ ...watching, settings: lens.settings });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView />);
  await user.click(await screen.findByRole("button", { name: "Investigation actions" }));
  await user.click(await screen.findByRole("menuitem", { name: "Pause monitoring" }));
  await waitFor(() => expect(proxy.put).toHaveBeenCalledTimes(1));
  expect(sentBody(proxy.put, "/lens/lens")).toEqual([{ ...watching.settings, enabled: false }]);
});

it("cancels the running job from the progress banner", async () => {
  testQueryClient.clear();
  const running = { ...lens.jobs[0], id: "live", status: "running" as const, stage: "Reading executions" };
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens")
      return { lenses: [{ ...lens, jobs: [running, lens.jobs[0]] }], workers: [], tracing_enabled: true };
    if (path === "/lens/lens/runs") return [running, lens.jobs[0]];
    return { data: [] };
  });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView />);
  await user.click(await screen.findByRole("button", { name: "Cancel" }));
  await waitFor(() => expect(proxy.post).toHaveBeenCalledWith("/lens/lens/cancel", expect.anything()));
});

it("refreshes run history as soon as the list reports a job the scheduler started", async () => {
  testQueryClient.clear();
  const runs = vi.fn().mockResolvedValue(lens.jobs);
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses: [lens], workers: [], tracing_enabled: true };
    if (path === "/lens/lens/runs") return runs();
    return { data: [] };
  });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationsView readOnly />);
  await user.click(await screen.findByRole("tab", { name: "History" }));
  const history = within(await screen.findByRole("tabpanel", { name: "History" }));
  expect(await history.findByText("completed")).toBeVisible();
  const fetched = runs.mock.calls.length;

  const queued = { ...lens.jobs[0], id: "scheduled", status: "queued" as const, stage: "Queued", findings: null };
  runs.mockResolvedValue([queued, lens.jobs[0]]);
  await act(async () => {
    testQueryClient.setQueryData(lensKeys.list("test"), {
      lenses: [{ ...lens, jobs: [queued, lens.jobs[0]] }],
      workers: [],
      tracing_enabled: true,
    });
  });
  expect(await history.findByText("queued")).toBeVisible();
  expect(runs.mock.calls.length).toBeGreaterThan(fetched);
});
