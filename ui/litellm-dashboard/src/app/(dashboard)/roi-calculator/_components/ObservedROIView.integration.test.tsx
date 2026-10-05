import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import ObservedROIView from "./ObservedROIView";
import { createObservedDemo } from "./observedDemo";
import type { ObservedSettings, ObservedSnapshot, ObservedStatus } from "./observedData";

const settings: ObservedSettings = {
  source_provider: "gitlab",
  api_url: "https://gitlab.com/api/v4",
  repos: ["org/service"],
  has_token: true,
  connection_type: "app",
  update_interval_minutes: 1440,
  ready: true,
};
const idle: ObservedStatus = {
  running: false,
  phase: "complete",
  stage: "",
  done: 0,
  total: 0,
  error: null,
  finished_at: null,
};
const period = {
  window: { start: "2026-09-01", end: "2026-09-28" },
  merged_prs: 1,
  median_merge_hours: 16 / 3600,
  human_authored: 1,
  agent_authored: 0,
  missing_author: 0,
  agents_without_requester: 0,
  matched_internal_prs: 0,
  new_bug_labeled_issues: 0,
  new_regression_labeled_issues: 0,
  explicitly_titled_revert_prs: 0,
  matched_users_recorded_spend: 0,
  spend_observation: "no_records" as const,
  human_summary: { median_merge_hours: 16 / 3600 },
};
const report: ObservedSnapshot = {
  source_provider: "gitlab",
  repos: ["org/service"],
  unmatched_logins: [],
  unlinked_branches: [],
  captured_at: "2026-09-29T00:00:00Z",
  periods: { current: period, previous: period, last_year: period },
  people: [],
  pulls: { current: [], previous: [], last_year: [] },
};
const app = { configured: true, api_url: null, callback_url: null };

afterEach(() => {
  vi.unstubAllGlobals();
  window.history.replaceState(null, "", "/");
});

describe("observed ROI dashboard", () => {
  it("defaults every contributor list to matched people and keeps the switch across tabs", async () => {
    const sample = createObservedDemo(7);
    const outside = {
      ...sample.pulls.current[0],
      url: "https://gitlab.com/outside/api/-/merge_requests/999",
      author: "outside",
      agent: false,
      title: "Outside change",
      source_branch: "outside-only",
      branch_cost: {
        repo: "gitlab.com/outside/api",
        branch: "outside-only",
        spend: 123,
        requests: 5,
        status: "matched" as const,
      },
    };
    const report = {
      ...sample,
      pulls: { ...sample.pulls, current: [...sample.pulls.current, outside] },
      unlinked_branches: [{ repo: "gitlab.com/outside/api", branch: "orphan-only", spend: 5, requests: 1 }],
    };
    const requests = vi.fn(async (input: string, _init: RequestInit) => {
      const path = new URL(input, "http://localhost").pathname;
      if (path.endsWith("/settings")) return Response.json(settings);
      if (path.endsWith("/report")) return Response.json({ report });
      if (path.endsWith("/sync")) return Response.json(idle);
      throw new Error(path);
    });
    vi.stubGlobal("fetch", requests);
    const user = userEvent.setup();
    render(<ObservedROIView accessToken="test-only-gateway-token" isViewOnly />);
    expect(await screen.findByRole("switch", { name: "Matched people only" })).toBeChecked();
    expect(screen.getByRole("tab", { name: "Engineers 3" })).toBeInTheDocument();
    expect(screen.queryByText("outside")).not.toBeInTheDocument();
    await user.click(screen.getByRole("switch", { name: "Matched people only" }));
    expect(screen.getByRole("tab", { name: "Engineers 4" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "View outside's merged changes" }));
    expect(await screen.findByRole("dialog", { name: "outside" })).toHaveTextContent("Outside change");
    expect(screen.queryByRole("button", { name: "Edit linked accounts" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Close" }));
    await user.click(screen.getByRole("switch", { name: "Matched people only" }));
    await user.click(screen.getByRole("tab", { name: "Merged changes" }));
    expect(screen.queryByText("Outside change")).not.toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Branch spend" }));
    expect(screen.getAllByText(/feature\/sample-/).length).toBeGreaterThan(0);
    expect(screen.queryByText("outside-only")).not.toBeInTheDocument();
    expect(screen.queryByText("orphan-only")).not.toBeInTheDocument();
    await user.click(screen.getByRole("switch", { name: "Matched people only" }));
    expect(screen.getByText("outside-only")).toBeInTheDocument();
    expect(screen.getByText("orphan-only")).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Merged changes" }));
    expect(screen.getByText("Outside change")).toBeInTheDocument();
    expect(requests.mock.calls.every(([, init]) => init.method === "GET")).toBe(true);
  });

  it("shows a useful empty matched view and exposes all changes when the filter is off", async () => {
    const sample = { ...createObservedDemo(7), people: [] };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string) => {
        const path = new URL(input, "http://localhost").pathname;
        if (path.endsWith("/settings")) return Response.json(settings);
        if (path.endsWith("/report")) return Response.json({ report: sample });
        return Response.json(idle);
      }),
    );
    const user = userEvent.setup();
    render(<ObservedROIView accessToken="test-only-gateway-token" />);
    expect(await screen.findByText("No merged changes from matched people in this period")).toBeInTheDocument();
    await user.click(screen.getByRole("switch", { name: "Matched people only" }));
    expect(screen.queryByText("No merged changes from matched people in this period")).not.toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: /Add repository search/ }).length).toBeGreaterThan(0);
  });

  it("previews every sample view before setup, changes sample periods without writes, and exits back to setup", async () => {
    const requests = vi.fn(async (input: string, _init: RequestInit) => {
      const path = new URL(input, "http://localhost").pathname;
      const disconnected = { ...settings, ready: false, repos: [], has_token: false };
      if (path.endsWith("/settings")) return Response.json(disconnected);
      if (path.endsWith("/report")) return Response.json({ report: null });
      if (path.endsWith("/sync")) return Response.json(idle);
      throw new Error(path);
    });
    vi.stubGlobal("fetch", requests);
    const user = userEvent.setup();
    render(<ObservedROIView accessToken="test-only-gateway-token" />);
    expect(await screen.findByRole("heading", { name: "Connect your repositories" })).toBeInTheDocument();
    expect(screen.queryByText("Ready to sync")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Preview sample report" }));
    expect(screen.getByRole("status")).toHaveTextContent("You’re viewing demo data");
    expect(screen.getByRole("tab", { name: "Engineers 3", selected: true })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Connections" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Link accounts" })).not.toBeInTheDocument();
    expect(window.location.search).toBe("?demo=1");
    await user.click(screen.getByRole("button", { name: "View Alex Rivera's merged changes" }));
    expect(await screen.findByRole("dialog", { name: "Alex Rivera" })).toHaveTextContent("alex-demo@example.com");
    expect(screen.getByRole("heading", { name: "Merged changes" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit linked accounts" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Close" }));
    await user.click(screen.getByRole("tab", { name: "Merged changes" }));
    expect(screen.getByRole("img", { name: /Merged changes by week/ })).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Quality" }));
    expect(screen.getByText("New regression-labeled issues")).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Branch spend" }));
    expect(screen.getAllByText(/feature\/sample-/).length).toBeGreaterThan(0);
    await user.click(screen.getByRole("combobox", { name: "Reporting period" }));
    await user.click(await screen.findByRole("option", { name: "Last 7 days" }));
    expect(screen.getByRole("combobox", { name: "Reporting period" })).toHaveTextContent("Last 7 days");
    await user.click(screen.getByRole("combobox", { name: "Comparison period" }));
    await user.click(await screen.findByRole("option", { name: "vs. same period last year" }));
    expect(screen.getByRole("combobox", { name: "Comparison period" })).toHaveTextContent("vs. same period last year");
    await user.click(screen.getByRole("button", { name: "Exit demo" }));
    expect(screen.getByRole("heading", { name: "Connect your repositories" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Connect GitHub or GitLab" })).toBeEnabled();
    expect(window.location.search).toBe("");
    expect(requests.mock.calls.every(([, init]) => init.method === "GET")).toBe(true);
  });

  it("keeps live sync and its report intact when entering and exiting the demo", async () => {
    const requests = vi.fn(async (input: string, _init: RequestInit) => {
      const path = new URL(input, "http://localhost").pathname;
      if (path.endsWith("/settings")) return Response.json(settings);
      if (path.endsWith("/report")) return Response.json({ report });
      if (path.endsWith("/sync")) return Response.json({ ...idle, running: true, stage: "Reading changes" });
      throw new Error(path);
    });
    vi.stubGlobal("fetch", requests);
    window.history.replaceState(null, "", "/roi-calculator/?from=review#report");
    const user = userEvent.setup();
    render(<ObservedROIView accessToken="test-only-gateway-token" />);
    expect(await screen.findByRole("button", { name: "Cancel sync" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "Preview sample report" }));
    expect(screen.queryByRole("button", { name: "Cancel sync" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "2 repositories" })).toBeInTheDocument();
    expect(window.location.search).toBe("?from=review&demo=1");
    await user.click(screen.getByRole("button", { name: "Exit demo" }));
    expect(screen.getByRole("button", { name: "Cancel sync" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "1 repository" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Merge requests", selected: true })).toBeInTheDocument();
    expect(window.location.search).toBe("?from=review");
    expect(window.location.hash).toBe("#report");
    expect(requests.mock.calls.every(([, init]) => init.method === "GET")).toBe(true);
  });

  it.each(["failed", "pending"])("opens a demo URL even when live requests are %s", async (state) => {
    window.history.replaceState(null, "", "/roi-calculator/?demo=1");
    const pending = Promise.withResolvers<Response>();
    vi.stubGlobal(
      "fetch",
      vi.fn(() => (state === "failed" ? Promise.reject(new Error("Live data unavailable")) : pending.promise)),
    );
    const user = userEvent.setup();
    render(<ObservedROIView accessToken="test-only-gateway-token" isViewOnly />);
    expect(screen.getByRole("status")).toHaveTextContent("You’re viewing demo data");
    expect(screen.getByText("Alex Rivera")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Exit demo" }));
    expect(screen.queryByText("Alex Rivera")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Connect your repositories" })).not.toBeInTheDocument();
    expect(window.location.search).toBe("");
    if (state === "failed") expect(await screen.findByRole("alert")).toHaveTextContent("Live data unavailable");
  });

  it("retries failures, keeps the report during cancellation, and refreshes after completion", async () => {
    let status: ObservedStatus = { ...idle, phase: "error", error: "Provider temporarily unavailable" };
    let completeOnPoll = false;
    let currentReport = { ...report, repos: ["org/service", "org/docs"] };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string, init: RequestInit) => {
        const path = new URL(input, "http://localhost").pathname;
        if (path.endsWith("/settings")) return Response.json(settings);
        if (path.endsWith("/report")) return Response.json({ report: currentReport });
        if (path.endsWith("/sync")) {
          if (init.method === "POST") status = { ...idle, running: true, phase: "pulls" };
          if (init.method === "DELETE") status = { ...idle, phase: "cancelled" };
          if (init.method === "GET" && completeOnPoll) {
            status = { ...idle, finished_at: "2026-09-29T00:01:00Z" };
            currentReport = { ...report, repos: ["org/updated"] };
          }
          return Response.json(status);
        }
        throw new Error(path);
      }),
    );
    const user = userEvent.setup();
    render(<ObservedROIView accessToken="test-only-gateway-token" />);
    expect(await screen.findByRole("tab", { name: "Merge requests", selected: true })).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Quality" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Provider temporarily unavailable");
    await user.click(screen.getByRole("button", { name: "Retry" }));
    await user.click(await screen.findByRole("button", { name: "Cancel sync" }));
    expect(await screen.findByRole("button", { name: "Sync now" })).toBeEnabled();
    expect(screen.queryByText("org/service")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "2 repositories" }));
    const repositories = await screen.findByRole("dialog", { name: "Repositories" });
    expect(within(repositories).getByText("org/service")).toBeInTheDocument();
    expect(within(repositories).getByText("org/docs")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Repositories" })).not.toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: "Sync now" }));
    expect(await screen.findByRole("button", { name: "Cancel sync" })).toBeEnabled();
    completeOnPoll = true;
    await user.click(await screen.findByRole("button", { name: "1 repository" }, { timeout: 4000 }));
    expect(
      await within(screen.getByRole("dialog", { name: "Repositories" })).findByText("org/updated"),
    ).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.getByRole("tab", { name: "Quality", selected: true })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sync now" })).toBeEnabled();
  });

  it("syncs the selected range and labels its equal-length comparison", async () => {
    let currentReport = report;
    const requested: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string, init: RequestInit) => {
        const url = new URL(input, "http://localhost");
        if (url.pathname.endsWith("/settings")) return Response.json(settings);
        if (url.pathname.endsWith("/report")) return Response.json({ report: currentReport });
        if (url.pathname.endsWith("/sync")) {
          if (init.method === "POST") {
            requested.push(url.searchParams.get("days") ?? "");
            currentReport = {
              ...report,
              periods: {
                ...report.periods,
                current: { ...period, window: { start: "2026-09-22", end: "2026-09-28" } },
                previous: { ...period, window: { start: "2026-09-15", end: "2026-09-21" } },
              },
            };
          }
          return Response.json(idle);
        }
        throw new Error(url.pathname);
      }),
    );
    const user = userEvent.setup();
    render(<ObservedROIView accessToken="test-only-gateway-token" />);
    await user.click(await screen.findByRole("combobox", { name: "Reporting period" }));
    await user.click(await screen.findByRole("option", { name: "Last 7 days" }));
    await waitFor(() => expect(requested).toEqual(["7"]));
    expect(await screen.findByText(/Comparing with Sep 15.*Sep 21/)).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Reporting period" })).toHaveTextContent("Last 7 days");
    expect(screen.getByRole("combobox", { name: "Comparison period" })).toHaveTextContent("vs. previous period");
  });

  it("shows a successful empty repository without a setup prompt or invented durations", async () => {
    const emptyPeriod = {
      ...period,
      merged_prs: 0,
      human_authored: 0,
      median_merge_hours: null,
      human_summary: { median_merge_hours: null },
    };
    const empty = { ...report, periods: { current: emptyPeriod, previous: emptyPeriod, last_year: emptyPeriod } };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string) => {
        const path = new URL(input, "http://localhost").pathname;
        if (path.endsWith("/settings")) return Response.json(settings);
        if (path.endsWith("/report")) return Response.json({ report: empty });
        if (path.endsWith("/sync")) return Response.json(idle);
        throw new Error(path);
      }),
    );
    render(<ObservedROIView accessToken="test-only-gateway-token" />);
    expect(await screen.findByRole("heading", { name: "No merged changes yet" })).toBeInTheDocument();
    expect(screen.getByText("No merges")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sync now" })).toBeEnabled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Connect your repositories" })).not.toBeInTheDocument();
    expect(screen.queryByText("0h")).not.toBeInTheDocument();
  });

  it.each([
    { query: "connected=gitlab", alerts: [] },
    { query: "connection_failed=1", alerts: ["Connection failed or expired. Try again or use a token"] },
    { query: "connection_cancelled=1", alerts: ["Connection cancelled. Choose an app or token to try again"] },
  ])("resumes setup after $query and refreshes saved changes after closing", async ({ query, alerts }) => {
    window.history.replaceState(null, "", `/roi-calculator/?${query}`);
    let currentSettings = settings;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string, init: RequestInit) => {
        const path = new URL(input, "http://localhost").pathname;
        if (path.endsWith("/apps")) return Response.json({ github: app, gitlab: app });
        if (path.endsWith("/repositories")) return Response.json({ repositories: [], has_more: false });
        if (path.endsWith("/settings")) {
          if (init.method === "PUT") currentSettings = { ...settings, repos: ["org/changed"] };
          return Response.json(currentSettings);
        }
        if (path.endsWith("/report")) return Response.json({ report: { ...report, repos: currentSettings.repos } });
        if (path.endsWith("/sync"))
          return init.method === "POST"
            ? Response.json({ detail: "Provider unavailable" }, { status: 502 })
            : Response.json(idle);
        throw new Error(path);
      }),
    );
    const user = userEvent.setup();
    render(<ObservedROIView accessToken="test-only-gateway-token" />);
    const dialog = await screen.findByRole("dialog", { name: "Choose repositories" });
    expect(
      within(dialog)
        .queryAllByRole("alert")
        .map((alert) => alert.textContent),
    ).toEqual(alerts);
    expect(window.location.search).toBe("");
    fireEvent.change(within(dialog).getByLabelText("Repositories"), { target: { value: "org/changed" } });
    await user.click(within(dialog).getByRole("button", { name: "Save and sync" }));
    expect(await within(dialog).findByRole("alert")).toHaveTextContent("Provider unavailable");
    await user.click(within(dialog).getByRole("button", { name: "Close" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    await user.click(await screen.findByRole("button", { name: "1 repository" }));
    expect(
      await within(screen.getByRole("dialog", { name: "Repositories" })).findByText("org/changed"),
    ).toBeInTheDocument();
  });
});
