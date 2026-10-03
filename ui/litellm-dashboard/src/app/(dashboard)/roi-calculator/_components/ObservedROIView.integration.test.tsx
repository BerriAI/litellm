import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import ObservedROIView from "./ObservedROIView";
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
  it("keeps the report during cancellation and replaces it when a later sync finishes", async () => {
    let status = idle;
    let completeOnPoll = false;
    let currentReport = report;
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
    expect(await screen.findByRole("columnheader", { name: "Merged MRs" })).toBeInTheDocument();
    expect(screen.getByText("<1m")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Sync now" }));
    await user.click(await screen.findByRole("button", { name: "Cancel sync" }));
    expect(await screen.findByRole("button", { name: "Sync now" })).toBeEnabled();
    expect(screen.getByText("org/service")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Sync now" }));
    expect(await screen.findByRole("button", { name: "Cancel sync" })).toBeEnabled();
    completeOnPoll = true;
    expect(await screen.findByText("org/updated", {}, { timeout: 4000 })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sync now" })).toBeEnabled();
  });

  it("resumes repository selection after app authorization and refreshes saved changes after closing", async () => {
    window.history.replaceState(null, "", "/roi-calculator/?connected=gitlab");
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
    expect(window.location.search).toBe("");
    fireEvent.change(within(dialog).getByLabelText("Repositories"), { target: { value: "org/changed" } });
    await user.click(within(dialog).getByRole("button", { name: "Save and sync" }));
    expect(await within(dialog).findByRole("alert")).toHaveTextContent("Provider unavailable");
    await user.click(within(dialog).getByRole("button", { name: "Close" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(await screen.findByText("org/changed")).toBeInTheDocument();
  });
});
