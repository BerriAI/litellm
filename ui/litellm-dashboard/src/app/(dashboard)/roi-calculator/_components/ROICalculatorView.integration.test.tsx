import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiClient } from "@/components/networking";
import ROICalculatorView from "./ROICalculatorView";

vi.mock("@/components/networking", () => ({
  apiClient: {
    delete: vi.fn(),
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
  },
}));

const summary = {
  id: null,
  mode: "live",
  start: "2026-09-01",
  end: "2026-09-30",
  synced_at: "2026-09-30T12:00:00Z",
  repos: ["org/repo"],
  estimator_model: "estimator",
  estimator_prompt: "Estimate hours.",
  warnings: [],
  effort_basis: "without_ai",
  metrics: {
    matched_spend: 12,
    output_hours: 4,
    total_spend: 20,
    total_output_hours: 4,
    excluded_spend: 8,
    cost_per_hour: 3,
    hours_per_dollar: 1 / 3,
    merged_prs: 1,
    estimated_prs: 1,
    matched_prs: 1,
    cohort_people: 1,
    people_with_prs: 1,
    pending_prs: 0,
  },
  people: [
    {
      id: "alice@example.com",
      email: "alice@example.com",
      logins: ["alice", "alice-work"],
      spend: 12,
      hours: 4,
      prs: 1,
      estimated_prs: 1,
      pending_prs: 0,
      match_methods: ["profile email"],
      eligible: true,
      cost_per_hour: 3,
    },
  ],
  pulls: [
    {
      repo: "org/repo",
      source_repo: "github.com/org/repo",
      source_branch: "feature/routing",
      branch_cost: {
        repo: "github.com/org/repo",
        branch: "feature/routing",
        status: "matched",
        spend: 8,
        requests: 12,
      },
      number: 42,
      title: "Improve request routing",
      url: "https://github.com/org/repo/pull/42",
      login: "alice",
      emails: ["alice@example.com"],
      profile_email: "alice@example.com",
      merged_at: "2026-09-12T00:00:00Z",
      head_sha: "abc",
      additions: 10,
      deletions: 2,
      changed_files: 1,
      commit_count: 1,
      incomplete_metadata: false,
      estimate: {
        status: "estimated",
        hours: 4,
        reasoning: "Updated routing and added a regression test.",
        model: "estimator",
        evidence_source: "pr_metadata",
        effort_basis: "without_ai",
        cached: false,
      },
      cache_key: "cache",
      email: "alice@example.com",
      match_method: "profile email",
      matched: true,
    },
  ],
  trend: [{ date: "2026-09-12", spend: 12, hours: 4, prs: 1 }],
} as const;

const settings = {
  github_api_url: "https://api.github.com",
  repos: ["org/repo"],
  estimator_model: "estimator",
  estimator_prompt: "Estimate hours.",
  backfill_days: 30,
  identity_map: {},
  has_github_token: true,
  default_prompt: "Estimate hours.",
  available_models: ["estimator"],
  ready: true,
};

const idleStatus = {
  running: false,
  phase: "idle",
  stage: "Idle",
  done: 0,
  total: 0,
  estimated: 0,
  reused: 0,
  needs_attention: 0,
  error: null,
};

describe("ROICalculatorView", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", "/roi-calculator/");
    vi.mocked(apiClient.get).mockReset();
    vi.mocked(apiClient.put).mockReset();
    vi.mocked(apiClient.post).mockReset();
    vi.mocked(apiClient.get).mockImplementation((path: string) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(settings);
      if (path === "/roi-calculator/report") return Promise.resolve({ report: summary });
      return Promise.resolve(idleStatus);
    });
    vi.mocked(apiClient.put).mockResolvedValue({ report: summary, identity_map: { alice: "alice@example.com" } });
  });

  it("shows the spend summary and opens an accessible pull reasoning dialog", async () => {
    render(<ROICalculatorView accessToken="token" />);

    expect(await screen.findByText("Gateway AI cost")).toBeInTheDocument();
    expect(screen.getByText("$20.00")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open estimate for org/repo pull request 42" }));

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Updated routing and added a regression test.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "View on GitHub" })).toHaveAttribute(
      "href",
      "https://github.com/org/repo/pull/42",
    );
  });

  it("separates the overview, people, and branch reports into three tabs", async () => {
    render(<ROICalculatorView accessToken="token" />);
    expect(await screen.findByRole("heading", { name: "Where AI costs are matched" })).toBeVisible();
    fireEvent.click(screen.getByRole("tab", { name: "Branches" }));
    expect(screen.getByRole("heading", { name: "Costs by branch" })).toBeVisible();
    expect(screen.getByRole("cell", { name: "$8.00" })).toBeVisible();
    fireEvent.click(screen.getByRole("tab", { name: "People" }));
    expect(screen.getByRole("heading", { name: "People and account matches" })).toBeVisible();
    fireEvent.click(screen.getByRole("tab", { name: "Overview" }));
    expect(screen.getByRole("heading", { name: "Highest-cost changes" })).toBeVisible();
    expect(screen.queryByRole("radio")).not.toBeInTheDocument();
  });

  it("shows incomplete repository results without a spend-per-hour figure", async () => {
    const warning = "Incomplete report: could not read org/unavailable. Spend-per-hour figures are unavailable.";
    vi.mocked(apiClient.get).mockImplementation((path: string) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(settings);
      if (path === "/roi-calculator/report") {
        return Promise.resolve({
          report: {
            ...summary,
            warnings: [warning],
            metrics: { ...summary.metrics, cost_per_hour: null, hours_per_dollar: null },
            people: summary.people.map((person) => ({ ...person, cost_per_hour: null })),
          },
        });
      }
      return Promise.resolve(idleStatus);
    });

    render(<ROICalculatorView accessToken="token" />);

    expect(await screen.findByRole("alert")).toHaveTextContent(warning);
    expect(screen.getByRole("button", { name: "Open estimate for org/repo pull request 42" })).toBeInTheDocument();
    expect(screen.queryByText("$3.00")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "People" }));
    fireEvent.click(screen.getByText("How this is calculated"));
    expect(
      screen.getByText("Spend per estimated hour is unavailable until all selected repositories can be read."),
    ).toBeVisible();
  });

  it("lets a view-only admin read the report without write controls", async () => {
    const runningStatus = {
      ...idleStatus,
      running: true,
      phase: "estimating",
      stage: "Estimating pull requests",
      total: 1,
    };
    vi.mocked(apiClient.get).mockImplementation((path: string) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(settings);
      if (path === "/roi-calculator/report") return Promise.resolve({ report: summary });
      return Promise.resolve(runningStatus);
    });

    render(<ROICalculatorView accessToken="token" userRole="Admin" isViewOnly />);

    expect(await screen.findByText("Gateway AI cost")).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent("Read-only access");
    expect(screen.queryByRole("button", { name: "Run analysis" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel sync" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "People" }));
    expect(screen.getByText("alice-work")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "alice-work" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
    expect(screen.getByLabelText("GitHub token (optional for public repositories)")).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Save settings" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run analysis" })).not.toBeInTheDocument();
  });

  it("lets an admin open the people view and save a manual email match", async () => {
    render(<ROICalculatorView accessToken="token" />);

    fireEvent.click(await screen.findByRole("tab", { name: "People" }));
    fireEvent.click(await screen.findByRole("button", { name: "alice-work" }));
    fireEvent.change(screen.getByLabelText("Gateway email"), {
      target: { value: "alice+work@example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save match" }));

    await waitFor(() =>
      expect(apiClient.put).toHaveBeenCalledWith("/roi-calculator/identity-map", {
        accessToken: "token",
        body: { github_login: "alice-work", email: "alice+work@example.com" },
      }),
    );
  });

  it("presents onboarding settings once when no report exists", async () => {
    const emptySettings = { ...settings, has_github_token: false, ready: false, repos: [], estimator_model: "" };
    vi.mocked(apiClient.get).mockImplementation((path: string) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(emptySettings);
      if (path === "/roi-calculator/report") return Promise.resolve({ report: null });
      return Promise.resolve(idleStatus);
    });

    render(<ROICalculatorView accessToken="token" />);

    expect(await screen.findByRole("heading", { name: "Connect your repositories" })).toBeInTheDocument();
    expect(screen.getByLabelText("GitHub token (optional for public repositories)")).toHaveAttribute(
      "type",
      "password",
    );
    expect(screen.getAllByText("Connect your repositories")).toHaveLength(1);
  });

  it.each(["github", "gitlab"])("only permits unauthenticated repository browsing for GitLab: %s", async (provider) => {
    const publicSettings = {
      ...settings,
      source_provider: provider,
      gitlab_api_url: "https://gitlab.com/api/v4",
      has_github_token: false,
      has_gitlab_token: false,
    };
    vi.mocked(apiClient.get).mockImplementation((path: string) => {
      if (path === "/roi-calculator/settings") {
        return Promise.resolve(publicSettings);
      }
      if (path === "/roi-calculator/report") return Promise.resolve({ report: summary });
      return Promise.resolve(idleStatus);
    });
    render(<ROICalculatorView accessToken="token" />);
    fireEvent.click(await screen.findByRole("button", { name: "Settings" }));
    const load = screen.getByRole("button", { name: "Load repositories" });
    if (provider === "github") expect(load).toBeDisabled();
    else expect(load).toBeEnabled();
    expect(screen.getByRole("textbox", { name: "Repository name" })).toBeEnabled();
  });

  it("closes the old settings dialog when saving a different source", async () => {
    const gitlabSettings = {
      ...settings,
      source_provider: "gitlab",
      gitlab_api_url: "https://gitlab.com/api/v4",
      has_gitlab_token: false,
      repos: [],
      ready: false,
    };
    vi.mocked(apiClient.put).mockResolvedValue(gitlabSettings);
    render(<ROICalculatorView accessToken="token" />);
    fireEvent.click(await screen.findByRole("button", { name: "Settings" }));
    fireEvent.change(screen.getByLabelText("Repository source"), { target: { value: "gitlab" } });
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
    expect(await screen.findByRole("heading", { name: "Connect your repositories" })).toBeVisible();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.getAllByLabelText("Repository source")).toHaveLength(1);
    expect(screen.getByLabelText("Repository source")).toHaveValue("gitlab");
  });

  it("keeps a running analysis visible when a source change finishes saving", async () => {
    const gitlabSettings = { ...settings, source_provider: "gitlab", repos: ["group/project"] };
    const runningStatus = { ...idleStatus, running: true, phase: "estimating", total: 1 };
    const saveRequest = Promise.withResolvers<typeof gitlabSettings>();
    vi.mocked(apiClient.put).mockReturnValue(saveRequest.promise);
    render(<ROICalculatorView accessToken="token" />);
    fireEvent.click(await screen.findByRole("button", { name: "Settings" }));
    fireEvent.change(screen.getByLabelText("Repository source"), { target: { value: "gitlab" } });
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
    vi.mocked(apiClient.get).mockResolvedValue(runningStatus);
    expect(await screen.findByRole("progressbar", { hidden: true }, { timeout: 3000 })).toBeInTheDocument();

    saveRequest.resolve(gitlabSettings);
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(screen.getByRole("progressbar", { name: "Sync progress" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Cancel sync" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Run analysis" })).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Connect your repositories" })).not.toBeInTheDocument();
    expect(apiClient.post).not.toHaveBeenCalled();
  });

  it.each(["success", "failure"])("ignores an old report refresh %s after switching sources", async (outcome) => {
    const gitlabSettings = { ...settings, source_provider: "gitlab", repos: [], ready: false };
    const oldRequest = Promise.withResolvers<{ report: typeof summary }>();
    const complete = { ...idleStatus, phase: "complete", finished_at: "2026-09-30T12:00:00Z" };
    vi.mocked(apiClient.put).mockResolvedValue(gitlabSettings);
    render(<ROICalculatorView accessToken="token" />);
    fireEvent.click(await screen.findByRole("button", { name: "Settings" }));
    vi.mocked(apiClient.get)
      .mockClear()
      .mockImplementation((path: string) =>
        path === "/roi-calculator/report" ? oldRequest.promise : Promise.resolve(complete),
      );
    await waitFor(
      () => expect(apiClient.get).toHaveBeenCalledWith("/roi-calculator/report", { accessToken: "token" }),
      {
        timeout: 3000,
      },
    );
    fireEvent.change(screen.getByLabelText("Repository source"), { target: { value: "gitlab" } });
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
    expect(await screen.findByRole("heading", { name: "Connect your repositories" })).toBeVisible();

    await act(async () => {
      if (outcome === "success") oldRequest.resolve({ report: summary });
      else oldRequest.reject(new Error("The old source is unavailable"));
    });
    expect(screen.getByRole("heading", { name: "Connect your repositories" })).toBeVisible();
    expect(screen.getByLabelText("Repository source")).toHaveValue("gitlab");
    expect(screen.queryByText("Improve request routing")).not.toBeInTheDocument();
    expect(screen.queryByText("The old source is unavailable")).not.toBeInTheDocument();

    const nextReport = { ...summary, pulls: [{ ...summary.pulls[0], title: "New source merge request" }] };
    vi.mocked(apiClient.get).mockImplementation((path: string) =>
      Promise.resolve(
        path === "/roi-calculator/report"
          ? { report: nextReport }
          : { ...complete, finished_at: "2026-09-30T13:00:00Z" },
      ),
    );
    expect(await screen.findByText("New source merge request", {}, { timeout: 3000 })).toBeVisible();
    expect(screen.queryByText("Improve request routing")).not.toBeInTheDocument();
  });

  it("clearly identifies the sample report and returns to setup when exiting", async () => {
    const emptySettings = { ...settings, has_github_token: false, ready: false, repos: [], estimator_model: "" };
    vi.mocked(apiClient.get).mockImplementation((path: string, options) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(emptySettings);
      if (path === "/roi-calculator/report") {
        return Promise.resolve({ report: options?.query?.mode === "demo" ? { ...summary, mode: "demo" } : null });
      }
      return Promise.resolve(idleStatus);
    });

    render(<ROICalculatorView accessToken="token" />);
    fireEvent.click(await screen.findByRole("button", { name: "Preview sample report" }));

    expect(await screen.findByText("You’re viewing demo data")).toBeVisible();
    expect(screen.getByRole("tab", { name: "Branches" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Cost / estimated hour")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Run analysis" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Settings" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Exit demo" }));

    expect(screen.getByRole("heading", { name: "Connect your repositories" })).toBeVisible();
    expect(screen.queryByText("You’re viewing demo data")).not.toBeInTheDocument();
    expect(apiClient.post).not.toHaveBeenCalled();
    expect(apiClient.put).not.toHaveBeenCalled();
  });

  it("opens sample PR costs from a live report and restores the live data on exit", async () => {
    const samplePull = {
      ...summary.pulls[0],
      title: "Sample usage breakdown",
      source_repo: "github.com/org/repo",
      source_branch: "feature/usage",
      branch_cost: {
        status: "matched",
        spend: 9.1,
        requests: 75,
        repo: "github.com/org/repo",
        branch: "feature/usage",
      },
    };
    vi.mocked(apiClient.get).mockImplementation((path: string, options) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(settings);
      if (path === "/roi-calculator/report") {
        return Promise.resolve({
          report: options?.query?.mode === "demo" ? { ...summary, mode: "demo", pulls: [samplePull] } : summary,
        });
      }
      return Promise.resolve(idleStatus);
    });

    render(<ROICalculatorView accessToken="token" />);
    fireEvent.click(await screen.findByRole("tab", { name: "Branches" }));
    fireEvent.change(screen.getByRole("searchbox"), { target: { value: "no matching PR" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview sample report" }));

    expect(await screen.findByText("You’re viewing demo data")).toBeVisible();
    expect(window.location.search).toBe("?demo=1");
    expect(screen.getByRole("tab", { name: "Branches" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("searchbox")).toHaveValue("");
    expect(screen.getByRole("cell", { name: "$9.10" })).toBeVisible();
    expect(screen.queryByText("Improve request routing")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run analysis" })).not.toBeInTheDocument();

    const runningStatus = { ...idleStatus, running: true, phase: "estimating", total: 1 };
    vi.mocked(apiClient.get).mockClear().mockResolvedValue(runningStatus);
    await waitFor(() => expect(apiClient.get).toHaveBeenCalledWith("/roi-calculator/sync", { accessToken: "token" }), {
      timeout: 3000,
    });
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Open estimate for org/repo pull request 42" }));
    expect(await screen.findByRole("dialog")).toBeVisible();
    expect(screen.getByText("75 requests")).toBeVisible();
    expect(screen.getByText(/branch:feature\/usage/)).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    fireEvent.click(screen.getByRole("button", { name: "Exit demo" }));

    expect(screen.getByText("Improve request routing")).toBeVisible();
    expect(screen.queryByText("Sample usage breakdown")).not.toBeInTheDocument();
    expect(window.location.search).toBe("");
    expect(screen.getByRole("button", { name: "Syncing…" })).toBeDisabled();
    expect(apiClient.post).not.toHaveBeenCalled();
    expect(apiClient.put).not.toHaveBeenCalled();
  });

  it("opens a demo link with sample data even while live analysis is running", async () => {
    window.history.replaceState(null, "", "/roi-calculator/?demo=1");
    const demoSummary = { ...summary, mode: "demo", metrics: { ...summary.metrics, total_spend: 38.4 } };
    const runningStatus = { ...idleStatus, running: true, phase: "estimating", total: 1 };
    vi.mocked(apiClient.get).mockImplementation((path: string, options) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(settings);
      if (path === "/roi-calculator/report") {
        return Promise.resolve({ report: options?.query?.mode === "demo" ? demoSummary : summary });
      }
      return Promise.resolve(runningStatus);
    });
    render(<ROICalculatorView accessToken="token" />);
    expect(await screen.findByText("You’re viewing demo data")).toBeVisible();
    expect(screen.getByText("$38.40")).toBeVisible();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run analysis" })).not.toBeInTheDocument();
    expect(apiClient.post).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Exit demo" }));
    expect(screen.getByRole("progressbar")).toBeVisible();
    expect(screen.getByText("$20.00")).toBeVisible();
    expect(window.location.search).toBe("");
  });

  it.each(["report", "sync"])("loads a demo link when the live %s request fails", async (failedRequest) => {
    window.history.replaceState(null, "", "/roi-calculator/?demo=1");
    vi.mocked(apiClient.get).mockImplementation((path: string, options) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(settings);
      if (options?.query?.mode === "demo") return Promise.resolve({ report: { ...summary, mode: "demo" } });
      if (path === `/roi-calculator/${failedRequest}`) return Promise.reject(new Error("Live data unavailable"));
      if (path === "/roi-calculator/report") return Promise.resolve({ report: summary });
      return Promise.resolve(idleStatus);
    });
    render(<ROICalculatorView accessToken="token" />);
    expect(await screen.findByText("You’re viewing demo data")).toBeVisible();
    expect(screen.getByText("Gateway AI cost")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(apiClient.post).not.toHaveBeenCalled();
  });

  it("shows the demo without waiting for a stalled live request", async () => {
    window.history.replaceState(null, "", "/roi-calculator/?demo=1");
    vi.mocked(apiClient.get).mockImplementation((path: string, options) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(settings);
      if (options?.query?.mode === "demo") return Promise.resolve({ report: { ...summary, mode: "demo" } });
      return new Promise(() => {});
    });
    render(<ROICalculatorView accessToken="token" />);
    expect(await screen.findByText("You’re viewing demo data")).toBeVisible();
    expect(screen.getByText("Gateway AI cost")).toBeVisible();
  });

  it("keeps the live calculator usable when a demo link cannot load sample data", async () => {
    window.history.replaceState(null, "", "/roi-calculator/?demo=1&from=review#overview");
    vi.mocked(apiClient.get).mockImplementation((path: string, options) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(settings);
      if (options?.query?.mode === "demo") return Promise.reject(new Error("Sample data unavailable"));
      if (path === "/roi-calculator/report") return Promise.resolve({ report: summary });
      return Promise.resolve(idleStatus);
    });
    render(<ROICalculatorView accessToken="token" />);
    expect(await screen.findByText("Gateway AI cost")).toBeVisible();
    expect(screen.getByText("$20.00")).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent("Sample data unavailable");
    expect(screen.getByRole("button", { name: "Settings" })).toBeEnabled();
    expect(screen.queryByText("You’re viewing demo data")).not.toBeInTheDocument();
    expect(apiClient.post).not.toHaveBeenCalled();
    expect(window.location.search).toBe("?from=review");
    expect(window.location.hash).toBe("#overview");
  });

  it.each(["report", "sync"])("waits for the live %s when exiting a demo", async (pendingRequest) => {
    window.history.replaceState(null, "", "/roi-calculator/?demo=1");
    const pending = Promise.withResolvers<{ report: typeof summary } | typeof idleStatus>();
    vi.mocked(apiClient.get).mockImplementation((path: string, options) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(settings);
      if (options?.query?.mode === "demo") return Promise.resolve({ report: { ...summary, mode: "demo" } });
      if (path === `/roi-calculator/${pendingRequest}`) return pending.promise;
      if (path === "/roi-calculator/report") return Promise.resolve({ report: summary });
      return Promise.resolve(idleStatus);
    });
    render(<ROICalculatorView accessToken="token" />);
    fireEvent.click(await screen.findByRole("button", { name: "Exit demo" }));
    expect(screen.getByText("Loading ROI Calculator…")).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Connect your repositories" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run analysis" })).not.toBeInTheDocument();
    expect(window.location.search).toBe("");

    pending.resolve(pendingRequest === "report" ? { report: summary } : idleStatus);
    expect(await screen.findByText("Gateway AI cost")).toBeVisible();
    expect(screen.getByText("$20.00")).toBeVisible();
    expect(screen.queryByText("Loading ROI Calculator…")).not.toBeInTheDocument();
    expect(apiClient.post).not.toHaveBeenCalled();
  });

  it.each(["report", "demo"])("retains a failed %s load after a successful sync poll", async (failedRequest) => {
    if (failedRequest === "demo") window.history.replaceState(null, "", "/roi-calculator/?demo=1");
    const message = `The ${failedRequest} is unavailable`;
    vi.mocked(apiClient.get).mockImplementation((path: string, options) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(settings);
      if (options?.query?.mode === "demo") return Promise.reject(new Error(message));
      if (path === "/roi-calculator/report") {
        return failedRequest === "report" ? Promise.reject(new Error(message)) : Promise.resolve({ report: summary });
      }
      return Promise.resolve(idleStatus);
    });
    render(<ROICalculatorView accessToken="token" />);
    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    const running = { ...idleStatus, running: true, phase: "estimating", total: 1 };
    vi.mocked(apiClient.get).mockImplementation((path: string) => {
      if (path === "/roi-calculator/report") return Promise.reject(new Error(message));
      return Promise.resolve(running);
    });
    expect(await screen.findByRole("progressbar", { name: "Sync progress" }, { timeout: 3000 })).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent(message);
  });

  it("retries a failed initial report and clears its error only when the report recovers", async () => {
    vi.mocked(apiClient.get).mockImplementation((path: string) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(settings);
      if (path === "/roi-calculator/report") return Promise.reject(new Error("Report unavailable"));
      return Promise.resolve(idleStatus);
    });
    render(<ROICalculatorView accessToken="token" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Report unavailable");
    expect(screen.getByRole("button", { name: "Settings" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Run analysis" })).toBeEnabled();
    vi.mocked(apiClient.get).mockImplementation((path: string) =>
      Promise.resolve(path === "/roi-calculator/report" ? { report: summary } : idleStatus),
    );
    expect(await screen.findByText("Gateway AI cost", {}, { timeout: 3000 })).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Connect your repositories" })).not.toBeInTheDocument();
  });

  it("returns to Overview and shows the last sync time when completion is polled from Settings", async () => {
    const runningStatus = {
      ...idleStatus,
      running: true,
      phase: "estimating",
      stage: "Estimating pull requests",
      total: 1,
    };
    const completedStatus = { ...idleStatus, phase: "complete", done: 57, total: 57, reused: 57 };
    vi.mocked(apiClient.get)
      .mockResolvedValueOnce(settings)
      .mockResolvedValueOnce({ report: null })
      .mockResolvedValueOnce(runningStatus)
      .mockResolvedValueOnce(completedStatus)
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            window.setTimeout(() => resolve({ report: summary }), 25);
          }),
      );

    render(<ROICalculatorView accessToken="token" />);

    expect(await screen.findByRole("progressbar", { name: "Sync progress" })).toBeInTheDocument();
    expect(await screen.findByText("Gateway AI cost", {}, { timeout: 5000 })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Connect your repositories" })).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Last synced Sep 30, 2026, 12:00 PM UTC");
  });

  it("shows the sync error returned by the status endpoint", async () => {
    const runningStatus = {
      ...idleStatus,
      running: true,
      phase: "estimating",
      stage: "Estimating pull requests",
      total: 1,
    };
    const errorStatus = {
      ...idleStatus,
      phase: "error",
      error: "The estimator could not score a pull request.",
    };
    vi.mocked(apiClient.get)
      .mockResolvedValueOnce(settings)
      .mockResolvedValueOnce({ report: null })
      .mockResolvedValueOnce(runningStatus)
      .mockResolvedValueOnce(errorStatus);

    render(<ROICalculatorView accessToken="token" />);

    expect(await screen.findByRole("alert", {}, { timeout: 5000 })).toHaveTextContent(
      "The estimator could not score a pull request.",
    );
    expect(screen.getByText("Sync failed")).toBeInTheDocument();
  });

  it("shows a report error and ends progress when the completed report cannot load", async () => {
    const runningStatus = {
      ...idleStatus,
      running: true,
      phase: "estimating",
      stage: "Estimating pull requests",
      total: 1,
    };
    const completedStatus = { ...idleStatus, phase: "complete", done: 1, total: 1 };
    vi.mocked(apiClient.get)
      .mockResolvedValueOnce(settings)
      .mockResolvedValueOnce({ report: null })
      .mockResolvedValueOnce(runningStatus)
      .mockResolvedValueOnce(completedStatus)
      .mockRejectedValueOnce(new Error("The report could not be loaded."));

    render(<ROICalculatorView accessToken="token" />);

    expect(await screen.findByRole("progressbar", { name: "Sync progress" })).toBeInTheDocument();
    expect(await screen.findByRole("alert", {}, { timeout: 5000 })).toHaveTextContent(
      "The report could not be loaded.",
    );
    expect(screen.queryByRole("progressbar", { name: "Sync progress" })).not.toBeInTheDocument();
  });

  it("clears a transient poll error when the next poll completes and loads the report", async () => {
    const runningStatus = {
      ...idleStatus,
      running: true,
      phase: "estimating",
      stage: "Estimating pull requests",
      total: 1,
    };
    const completedStatus = { ...idleStatus, phase: "complete", done: 1, total: 1 };
    vi.mocked(apiClient.get)
      .mockResolvedValueOnce(settings)
      .mockResolvedValueOnce({ report: null })
      .mockResolvedValueOnce(runningStatus)
      .mockRejectedValueOnce(new Error("The sync status could not be loaded."))
      .mockResolvedValueOnce(completedStatus)
      .mockResolvedValueOnce({ report: summary });

    render(<ROICalculatorView accessToken="token" />);

    expect(await screen.findByRole("progressbar", { name: "Sync progress" })).toBeInTheDocument();
    expect(await screen.findByRole("alert", {}, { timeout: 5000 })).toHaveTextContent(
      "The sync status could not be loaded.",
    );
    expect(await screen.findByText("Gateway AI cost", {}, { timeout: 7000 })).toBeInTheDocument();
    expect(screen.queryByText("The sync status could not be loaded.")).not.toBeInTheDocument();
  });
  it("saves the edited schedule before running from Settings", async () => {
    vi.mocked(apiClient.put).mockResolvedValue(settings);
    vi.mocked(apiClient.post).mockResolvedValue({ ...idleStatus, running: true });
    render(<ROICalculatorView accessToken="token" />);
    fireEvent.click(await screen.findByRole("button", { name: "Settings" }));
    fireEvent.change(screen.getByLabelText("Update interval (hours)"), { target: { value: "6" } });
    fireEvent.click(screen.getByRole("button", { name: "Save and run analysis" }));
    await waitFor(() => expect(apiClient.post).toHaveBeenCalledWith("/roi-calculator/sync", { accessToken: "token" }));
    expect(apiClient.put).toHaveBeenCalledWith(
      "/roi-calculator/settings",
      expect.objectContaining({
        body: expect.objectContaining({ update_interval_minutes: 360, estimator_model: "estimator" }),
      }),
    );
    expect(vi.mocked(apiClient.put).mock.invocationCallOrder[0]).toBeLessThan(
      vi.mocked(apiClient.post).mock.invocationCallOrder[0],
    );
  });
});
