import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

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
vi.mock("@/components/ui/chart", () => ({
  ChartContainer: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  ChartLegend: () => null,
  ChartLegendContent: () => null,
  ChartTooltip: () => null,
  ChartTooltipContent: () => null,
}));
vi.mock("recharts", () => ({
  Bar: () => null,
  CartesianGrid: () => null,
  ComposedChart: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
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
    vi.mocked(apiClient.get).mockReset();
    vi.mocked(apiClient.put).mockReset();
    vi.mocked(apiClient.get).mockImplementation((path: string) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(settings);
      if (path === "/roi-calculator/report") return Promise.resolve({ report: summary });
      return Promise.resolve(idleStatus);
    });
    vi.mocked(apiClient.put).mockResolvedValue({ report: summary, identity_map: { alice: "alice@example.com" } });
  });

  it("shows the spend summary and opens an accessible pull reasoning dialog", async () => {
    render(<ROICalculatorView accessToken="token" />);

    expect(await screen.findByText("Spend per estimated engineering hour")).toBeInTheDocument();
    expect(screen.getByText("$3.00")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open estimate for org/repo pull request 42" }));

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Updated routing and added a regression test.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "View on GitHub" })).toHaveAttribute(
      "href",
      "https://github.com/org/repo/pull/42",
    );
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

    expect(await screen.findByText("Spend per estimated engineering hour")).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent("Read-only access");
    expect(screen.queryByRole("button", { name: "Run analysis" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel sync" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "People" }));
    expect(screen.getByText("alice-work")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "alice-work" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Settings" }));
    expect(screen.getByLabelText("GitHub token")).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Save settings" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run analysis" })).not.toBeInTheDocument();
  });

  it("lets an admin open the people view and save a manual email match", async () => {
    render(<ROICalculatorView accessToken="token" />);

    fireEvent.click(await screen.findByRole("tab", { name: "People" }));
    fireEvent.click(await screen.findByRole("button", { name: "alice-work" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Gateway email" }), {
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
    vi.mocked(apiClient.get).mockImplementation((path: string) => {
      if (path === "/roi-calculator/settings") return Promise.resolve(settings);
      if (path === "/roi-calculator/report") return Promise.resolve({ report: null });
      return Promise.resolve(idleStatus);
    });

    render(<ROICalculatorView accessToken="token" />);

    expect(await screen.findByRole("heading", { name: "Connect GitHub to get started" })).toBeInTheDocument();
    expect(screen.getByLabelText("GitHub token")).toHaveAttribute("type", "password");
    expect(screen.getAllByText("Connect GitHub to get started")).toHaveLength(1);
  });

  it("returns to Overview and shows the last sync time when completion is polled from Settings", async () => {
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
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            window.setTimeout(() => resolve({ report: summary }), 25);
          }),
      );

    render(<ROICalculatorView accessToken="token" />);

    expect(await screen.findByRole("progressbar", { name: "Sync progress" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Settings" }));
    expect(await screen.findByText("Spend per estimated engineering hour", {}, { timeout: 5000 })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Connect GitHub to get started" })).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Up to date · Last synced Sep 30, 2026, 12:00 PM UTC");
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
});
