import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";

vi.mock("./useShadowEval", () => ({
  useShadowEvalJobs: vi.fn(),
  useShadowEvalJob: vi.fn(),
  useStartShadowEval: vi.fn(),
  useStopShadowEval: vi.fn(),
}));

import ShadowEvalSection from "./ShadowEvalSection";
import {
  useShadowEvalJob,
  useShadowEvalJobs,
  useStartShadowEval,
  useStopShadowEval,
  type ShadowEvalJob,
} from "./useShadowEval";

const job = (overrides: Partial<ShadowEvalJob> = {}): ShadowEvalJob => ({
  job_id: "job-1",
  status: "running",
  router_name: "claude-auto",
  shadow_percentage: 10,
  request_count: 500,
  completed_count: 42,
  failed_count: 1,
  results: {
    groups: [
      {
        tier: "SIMPLE",
        turn_count: 30,
        real_win_rate_pct: 20.0,
        shadow_win_rate_pct: 55.0,
        tie_rate_pct: 25.0,
        avg_judge_confidence: 0.81,
      },
      {
        tier: "REASONING",
        turn_count: 12,
        real_win_rate_pct: 50.0,
        shadow_win_rate_pct: 33.3,
        tie_rate_pct: 16.7,
        avg_judge_confidence: 0.74,
      },
    ],
    overall_shadow_win_rate_pct: 48.0,
    overall_tie_rate_pct: 22.0,
  },
  cost_estimate: 45.0,
  cost_actual: 3.21,
  created_at: "2026-08-07T00:00:00Z",
  completed_at: null,
  ...overrides,
});

const mockHooks = ({
  jobs = [],
  detail = undefined,
  detailsById = undefined,
}: {
  jobs?: ShadowEvalJob[];
  detail?: ShadowEvalJob;
  detailsById?: Record<string, ShadowEvalJob>;
}) => {
  vi.mocked(useShadowEvalJobs).mockReturnValue({ data: jobs, error: null } as unknown as ReturnType<
    typeof useShadowEvalJobs
  >);
  vi.mocked(useShadowEvalJob).mockImplementation(
    (_token, jobId) =>
      ({
        data: detailsById ? (jobId ? detailsById[jobId] : undefined) : detail,
      }) as unknown as ReturnType<typeof useShadowEvalJob>,
  );
  vi.mocked(useStartShadowEval).mockReturnValue({
    mutate: vi.fn(),
    isPending: false,
    error: null,
  } as unknown as ReturnType<typeof useStartShadowEval>);
  vi.mocked(useStopShadowEval).mockReturnValue({ mutate: vi.fn(), isPending: false } as unknown as ReturnType<
    typeof useStopShadowEval
  >);
};

describe("ShadowEvalSection", () => {
  it("shows the start form when there are no jobs", () => {
    mockHooks({ jobs: [] });
    render(<ShadowEvalSection accessToken="token" />);
    expect(screen.getByText("Start a shadow eval")).toBeInTheDocument();
    expect(screen.getByText("Start shadow eval")).toBeInTheDocument();
  });

  it("renders per-tier win rates for an active job", () => {
    const j = job();
    mockHooks({ jobs: [j], detail: j });
    render(<ShadowEvalSection accessToken="token" />);
    expect(screen.getByText("SIMPLE")).toBeInTheDocument();
    expect(screen.getByText("REASONING")).toBeInTheDocument();
    expect(screen.getByText("55.0%")).toBeInTheDocument();
    // Overall good-or-better = shadow wins + ties = 70%
    expect(screen.getByText("70.0%")).toBeInTheDocument();
  });

  it("flags low-sample tiers", () => {
    const j = job();
    mockHooks({ jobs: [j], detail: j });
    render(<ShadowEvalSection accessToken="token" />);
    // REASONING tier has 12 turns < 30
    expect(screen.getByText("(low sample)")).toBeInTheDocument();
  });

  it("shows a stop button for running jobs but not completed ones", () => {
    const running = job({ status: "running" });
    mockHooks({ jobs: [running], detail: running });
    const { unmount } = render(<ShadowEvalSection accessToken="token" />);
    expect(screen.getByText("Stop")).toBeInTheDocument();
    unmount();

    const done = job({ status: "completed" });
    mockHooks({ jobs: [done], detail: done });
    render(<ShadowEvalSection accessToken="token" />);
    expect(screen.queryByText("Stop")).not.toBeInTheDocument();
  });

  it("offers the start form again once the latest job is completed", () => {
    const done = job({ status: "completed" });
    mockHooks({ jobs: [done], detail: done });
    render(<ShadowEvalSection accessToken="token" />);
    expect(screen.getByText("Start a shadow eval")).toBeInTheDocument();
  });

  it("keeps an older populated job reachable when the newest job has no verdicts", async () => {
    const user = userEvent.setup();
    const empty = job({
      job_id: "job-new",
      status: "pending",
      completed_count: 0,
      failed_count: 0,
      results: null,
      cost_actual: 0,
    });
    const older = job({ job_id: "job-old", status: "completed" });
    mockHooks({
      jobs: [empty, older],
      detailsById: { "job-new": empty, "job-old": older },
    });
    render(<ShadowEvalSection accessToken="token" />);

    // The newest job is empty, so its verdicts are absent from the primary card...
    expect(screen.queryByText("SIMPLE")).not.toBeInTheDocument();

    // ...but the older job with verdicts must still be reachable.
    await user.click(screen.getByRole("button", { name: /Previous evaluations \(1\)/ }));
    await user.click(screen.getByRole("button", { name: /job-old-row|10% via claude-auto/ }));

    expect(await screen.findByText("SIMPLE")).toBeInTheDocument();
    expect(screen.getByText("REASONING")).toBeInTheDocument();
  });
});
