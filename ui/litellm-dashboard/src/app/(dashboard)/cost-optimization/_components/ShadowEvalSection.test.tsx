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

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: vi.fn(() => ({
    data: {
      keys: [
        { token: "hash-alpha", token_id: "id-1", key_name: "sk-...alpha", key_alias: "prod-alpha" },
        { token: "hash-beta", token_id: "id-2", key_name: "sk-...beta", key_alias: "staging-beta" },
      ],
      total_count: 2,
      current_page: 1,
      total_pages: 1,
    },
    isPending: false,
  })),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAutoRouters: vi.fn(() => ({
    data: [
      { model_name: "claude-auto", litellm_params: { model: "auto_router/claude-auto" } },
      { model_name: "gpt-auto", litellm_params: { model: "auto_router/gpt-auto" } },
    ],
  })),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({
  useModelCostMap: vi.fn(() => ({
    data: {
      "claude-sonnet-5": { litellm_provider: "anthropic", mode: "chat" },
      "gpt-4o": { litellm_provider: "openai", mode: "chat" },
      "gemini/gemini-2.5-pro": { litellm_provider: "gemini", mode: "chat" },
      "gpt-4o-mini": { litellm_provider: "openai", mode: "chat" },
      "text-embedding-3-large": { litellm_provider: "openai", mode: "embedding" },
    },
  })),
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
  ends_at: null,
  completed_at: null,
  api_key_id: "hashed-key-abc",
  team_id: null,
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
  vi.mocked(useShadowEvalJob).mockImplementation((_token, jobId) => {
    let data = detail;
    if (detailsById) {
      data = jobId ? detailsById[jobId] : undefined;
    }
    return { data } as unknown as ReturnType<typeof useShadowEvalJob>;
  });
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

  it("starts a job from picked key + router with duration, submitting the key token not its alias", async () => {
    const user = userEvent.setup();
    const mutate = vi.fn();
    mockHooks({ jobs: [] });
    vi.mocked(useStartShadowEval).mockReturnValue({ mutate, isPending: false, error: null } as unknown as ReturnType<
      typeof useStartShadowEval
    >);
    render(<ShadowEvalSection accessToken="token" />);

    await user.click(screen.getByPlaceholderText("Search keys by alias"));
    await user.click(await screen.findByText("prod-alpha"));
    await user.click(screen.getByPlaceholderText("Select an auto-router"));
    await user.click(await screen.findByText("gpt-auto"));
    await user.click(screen.getByPlaceholderText("Select a judge model"));
    await user.click(await screen.findByRole("option", { name: /anthropic\/claude-sonnet-5/ }));
    await user.click(screen.getByText("Start shadow eval"));

    const expectedBody = {
      api_key_id: "hash-alpha",
      router_name: "gpt-auto",
      shadow_percentage: 10,
      duration_days: 7,
      judge_model: "anthropic/claude-sonnet-5",
    };
    expect(mutate).toHaveBeenCalledWith({ body: expectedBody });
  });

  it("filters the key list as you type via the server-side alias search", async () => {
    const user = userEvent.setup();
    const { useKeys } = await import("@/app/(dashboard)/hooks/keys/useKeys");
    mockHooks({ jobs: [] });
    render(<ShadowEvalSection accessToken="token" />);

    await user.type(screen.getByPlaceholderText("Search keys by alias"), "prod");

    await vi.waitFor(() => {
      const calls = vi.mocked(useKeys).mock.calls;
      expect(calls[calls.length - 1][2]).toMatchObject({ selectedKeyAlias: "prod" });
    });
  });

  it("explains what the judge model is for and recommends models across providers", () => {
    mockHooks({ jobs: [] });
    render(<ShadowEvalSection accessToken="token" />);
    expect(screen.getByText("Judge model")).toBeInTheDocument();
    expect(screen.getByText("anthropic/claude-sonnet-5")).toBeInTheDocument();
    expect(screen.getByText("openai/gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("gemini/gemini-2.5-pro")).toBeInTheDocument();
  });

  it("does not send a default judge model without an explicit pick", async () => {
    const user = userEvent.setup();
    const mutate = vi.fn();
    mockHooks({ jobs: [] });
    vi.mocked(useStartShadowEval).mockReturnValue({ mutate, isPending: false, error: null } as unknown as ReturnType<
      typeof useStartShadowEval
    >);
    render(<ShadowEvalSection accessToken="token" />);

    await user.click(screen.getByPlaceholderText("Search keys by alias"));
    await user.click(await screen.findByText("prod-alpha"));
    await user.click(screen.getByPlaceholderText("Select an auto-router"));
    await user.click(await screen.findByText("gpt-auto"));

    expect(screen.getByText("Start shadow eval")).toBeDisabled();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("offers the recommended judge models ahead of the rest of the catalog", async () => {
    const user = userEvent.setup();
    mockHooks({ jobs: [] });
    render(<ShadowEvalSection accessToken="token" />);

    await user.click(screen.getByPlaceholderText("Select a judge model"));

    const recommended = await screen.findAllByText("Recommended");
    expect(recommended).toHaveLength(3);
  });

  it("shows when an active job will end", () => {
    const j = job({ ends_at: new Date(Date.now() + 3 * 86_400_000).toISOString() });
    mockHooks({ jobs: [j], detail: j });
    render(<ShadowEvalSection accessToken="token" />);
    expect(screen.getByText(/ends in 3 days/)).toBeInTheDocument();
  });

  it("keeps an older populated job reachable when the newest job has no verdicts", async () => {
    const user = userEvent.setup();
    const emptyOverrides: Partial<ShadowEvalJob> = {
      job_id: "job-new",
      status: "pending",
      completed_count: 0,
      failed_count: 0,
      results: null,
      cost_actual: 0,
    };
    const empty = job(emptyOverrides);
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

  it("labels an unexpanded previous job as viewable, not verdictless", async () => {
    const user = userEvent.setup();
    const current = job({ job_id: "job-new" });
    const older = job({ job_id: "job-old", status: "completed", results: null });
    mockHooks({ jobs: [current, older], detailsById: { "job-new": current } });
    render(<ShadowEvalSection accessToken="token" />);

    await user.click(screen.getByRole("button", { name: /Previous evaluations \(1\)/ }));

    expect(screen.getByText("view results")).toBeInTheDocument();
    expect(screen.queryByText("no verdicts")).not.toBeInTheDocument();
  });
});
