import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useInfiniteKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { ApiError } from "@/lib/http/client";

vi.mock("./useShadowEval", () => ({
  useShadowEvalJobs: vi.fn(),
  useShadowEvalJob: vi.fn(),
  useStartShadowEval: vi.fn(),
  useStopShadowEval: vi.fn(),
}));

const authorizedRoleMock = vi.fn(() => ({ accessToken: "token", isViewOnly: false }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => authorizedRoleMock() }));

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useInfiniteKeys: vi.fn(() => ({
    data: {
      pages: [
        {
          keys: [
            { token: "hash-alpha", token_id: "id-1", key_name: "sk-...alpha", key_alias: "prod-alpha" },
            { token: "hash-beta", token_id: "id-2", key_name: "sk-...beta", key_alias: "staging-beta" },
          ],
          total_count: 2,
          current_page: 1,
          total_pages: 1,
        },
      ],
    },
    isPending: false,
    isError: false,
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
  })),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAutoRouters: vi.fn(() => ({
    data: [
      { model_name: "claude-auto", litellm_params: { model: "auto_router/claude-auto" } },
      { model_name: "gpt-auto", litellm_params: { model: "auto_router/gpt-auto" } },
    ],
  })),
  usePlainModelGroups: vi.fn(() => new Set(["prod-claude"])),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({
  useModelCostMap: vi.fn(() => ({
    data: {
      "claude-sonnet-5": { litellm_provider: "anthropic", mode: "chat" },
      "gpt-4o": { litellm_provider: "openai", mode: "chat" },
      "gemini/gemini-2.5-pro": { litellm_provider: "gemini", mode: "chat" },
      "text-embedding-3-large": { litellm_provider: "openai", mode: "embedding" },
    },
  })),
}));

import ShadowEvalSection, { shadowedKeyLabel } from "./ShadowEvalSection";
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
  direction: "forward",
  baseline_model: null,
  judge_model: "anthropic/claude-sonnet-5",
  shadow_percentage: 10,
  keys: [
    {
      api_key_id: "hashed-key-abc",
      max_turns: 10000,
      max_budget: 10,
      spend: 3.21,
      stopped_at: null,
      key_alias: "prod-alpha",
      key_name: "sk-...alpha",
    },
  ],
  judged_count: 42,
  error_count: 1,
  judge_spend: 3.21,
  results: {
    by_tier: [
      {
        group: "SIMPLE",
        turn_count: 30,
        real_win_rate_pct: 20.0,
        shadow_win_rate_pct: 55.0,
        tie_rate_pct: 25.0,
        avg_judge_confidence: 0.81,
      },
      {
        group: "REASONING",
        turn_count: 12,
        real_win_rate_pct: 50.0,
        shadow_win_rate_pct: 33.3,
        tie_rate_pct: 16.7,
        avg_judge_confidence: 0.74,
      },
    ],
    by_current_model: [
      {
        group: "gpt-4o",
        turn_count: 42,
        real_win_rate_pct: 30.0,
        shadow_win_rate_pct: 45.0,
        tie_rate_pct: 25.0,
        avg_judge_confidence: 0.8,
      },
    ],
    by_key: [],
    overall_shadow_win_rate_pct: 48.0,
    overall_tie_rate_pct: 22.0,
  },
  created_at: "2026-08-07T00:00:00Z",
  ends_at: "2026-09-07T00:00:00Z",
  last_error: null,
  ...overrides,
});

const keyEntry = (
  api_key_id: string,
  overrides: Partial<ShadowEvalJob["keys"][number]> = {},
): ShadowEvalJob["keys"][number] => ({
  api_key_id,
  max_turns: 10000,
  max_budget: 10,
  spend: 0,
  stopped_at: null,
  attempt_count: null,
  key_alias: null,
  key_name: null,
  ...overrides,
});

const mockHooks = ({
  jobs = [],
  detailsById = {},
  error = null,
  detailError = false,
  isPending = false,
}: {
  jobs?: ShadowEvalJob[];
  detailsById?: Record<string, ShadowEvalJob>;
  error?: Error | null;
  detailError?: boolean;
  isPending?: boolean;
}) => {
  vi.mocked(useShadowEvalJobs).mockReturnValue({
    data: error || isPending ? undefined : jobs,
    error,
    isPending,
  } as unknown as ReturnType<typeof useShadowEvalJobs>);
  vi.mocked(useShadowEvalJob).mockImplementation(
    (jobId) =>
      ({
        data: jobId ? detailsById[jobId] : undefined,
        isError: detailError ?? false,
      }) as unknown as ReturnType<typeof useShadowEvalJob>,
  );
  const start = { mutate: vi.fn(), isPending: false };
  const stop = { mutate: vi.fn(), isPending: false };
  vi.mocked(useStartShadowEval).mockReturnValue(start as unknown as ReturnType<typeof useStartShadowEval>);
  vi.mocked(useStopShadowEval).mockReturnValue(stop as unknown as ReturnType<typeof useStopShadowEval>);
  return { start, stop };
};

describe("ShadowEvalSection", () => {
  beforeEach(() => {
    authorizedRoleMock.mockReturnValue({ accessToken: "token", isViewOnly: false });
  });

  it("shows a key picker load failure instead of posing as no matching keys", async () => {
    const user = userEvent.setup();
    const defaultKeysImpl = vi.mocked(useInfiniteKeys).getMockImplementation();
    vi.mocked(useInfiniteKeys).mockReturnValue({
      data: undefined,
      isPending: false,
      isError: true,
      fetchNextPage: vi.fn(),
      hasNextPage: false,
      isFetchingNextPage: false,
    } as unknown as ReturnType<typeof useInfiniteKeys>);
    mockHooks({});
    render(<ShadowEvalSection />);

    await user.click(screen.getByPlaceholderText("Search keys by alias"));
    expect(await screen.findByText("Keys could not be loaded. Refresh the page to retry.")).toBeInTheDocument();
    expect(screen.queryByText("No matching keys")).not.toBeInTheDocument();
    if (defaultKeysImpl) vi.mocked(useInfiniteKeys).mockImplementation(defaultKeysImpl);
  });

  it("offers the start form while the list is still loading", () => {
    mockHooks({ isPending: true });
    render(<ShadowEvalSection />);
    expect(screen.getByText("Loading evaluations...")).toBeInTheDocument();
    expect(screen.getByText("Start a shadow eval")).toBeInTheDocument();
  });

  it("re-offers the start form when the polled detail sees the job finish before the list does", () => {
    mockHooks({
      jobs: [job({ status: "running" })],
      detailsById: { "job-1": job({ status: "completed" }) },
    });
    render(<ShadowEvalSection />);
    expect(screen.getByText("Start a shadow eval")).toBeInTheDocument();
  });

  it("gives every active job its own card with a stop button, with the form still offered", () => {
    mockHooks({
      jobs: [
        job({ job_id: "job-a", status: "running", keys: [keyEntry("key-a")] }),
        job({ job_id: "job-b", status: "running", keys: [keyEntry("key-b")] }),
      ],
    });
    render(<ShadowEvalSection />);
    expect(screen.getAllByRole("button", { name: "Stop" })).toHaveLength(2);
    expect(screen.getByText("Start a shadow eval")).toBeInTheDocument();
    expect(screen.queryByText(/Previous evaluations/)).not.toBeInTheDocument();
  });

  it("renders the active card from the list row while its detail is still loading", () => {
    mockHooks({ jobs: [job({ status: "running" })], detailsById: {} });
    render(<ShadowEvalSection />);
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
  });

  it("hides the start form and stop button from view-only admins", () => {
    authorizedRoleMock.mockReturnValue({ accessToken: "token", isViewOnly: true });
    mockHooks({ jobs: [job({ status: "running" })] });
    render(<ShadowEvalSection />);
    expect(screen.queryByText("Start a shadow eval")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument();
    expect(screen.getByText("running")).toBeInTheDocument();
  });

  it("never labels a collapsed previous eval as empty from a countless list row", () => {
    const countlessListRow: Partial<ShadowEvalJob> = {
      job_id: "job-old",
      status: "stopped",
      judged_count: null,
      error_count: null,
      judge_spend: null,
      results: null,
    };
    mockHooks({ jobs: [job({ status: "running" }), job(countlessListRow)] });
    render(<ShadowEvalSection />);
    fireEvent.click(screen.getByRole("button", { name: /Previous evaluations/ }));
    expect(screen.getByText("view results")).toBeInTheDocument();
    expect(screen.queryByText("no verdicts")).not.toBeInTheDocument();
    expect(screen.queryByText(/0 judged/)).not.toBeInTheDocument();
  });

  it("surfaces a non-403 list failure instead of posing as an empty state", () => {
    mockHooks({ error: new Error("boom") });
    render(<ShadowEvalSection />);
    expect(screen.getByText(/Existing evaluations could not be loaded/)).toBeInTheDocument();
    expect(screen.getByText("Start a shadow eval")).toBeInTheDocument();
  });

  it("shows a failure line instead of loading forever when the detail fetch errors", () => {
    mockHooks({
      jobs: [job({ status: "completed", judged_count: 12, results: null })],
      detailsById: {},
      detailError: true,
    });
    render(<ShadowEvalSection />);
    expect(screen.getByText(/Results could not be loaded/)).toBeInTheDocument();
    expect(screen.queryByText("Loading results...")).not.toBeInTheDocument();
  });

  it("shows the failure line over the collecting copy when an active job's detail errors", () => {
    mockHooks({ jobs: [job({ status: "running", results: null })], detailsById: {}, detailError: true });
    render(<ShadowEvalSection />);
    expect(screen.getByText(/Results could not be loaded/)).toBeInTheDocument();
    expect(screen.queryByText(/Collecting verdicts/)).not.toBeInTheDocument();
  });

  it("never claims no verdicts for a judged job whose results have not loaded yet", () => {
    mockHooks({ jobs: [job({ status: "completed", judged_count: 12, results: null })], detailsById: {} });
    render(<ShadowEvalSection />);
    expect(screen.getByText("Loading results...")).toBeInTheDocument();
    expect(screen.queryByText(/No verdicts were recorded/)).not.toBeInTheDocument();
  });

  it("shows the start form when there are no jobs", () => {
    mockHooks({});
    render(<ShadowEvalSection />);
    expect(screen.getByText("Start a shadow eval")).toBeInTheDocument();
    expect(screen.getByText("Start shadow eval")).toBeInTheDocument();
  });

  it("renders the latest job's results with the headline stat, verdict split, and both stratifications", () => {
    const j = job();
    mockHooks({ jobs: [j], detailsById: { "job-1": j } });
    render(<ShadowEvalSection />);

    expect(screen.getByText("Router matched or beat your current model")).toBeInTheDocument();
    expect(screen.getByText("70.0%")).toBeInTheDocument();
    expect(screen.getByText("of 42 judged responses")).toBeInTheDocument();
    expect(screen.getByText(/Tie 22.0%/)).toBeInTheDocument();
    expect(screen.getByText(/Current model won 30.0%/)).toBeInTheDocument();
    expect(screen.getByText("gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("SIMPLE")).toBeInTheDocument();
    expect(screen.getByText("REASONING")).toBeInTheDocument();
    expect(screen.getByText("55.0%")).toBeInTheDocument();
  });

  it("shows the ends-in text while a job is still sampling", () => {
    const j = job({ ends_at: new Date(Date.now() + 3 * 86_400_000).toISOString() });
    mockHooks({ jobs: [j], detailsById: { "job-1": j } });
    render(<ShadowEvalSection />);
    expect(screen.getByText(/ends in 3 days/)).toBeInTheDocument();
  });

  it("shows recorded eval spend against the job's dollar budget", () => {
    const j = job();
    mockHooks({ jobs: [j], detailsById: { "job-1": j } });
    render(<ShadowEvalSection />);
    expect(screen.getByText(/\$3\.21 of \$10\.00 eval spend/)).toBeInTheDocument();
  });

  it("shows spend without a budget cap for a job from before spend budgets existed", () => {
    const j = job({ keys: [keyEntry("hashed-key-abc", { max_budget: null, spend: 3.21 })] });
    mockHooks({ jobs: [j], detailsById: { "job-1": j } });
    render(<ShadowEvalSection />);
    expect(screen.getByText(/\$3\.21 eval spend/)).toBeInTheDocument();
    expect(screen.queryByText(/of \$/)).not.toBeInTheDocument();
  });

  it("flags rows with fewer than 30 judged turns as low sample", () => {
    const j = job();
    mockHooks({ jobs: [j], detailsById: { "job-1": j } });
    render(<ShadowEvalSection />);
    expect(screen.getAllByText("(low sample)")).toHaveLength(1);
  });

  it("surfaces the last failure so a growing error_count is diagnosable", () => {
    const j = job({ error_count: 7, last_error: "judge call failed: LLM Provider NOT provided" });
    mockHooks({ jobs: [j], detailsById: { "job-1": j } });
    render(<ShadowEvalSection />);
    expect(screen.getByText(/LLM Provider NOT provided/)).toBeInTheDocument();
  });

  it("stops the running job from the stop button", async () => {
    const user = userEvent.setup();
    const j = job();
    const { stop } = mockHooks({ jobs: [j], detailsById: { "job-1": j } });
    render(<ShadowEvalSection />);

    await user.click(screen.getByText("Stop"));

    expect(stop.mutate).toHaveBeenCalledWith("job-1");
  });

  it("hides the stop button and offers the start form once the latest job completed", () => {
    const done = job({ status: "completed" });
    mockHooks({ jobs: [done], detailsById: { "job-1": done } });
    render(<ShadowEvalSection />);
    expect(screen.queryByText("Stop")).not.toBeInTheDocument();
    expect(screen.getByText("Start a shadow eval")).toBeInTheDocument();
  });

  it("renders nothing for non-admins when the proxy answers 403", () => {
    mockHooks({ error: new ApiError("forbidden", 403, {}) });
    const { container } = render(<ShadowEvalSection />);
    expect(container).toBeEmptyDOMElement();
  });

  it("keeps the start button disabled until key, router, and judge model are picked, then submits every picked key", async () => {
    const user = userEvent.setup();
    const { start } = mockHooks({});
    render(<ShadowEvalSection />);

    expect(screen.getByText("Start shadow eval")).toBeDisabled();

    const keyInput = screen.getByPlaceholderText("Search keys by alias");
    await user.click(keyInput);
    const keyList = await screen.findByTestId("paginated-multi-select-list");
    await user.click(within(keyList).getByText("prod-alpha"));
    await user.click(keyInput);
    await user.click(within(keyList).getByText("staging-beta"));
    await user.click(screen.getByPlaceholderText("Select an auto-router"));
    await user.click(await screen.findByText("gpt-auto"));

    expect(screen.getByText("Start shadow eval")).toBeDisabled();

    await user.click(screen.getByPlaceholderText("Select a judge model"));
    await user.click(await screen.findByRole("option", { name: /anthropic\/claude-sonnet-5/ }));
    await user.click(screen.getByText("Start shadow eval"));

    const expectedBody = {
      api_key_ids: ["hash-alpha", "hash-beta"],
      router_name: "gpt-auto",
      direction: "forward",
      shadow_percentage: 10,
      duration_days: 7,
      max_budget: 10,
      judge_model: "anthropic/claude-sonnet-5",
    };
    expect(start.mutate).toHaveBeenCalledWith(expectedBody);
  });

  it("requires a baseline model in reverse mode and submits it, while forward mode never shows the picker", async () => {
    const user = userEvent.setup();
    const { start } = mockHooks({});
    render(<ShadowEvalSection />);

    expect(screen.queryByPlaceholderText("Select a baseline model")).not.toBeInTheDocument();

    await user.click(screen.getByText("Adoption check: key's traffic vs the router"));
    await user.click(await screen.findByText("Regression check: router's picks vs a baseline"));
    await user.click(screen.getByPlaceholderText("Search keys by alias"));
    const keyList = await screen.findByTestId("paginated-multi-select-list");
    await user.click(within(keyList).getByText("prod-alpha"));
    await user.click(screen.getByPlaceholderText("Select an auto-router"));
    await user.click(await screen.findByText("gpt-auto"));
    await user.click(screen.getByPlaceholderText("Select a judge model"));
    await user.click(await screen.findByRole("option", { name: /anthropic\/claude-sonnet-5/ }));

    expect(screen.getByText("Start shadow eval")).toBeDisabled();

    await user.click(screen.getByPlaceholderText("Select a baseline model"));
    expect(await screen.findByRole("option", { name: /openai\/gpt-4o/ })).toBeInTheDocument();
    await user.click(screen.getByRole("option", { name: /prod-claude/ }));
    await user.click(screen.getByText("Start shadow eval"));

    const expectedBody = {
      api_key_ids: ["hash-alpha"],
      router_name: "gpt-auto",
      direction: "reverse",
      baseline_model: "prod-claude",
      shadow_percentage: 10,
      duration_days: 7,
      max_budget: 10,
      judge_model: "anthropic/claude-sonnet-5",
    };
    expect(start.mutate).toHaveBeenCalledWith(expectedBody);
  });

  it("flips the arm labels and headline for a reverse job's results", () => {
    const j = job({ direction: "reverse", baseline_model: "openai/gpt-4o" });
    mockHooks({ jobs: [j], detailsById: { "job-1": j } });
    render(<ShadowEvalSection />);

    expect(
      screen.getByText(
        (_, element) => element?.textContent === "Comparing claude-auto to openai/gpt-4o on 10% of prod-alpha traffic",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("Router matched or beat the baseline")).toBeInTheDocument();
    expect(screen.getByText("52.0%")).toBeInTheDocument();
    expect(screen.getByText(/Router won 30.0%/)).toBeInTheDocument();
    expect(screen.getByText(/Baseline won 48.0%/)).toBeInTheDocument();
    expect(screen.getAllByText("Baseline wins")).toHaveLength(2);
    expect(screen.getByText("Router pick")).toBeInTheDocument();
    expect(screen.queryByText(/Current model/)).not.toBeInTheDocument();
    expect(screen.queryByText("Compared against")).not.toBeInTheDocument();
  });

  it("labels the shadowed key by alias, then masked name, then truncated hash", () => {
    expect(shadowedKeyLabel(job().keys[0])).toBe("prod-alpha");
    expect(shadowedKeyLabel(keyEntry("hashed-key-abc", { key_name: "sk-...alpha" }))).toBe("sk-...alpha");
    expect(shadowedKeyLabel(keyEntry("hashed-key-abc"))).toBe("hashed-key…");
  });

  it("breaks results down per key, so one key exhausting its own budget is visible while a sibling runs on", () => {
    mockHooks({
      jobs: [
        job({
          judged_count: 205,
          keys: [
            keyEntry("hash-spent", { max_budget: 2, spend: 1.5, stopped_at: "2026-08-08T00:00:00Z" }),
            keyEntry("hash-hungry", { max_budget: 5, spend: 0.2 }),
          ],
          results: {
            by_tier: [],
            by_current_model: [],
            by_key: [
              {
                group: "hash-spent",
                turn_count: 200,
                real_win_rate_pct: 20.0,
                shadow_win_rate_pct: 60.0,
                tie_rate_pct: 20.0,
                avg_judge_confidence: 0.9,
              },
            ],
            overall_shadow_win_rate_pct: 60.0,
            overall_tie_rate_pct: 20.0,
          },
        }),
      ],
    });
    render(<ShadowEvalSection />);

    const spent = screen.getByText("hash-spent…").closest("tr");
    const hungry = screen.getByText("hash-hungr…").closest("tr");
    if (!spent || !hungry) throw new Error("expected a table row per scoped key");

    expect(within(spent).getByText("stopped")).toBeInTheDocument();
    expect(within(spent).getByText("$1.50 / $2.00")).toBeInTheDocument();
    expect(within(spent).getByText("60.0%")).toBeInTheDocument();

    expect(within(hungry).getByText("running")).toBeInTheDocument();
    expect(within(hungry).getByText("$0.2000 / $5.00")).toBeInTheDocument();
    expect(within(hungry).getByText("No verdicts yet")).toBeInTheDocument();

    expect(screen.getByText(/205 turns judged/)).toBeInTheDocument();
    expect(screen.getByText(/Shadowing 10% of/)).toBeInTheDocument();
    expect(screen.getByText("2 keys")).toBeInTheDocument();
  });

  it("reads a key that spent its budget as completed even before the sweep stamps it", () => {
    const legacyTurnBudgetLeg = { max_budget: null, spend: 0.5, max_turns: 500, attempt_count: 3 };
    mockHooks({
      jobs: [
        job({
          keys: [
            keyEntry("hash-spent", { max_budget: 2, spend: 2, attempt_count: 40 }),
            keyEntry("hash-hungry", legacyTurnBudgetLeg),
          ],
        }),
      ],
    });
    render(<ShadowEvalSection />);

    const spent = screen.getByText("hash-spent…").closest("tr");
    const hungry = screen.getByText("hash-hungr…").closest("tr");
    if (!spent || !hungry) throw new Error("expected a table row per scoped key");
    expect(within(spent).getByText("completed")).toBeInTheDocument();
    expect(within(spent).getByText("$2.00 / $2.00")).toBeInTheDocument();
    expect(within(hungry).getByText("running")).toBeInTheDocument();
    expect(within(hungry).getByText("3 / 500 turns")).toBeInTheDocument();
  });

  it("shows the per-key table while a multi-key job is still collecting, before any verdicts exist", () => {
    mockHooks({
      jobs: [
        job({
          judged_count: 0,
          results: null,
          keys: [
            keyEntry("hash-spent", { max_budget: 0.5, spend: 0.5, attempt_count: 2 }),
            keyEntry("hash-hungry", { max_budget: 5, spend: 0.01, attempt_count: 1 }),
          ],
        }),
      ],
    });
    render(<ShadowEvalSection />);

    const spent = screen.getByText("hash-spent…").closest("tr");
    if (!spent) throw new Error("expected a per-key row before verdicts exist");
    expect(within(spent).getByText("completed")).toBeInTheDocument();
    expect(within(spent).getByText("$0.5000 / $0.5000")).toBeInTheDocument();
    expect(screen.getByText("Budget used")).toBeInTheDocument();
    expect(screen.queryByText("Judged turns")).not.toBeInTheDocument();
    expect(screen.getByText(/Collecting verdicts/)).toBeInTheDocument();
  });

  it("reads every key as completed once the job's window closes, whatever its own stop state", () => {
    mockHooks({
      jobs: [
        job({
          status: "completed",
          keys: [
            keyEntry("hash-spent", { max_turns: 200, stopped_at: "2026-08-08T00:00:00Z" }),
            keyEntry("hash-hungry", { max_turns: 500 }),
          ],
        }),
      ],
    });
    render(<ShadowEvalSection />);

    const hungry = screen.getByText("hash-hungr…").closest("tr");
    if (!hungry) throw new Error("expected a table row per scoped key");
    expect(within(hungry).getByText("completed")).toBeInTheDocument();
    expect(within(hungry).queryByText("running")).not.toBeInTheDocument();
  });

  it("keeps an older job's verdicts reachable through the previous evaluations list", async () => {
    const user = userEvent.setup();
    const emptyOverrides: Partial<ShadowEvalJob> = {
      job_id: "job-new",
      status: "running",
      judged_count: 0,
      error_count: 0,
      results: null,
    };
    const current = job(emptyOverrides);
    const older = job({ job_id: "job-old", status: "completed", results: null });
    mockHooks({ jobs: [current, older], detailsById: { "job-new": current, "job-old": job({ job_id: "job-old" }) } });
    render(<ShadowEvalSection />);

    expect(screen.queryByText("SIMPLE")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Previous evaluations \(1\)/ }));
    expect(screen.getByText("view results")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /10% of prod-alpha traffic via claude-auto/ }));

    expect(await screen.findByText("SIMPLE")).toBeInTheDocument();
    expect(screen.getByText("REASONING")).toBeInTheDocument();
  });
});
