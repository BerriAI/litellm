import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { AutoRouterBenchmarksResponse, AutoRouterGroupBenchmark } from "@/components/networking";

const autoRouterBenchmarksCall = vi.fn();
vi.mock("@/components/networking", () => ({
  autoRouterBenchmarksCall: (...args: unknown[]) => autoRouterBenchmarksCall(...args),
}));

import AutoRouterBenchmarksPanel from "./AutoRouterBenchmarksPanel";

const group = (overrides: Partial<AutoRouterGroupBenchmark>): AutoRouterGroupBenchmark => ({
  model_group: "auto",
  router_kind: "complexity",
  baseline_model: "claude-opus-5",
  sessions: 75,
  turns: 1071,
  avg_turns_per_session: 14.3,
  avg_session_length_seconds: 2557.6,
  total_tokens: 142_000_000,
  avg_tokens_per_session: 1_899_868,
  actual_spend: 86.76,
  baseline_spend: 728.01,
  savings: 641.25,
  savings_pct: 88.1,
  ...overrides,
});

const resolve = (groups: AutoRouterGroupBenchmark[]): void => {
  const response: AutoRouterBenchmarksResponse = { start_date: "2026-06-29", end_date: "2026-07-29", groups };
  autoRouterBenchmarksCall.mockResolvedValue(response);
};

describe("AutoRouterBenchmarksPanel", () => {
  beforeEach(() => {
    autoRouterBenchmarksCall.mockReset();
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders the four session metrics with the routed group's numbers", async () => {
    resolve([group({})]);
    render(<AutoRouterBenchmarksPanel accessToken="sk-test" />);

    await waitFor(() => expect(screen.getByText("Turns per session")).toBeInTheDocument());
    expect(screen.getByText("14.3")).toBeInTheDocument();
    // 2557.6s -> 42.6m
    expect(screen.getByText("42.6m")).toBeInTheDocument();
    // 1,899,868 tokens -> compact 1.9M
    expect(screen.getByText("1.9M")).toBeInTheDocument();
    // savings dollars + percent against the baseline model
    expect(screen.getByText("$641.25")).toBeInTheDocument();
    expect(screen.getByText(/88% vs claude-opus-5/)).toBeInTheDocument();
  });

  it("labels the savings as an estimate and names the baseline it compares against", async () => {
    resolve([group({})]);
    render(<AutoRouterBenchmarksPanel accessToken="sk-test" />);

    await waitFor(() => expect(screen.getByText("Estimated savings")).toBeInTheDocument());
    expect(screen.getByText(/does not model the caching a single-model baseline would have had/)).toBeInTheDocument();
  });

  it("renders one card per auto-router group", async () => {
    resolve([group({ model_group: "auto" }), group({ model_group: "claude-auto", avg_turns_per_session: 4.1 })]);
    render(<AutoRouterBenchmarksPanel accessToken="sk-test" />);

    await waitFor(() => expect(screen.getByText("auto")).toBeInTheDocument());
    expect(screen.getByText("claude-auto")).toBeInTheDocument();
    expect(screen.getByText("4.1")).toBeInTheDocument();
  });

  it("renders nothing when no auto-router has routed sessions", async () => {
    resolve([]);
    const { container } = render(<AutoRouterBenchmarksPanel accessToken="sk-test" />);

    await waitFor(() => expect(autoRouterBenchmarksCall).toHaveBeenCalled());
    expect(container.textContent).not.toContain("Turns per session");
  });

  it("does not call the endpoint without an access token", () => {
    render(<AutoRouterBenchmarksPanel accessToken={null} />);
    expect(autoRouterBenchmarksCall).not.toHaveBeenCalled();
  });
});
