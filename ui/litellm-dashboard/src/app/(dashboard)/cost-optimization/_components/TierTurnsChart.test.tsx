import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import type { AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";

vi.mock("@/components/shared/charts", () => ({
  DonutChart: ({ label }: { label: string }) => <div data-testid="donut">{label}</div>,
  SEQUENTIAL_COLOR_RAMP: ["indigo", "blue"],
  chartColorValue: (color: string) => color,
}));

import TierTurnsChart, { tierDisplayLabel } from "./TierTurnsChart";
import type { AutoRouterBenchmarkGroup, BenchmarkView } from "./autoRouterBenchmarks";

const totalsOnly = {
  sessions: 3,
  turns: 9,
  avg_turns_per_session: 3,
  avg_session_seconds: 60,
  avg_tokens_per_session: 100,
  spend: 1,
  saved_spend: 1,
  baseline_spend: 2,
  saved_pct: 50,
  saved_per_session: 0.33,
  cache: {
    coverage_pct: 0,
    hit_rate_pct: 0,
    same_model: { turns: 0, hits: 0, hit_rate_pct: 0 },
    first_visit: { turns: 0, hits: 0, hit_rate_pct: 0 },
    return_to_tier: { turns: 0, hits: 0, hit_rate_pct: 0 },
    unordered_turns: 0,
    return_misses_expired: 0,
    return_misses_within_ttl: 0,
    return_misses_unknown: 0,
    ttl_5m_turns: 0,
    ttl_1h_turns: 0,
  },
};

const groupView = (overrides: Partial<AutoRouterBenchmarkGroup> = {}): BenchmarkView => ({
  label: "claude-auto",
  stats: {
    ...totalsOnly,
    router_name: "claude-auto",
    router_type: "complexity",
    tier_turns: { SIMPLE: 3, COMPLEX: 1 },
    ...overrides,
  } as AutoRouterBenchmarkGroup,
});

const deployment = (config: unknown): AutoRouterDeployment => ({
  model_name: "claude-auto",
  litellm_params: { model: "auto_router/claude-auto", complexity_router_config: config },
});

describe("tierDisplayLabel", () => {
  it("prefers the admin's custom label for a canonical complexity tier", () => {
    expect(tierDisplayLabel("SIMPLE", { SIMPLE: "Cheap" })).toBe("Cheap");
  });

  it("falls back to the canonical name when that tier has no custom label", () => {
    expect(tierDisplayLabel("COMPLEX", { SIMPLE: "Cheap" })).toBe("Complex");
    expect(tierDisplayLabel("REASONING", undefined)).toBe("Reasoning");
  });

  it("shows a non-complexity tier verbatim, since no label map covers a quality router's tier", () => {
    expect(tierDisplayLabel("3", { SIMPLE: "Cheap" })).toBe("3");
  });
});

describe("TierTurnsChart", () => {
  it("labels each slice with its tier and share of the tiered turns", () => {
    render(<TierTurnsChart view={groupView()} autoRouters={[deployment({ tier_labels: { SIMPLE: "Cheap" } })]} />);

    expect(screen.getByText("Cheap 75%")).toBeInTheDocument();
    expect(screen.getByText("Complex 25%")).toBeInTheDocument();
    expect(screen.getByTestId("donut")).toHaveTextContent("4 total turns");
  });

  it("reads tier_labels out of a config stored as a JSON string", () => {
    const stored = JSON.stringify({ tier_labels: { SIMPLE: "Cheap" } });
    render(<TierTurnsChart view={groupView()} autoRouters={[deployment(stored)]} />);

    expect(screen.getByText("Cheap 75%")).toBeInTheDocument();
  });

  it("uses canonical names when the router is not in the deployment list", () => {
    render(<TierTurnsChart view={groupView()} autoRouters={[]} />);

    expect(screen.getByText("Simple 75%")).toBeInTheDocument();
    expect(screen.getByText("Complex 25%")).toBeInTheDocument();
  });

  it("lists each tier's assigned models below its name and share", () => {
    render(
      <TierTurnsChart
        view={groupView()}
        autoRouters={[deployment({ tiers: { SIMPLE: ["gpt-4o-mini"], COMPLEX: ["gpt-4o", "claude-3-opus"] } })]}
      />,
    );

    expect(screen.getByText("gpt-4o-mini")).toBeInTheDocument();
    expect(screen.getByText("gpt-4o, claude-3-opus")).toBeInTheDocument();
  });

  it("widens a bare string tier (pinned single model) into its one-model list", () => {
    render(<TierTurnsChart view={groupView()} autoRouters={[deployment({ tiers: { SIMPLE: "gpt-4o-mini" } })]} />);

    expect(screen.getByText("gpt-4o-mini")).toBeInTheDocument();
  });

  it("omits the model line for a tier with no configured models", () => {
    render(<TierTurnsChart view={groupView()} autoRouters={[deployment({ tiers: { SIMPLE: [] } })]} />);

    expect(screen.getByText("Simple 75%")).toBeInTheDocument();
  });

  it("shows no models for a quality router's numeric tier, which has no per-tier model list", () => {
    render(
      <TierTurnsChart
        view={groupView({ router_type: "quality", tier_turns: { "3": 3, "1": 1 } })}
        autoRouters={[deployment({ quality_router_config: { available_models: ["gpt-4o"] } })]}
      />,
    );

    expect(screen.getByText("3 75%")).toBeInTheDocument();
    expect(screen.getByText("1 25%")).toBeInTheDocument();
    expect(screen.queryByText("gpt-4o")).not.toBeInTheDocument();
  });

  it("ignores a same-named deployment of a different router type", () => {
    const qualityDeployment = {
      model_name: "claude-auto",
      litellm_params: { model: "auto_router/claude-auto", quality_router_config: { available_models: ["gpt-4o"] } },
    };

    render(
      <TierTurnsChart
        view={groupView()} // complexity router
        autoRouters={[qualityDeployment] as AutoRouterDeployment[]}
      />,
    );

    expect(screen.getByText("Simple 75%")).toBeInTheDocument();
    expect(screen.queryByText("gpt-4o")).not.toBeInTheDocument();
  });

  it("renders nothing for the all-routers view, which carries no router identity", () => {
    const { container } = render(
      <TierTurnsChart view={{ label: "All auto-routers", stats: totalsOnly }} autoRouters={[]} />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the router recorded no tiers", () => {
    const { container } = render(<TierTurnsChart view={groupView({ tier_turns: {} })} autoRouters={[]} />);

    expect(container).toBeEmptyDOMElement();
  });
});
