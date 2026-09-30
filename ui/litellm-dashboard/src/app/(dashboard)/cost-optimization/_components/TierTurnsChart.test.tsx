import { fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import type { AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import { ApiError } from "@/lib/http/client";

vi.mock("@/components/shared/charts", () => ({
  DonutChart: ({ label }: { label: string }) => <div data-testid="donut">{label}</div>,
  DEFAULT_COLOR_CYCLE: ["blue", "cyan", "sky", "indigo", "violet", "purple", "fuchsia", "slate"],
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
  savings_estimated_turns: 9,
  savings_estimated_actual_spend: 1,
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

  it("lists a custom tier's models, which the built-in name guard used to hide", () => {
    render(
      <TierTurnsChart
        view={groupView({ tier_turns: { CASUAL: 3, SECURITY_REVIEW: 1 } })}
        autoRouters={[deployment({ tiers: { CASUAL: ["gpt-4o-mini"], SECURITY_REVIEW: ["o1-preview"] } })]}
      />,
    );

    expect(screen.getByText(/SECURITY_REVIEW/)).toBeInTheDocument();
    expect(screen.getByText("o1-preview")).toBeInTheDocument();
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

it("switches the existing donut to actual spend and shows each destination model", () => {
  render(
    <TierTurnsChart
      view={groupView()}
      autoRouters={[]}
      usage={[
        { model: "fast", router_name: "claude-auto", router_type: "complexity", tier: "SIMPLE", requests: 3, spend: 1 },
        {
          model: "strong",
          router_name: "claude-auto",
          router_type: "complexity",
          tier: "SIMPLE",
          requests: 1,
          spend: 5,
        },
        { model: "fallback", router_name: "claude-auto", router_type: "complexity", tier: null, requests: 1, spend: 4 },
      ]}
    />,
  );
  expect(screen.getByText("Simple 80% · $6.00")).toBeInTheDocument();
  expect(screen.getByText("strong · 1 request (20%) · $5.00")).toBeInTheDocument();
  expect(screen.getByText("Default / no tier 20% · $4.00")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Spend" }));
  expect(screen.getByTestId("donut")).toHaveTextContent("$10.00");
  expect(screen.getByText("Simple 60% · $6.00")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Spend" })).toHaveAttribute("aria-pressed", "true");
});

it("keeps configured aliases and unused tiers visible separately from recorded destinations", () => {
  render(
    <TierTurnsChart
      view={groupView()}
      autoRouters={[deployment({ tiers: { SIMPLE: ["public-fast", "unused"], COMPLEX: "public-strong" } })]}
      usage={[
        {
          model: "provider/fast",
          router_name: "claude-auto",
          router_type: "complexity",
          tier: "SIMPLE",
          requests: 3,
          spend: 1,
        },
      ]}
    />,
  );
  expect(screen.getByText("Configured: public-fast, unused")).toBeVisible();
  expect(screen.getByText("provider/fast · 3 requests (100%) · $1.00")).toBeVisible();
  expect(screen.getByText("Configured: public-strong")).toBeVisible();
  expect(screen.getByText("No retained requests for this tier")).toBeVisible();
  expect(screen.getByTestId("donut")).toHaveTextContent("3 total requests");
});

it("keeps traffic visible when requests have no spend, including after changing the range in Spend mode", () => {
  const usage = [
    { model: "fast", router_name: "claude-auto", router_type: "complexity", tier: "SIMPLE", requests: 3, spend: 1 },
    { model: "strong", router_name: "claude-auto", router_type: "complexity", tier: "COMPLEX", requests: 1, spend: 2 },
  ];
  const { rerender } = render(<TierTurnsChart view={groupView()} autoRouters={[]} usage={usage} />);
  fireEvent.click(screen.getByRole("button", { name: "Spend" }));
  expect(screen.getByTestId("donut")).toHaveTextContent("$3.00");
  rerender(<TierTurnsChart view={groupView()} autoRouters={[]} usage={usage.map((row) => ({ ...row, spend: 0 }))} />);
  expect(screen.getByRole("button", { name: "Spend" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Traffic" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByText(/No model spend recorded; showing traffic/)).toBeVisible();
  expect(screen.getByText("Simple 75% · $0.00")).toBeVisible();
  expect(screen.getByText("Complex 25% · $0.00")).toBeVisible();
  expect(screen.getByTestId("donut")).toHaveTextContent("4 total requests");
});

it("explains the date limit while retaining the existing tier traffic chart", () => {
  render(
    <TierTurnsChart
      view={groupView()}
      autoRouters={[]}
      usageError={new ApiError("Select a range of 93 days or fewer", 400, {})}
    />,
  );
  expect(screen.getByText(/Select a range of 93 days or fewer/)).toBeVisible();
  expect(screen.getByTestId("donut")).toHaveTextContent("4 total turns");
  expect(screen.queryByRole("button", { name: "Spend" })).not.toBeInTheDocument();
});
