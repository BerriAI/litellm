import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { DeploymentPricingBreakdown } from "./DeploymentPricingBreakdown";
import type { DeploymentPricing } from "./deploymentPricing";

const OPENAI: DeploymentPricing = {
  provider: "openai",
  costs: {
    input_cost_per_token: 2.5e-6,
    output_cost_per_token: 1e-5,
    input_cost_per_token_priority: 5e-6,
    cache_read_input_token_cost_above_272k_tokens_priority: 1e-6,
    output_cost_per_second_720p: 0.05,
  },
  tiered_pricing: [{ range: [0, 256_000], costs: { input_cost_per_token: 5e-8, output_cost_per_token: 4e-7 } }],
};

const AZURE: DeploymentPricing = { provider: "azure", costs: { input_cost_per_token: 2.75e-6 }, tiered_pricing: [] };

const pricesOf = (region: HTMLElement, group: string, charge: string) => {
  const rows = within(within(region).getByRole("rowgroup", { name: group })).getAllByRole("row");
  const row = rows.find((candidate) => within(candidate).queryByRole("rowheader", { name: charge }));
  expect(row).toBeDefined();
  return within(row as HTMLElement)
    .getAllByRole("cell")
    .map((cell) => cell.textContent);
};

describe("DeploymentPricingBreakdown", () => {
  it("renders one block per deployment, headed by its provider, with that deployment's own prices", () => {
    render(<DeploymentPricingBreakdown deployments={[OPENAI, AZURE]} heading={<h3>Pricing</h3>} />);

    const openai = screen.getByRole("region", { name: "OpenAI pricing" });
    const azure = screen.getByRole("region", { name: "Azure pricing" });
    expect(within(openai).getByRole("heading", { name: /OpenAI/ })).toBeInTheDocument();
    expect(pricesOf(openai, "Base prices", "Input / 1M tokens")).toEqual(["$2.50", "$5.00"]);
    expect(pricesOf(azure, "Base prices", "Input / 1M tokens")).toEqual(["$2.75"]);
    expect(within(azure).queryByText(/Blank cells/)).not.toBeInTheDocument();
    expect(screen.getByText(/This model has 2 deployments/)).toBeInTheDocument();
    expect(screen.getAllByText(/Adjustments made per request/)).toHaveLength(1);
  });

  it("lays prices out by service tier column and context threshold group", () => {
    render(<DeploymentPricingBreakdown deployments={[OPENAI]} />);

    const openai = screen.getByRole("region", { name: "OpenAI pricing" });
    const header = within(openai).getAllByRole("rowgroup")[0];
    expect(
      within(header)
        .getAllByRole("columnheader")
        .map((cell) => cell.textContent),
    ).toEqual(["Charge", "Standard", "Priority"]);
    expect(pricesOf(openai, "Base prices", "Output / 1M tokens")).toEqual(["$10.00", ""]);
    expect(pricesOf(openai, "Above 272K tokens", "Cache read / 1M tokens")).toEqual(["", "$1.00"]);
    expect(within(openai).getByText("Blank cells aren't priced separately for this deployment.")).toBeInTheDocument();
    expect(screen.queryByText(/This model has/)).not.toBeInTheDocument();
  });

  it("shows tiered pricing ranges and keys it could not read", () => {
    render(<DeploymentPricingBreakdown deployments={[OPENAI]} />);

    expect(screen.getByText("0–256K tokens: $0.05 input / $0.40 output")).toBeInTheDocument();
    expect(screen.getByText("Output / second 720p")).toBeInTheDocument();
    expect(screen.queryByText("output_cost_per_second_720p")).not.toBeInTheDocument();
    expect(screen.getByText("$0.05")).toBeInTheDocument();
  });

  it("shows only the request-size ranges for a deployment priced purely by tiers", () => {
    render(
      <DeploymentPricingBreakdown
        deployments={[
          {
            provider: "dashscope",
            costs: {},
            tiered_pricing: [{ range: [0, 256_000], costs: { input_cost_per_token: 5e-8 } }],
          },
        ]}
      />,
    );

    expect(screen.getByText("0–256K tokens: $0.05 input")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it.each([
    ["missing", undefined],
    ["null", null],
    ["empty", []],
  ])("renders nothing, heading included, when deployment pricing is %s", (_, deployments) => {
    const { container } = render(
      <DeploymentPricingBreakdown deployments={deployments} heading={<h3>Pricing by deployment</h3>} />,
    );

    expect(container).toBeEmptyDOMElement();
  });
});
