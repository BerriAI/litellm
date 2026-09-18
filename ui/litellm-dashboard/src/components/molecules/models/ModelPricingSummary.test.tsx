import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ModelPricingSummary } from "./ModelPricingSummary";

const tokenPriced = { input_cost: "1.50", output_cost: "2.00" };

describe("ModelPricingSummary", () => {
  it("shows per-million-token rates for a token priced model", () => {
    render(<ModelPricingSummary model={tokenPriced} />);
    expect(screen.getByText("Input: $1.50/1M tokens")).toBeInTheDocument();
    expect(screen.getByText("Output: $2.00/1M tokens")).toBeInTheDocument();
  });

  it("hides $0.00 token rates and shows per-second tiers for a per-second priced model", () => {
    render(
      <ModelPricingSummary
        model={{
          input_cost: "0.00",
          output_cost: "0.00",
          output_cost_per_second: 0.1,
          output_cost_per_second_tiers: [
            { resolution: "1080p", cost: 0.12 },
            { resolution: "4k", cost: 0.3 },
          ],
        }}
      />,
    );
    expect(screen.getByText("Output: $0.10/s")).toBeInTheDocument();
    expect(screen.getByText("Output (1080p): $0.12/s")).toBeInTheDocument();
    expect(screen.getByText("Output (4k): $0.30/s")).toBeInTheDocument();
    expect(screen.queryByText(/1M tokens/)).not.toBeInTheDocument();
  });

  it("keeps a positive token rate next to the per-second rate", () => {
    render(
      <ModelPricingSummary
        model={{
          input_cost: "0.60",
          output_cost: "0.00",
          output_cost_per_second: 0.015,
        }}
      />,
    );
    expect(screen.getByText("Input: $0.60/1M tokens")).toBeInTheDocument();
    expect(screen.getByText("Output: $0.015/s")).toBeInTheDocument();
    expect(screen.queryByText("Output: $0.00/1M tokens")).not.toBeInTheDocument();
  });

  it("renders a dash when the model has no pricing at all", () => {
    render(
      <ModelPricingSummary model={{ input_cost: null, output_cost: null }} />,
    );
    expect(screen.getByText("-")).toBeInTheDocument();
    expect(screen.queryByText(/\$/)).not.toBeInTheDocument();
  });
});
