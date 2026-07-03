import React from "react";
import { describe, it, expect } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import { CostBreakdownViewer, CostBreakdown } from "./CostBreakdownViewer";

async function expandCostBreakdown() {
  const user = userEvent.setup();
  await user.click(screen.getByText("Cost Breakdown"));
}

describe("CostBreakdownViewer", () => {
  it("renders nothing when costBreakdown is null", () => {
    const { container } = renderWithProviders(<CostBreakdownViewer costBreakdown={null} totalSpend={0} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when costBreakdown is undefined", () => {
    const { container } = renderWithProviders(<CostBreakdownViewer costBreakdown={undefined} totalSpend={0} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders cost breakdown with input and output costs", async () => {
    renderWithProviders(
      <CostBreakdownViewer
        costBreakdown={{
          input_cost: 0.001,
          output_cost: 0.002,
          total_cost: 0.003,
          original_cost: 0.003,
        }}
        totalSpend={0.003}
        promptTokens={100}
        completionTokens={200}
      />,
    );

    expect(screen.getByText("Cost Breakdown")).toBeInTheDocument();
    await expandCostBreakdown();
    expect(screen.getByText("Input Cost:")).toBeInTheDocument();
    expect(screen.getByText("Output Cost:")).toBeInTheDocument();
    expect(screen.getByText("Final Calculated Cost:")).toBeInTheDocument();
  });

  it("shows token counts when the panel is expanded", async () => {
    renderWithProviders(
      <CostBreakdownViewer
        costBreakdown={{
          input_cost: 0.001,
          output_cost: 0.002,
        }}
        totalSpend={0.003}
        promptTokens={500}
        completionTokens={200}
      />,
    );

    await expandCostBreakdown();

    expect(screen.getByText("Input Cost:")).toBeInTheDocument();
    expect(screen.getByText("Output Cost:")).toBeInTheDocument();
    expect(screen.getByText(/500 prompt tokens/)).toBeInTheDocument();
    expect(screen.getByText(/200 completion tokens/)).toBeInTheDocument();
  });

  it("shows non-null, non-zero additional_costs", async () => {
    renderWithProviders(
      <CostBreakdownViewer
        costBreakdown={{
          input_cost: 0.001,
          output_cost: 0.002,
          total_cost: 0.00312,
          original_cost: 0.00312,
          additional_costs: {
            "Azure Model Router Flat Cost": 0.00012,
            "Routing Fee": 0.0005,
          },
        }}
        totalSpend={0.00312}
        promptTokens={100}
        completionTokens={200}
      />,
    );

    await expandCostBreakdown();
    expect(screen.getByText("Azure Model Router Flat Cost:")).toBeInTheDocument();
    expect(screen.getByText("Routing Fee:")).toBeInTheDocument();
  });

  it("filters out null and zero additional_costs", async () => {
    renderWithProviders(
      <CostBreakdownViewer
        costBreakdown={{
          input_cost: 0.001,
          output_cost: 0.002,
          total_cost: 0.00312,
          original_cost: 0.00312,
          additional_costs: {
            "Azure Model Router Flat Cost": 0.00012,
            "Zero Cost": 0,
            "Null Cost": null as unknown as number,
          },
        }}
        totalSpend={0.00312}
        promptTokens={100}
        completionTokens={200}
      />,
    );

    await expandCostBreakdown();
    expect(screen.getByText("Azure Model Router Flat Cost:")).toBeInTheDocument();
    expect(screen.queryByText("Zero Cost:")).not.toBeInTheDocument();
    expect(screen.queryByText("Null Cost:")).not.toBeInTheDocument();
  });

  it("renders when only additional_costs exist (no input/output costs)", async () => {
    const { container } = renderWithProviders(
      <CostBreakdownViewer
        costBreakdown={{
          total_cost: 0.00012,
          original_cost: 0.00012,
          additional_costs: {
            "Model Router Flat Cost": 0.00012,
          },
        }}
        totalSpend={0.00012}
      />,
    );

    expect(screen.getByText("Cost Breakdown")).toBeInTheDocument();
    await expandCostBreakdown();
    expect(screen.getByText("Model Router Flat Cost:")).toBeInTheDocument();
    expect(container).not.toBeEmptyDOMElement();
  });

  it("returns null when additional_costs are all null/zero", () => {
    const { container } = renderWithProviders(
      <CostBreakdownViewer
        costBreakdown={{
          additional_costs: {
            Zero: 0,
            Null: null as unknown as number,
          },
        }}
        totalSpend={0}
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("expands to show additional_costs on click", async () => {
    renderWithProviders(
      <CostBreakdownViewer
        costBreakdown={{
          input_cost: 0.001,
          output_cost: 0.002,
          total_cost: 0.00312,
          original_cost: 0.00312,
          additional_costs: {
            "Azure Model Router Flat Cost": 0.00012,
          },
        }}
        totalSpend={0.00312}
        promptTokens={100}
        completionTokens={200}
      />,
    );

    expect(screen.queryByText("Azure Model Router Flat Cost:")).not.toBeInTheDocument();
    await expandCostBreakdown();
    expect(screen.getByText("Azure Model Router Flat Cost:")).toBeInTheDocument();
  });

  it("shows '(Cached)' in the header when cacheHit is true", () => {
    const breakdown: CostBreakdown = {
      input_cost: 0.001,
      output_cost: 0.002,
      total_cost: 0.003,
    };

    renderWithProviders(<CostBreakdownViewer costBreakdown={breakdown} totalSpend={0} cacheHit="true" />);

    expect(screen.getByText(/\(Cached\)/)).toBeInTheDocument();
  });

  it("shows discount label with percentage when panel is expanded", async () => {
    const breakdown: CostBreakdown = {
      input_cost: 0.01,
      output_cost: 0.02,
      discount_percent: 0.1,
      discount_amount: 0.003,
    };

    renderWithProviders(<CostBreakdownViewer costBreakdown={breakdown} totalSpend={0.027} />);

    await expandCostBreakdown();

    expect(screen.getByText(/Discount \(10\.00%\)/)).toBeInTheDocument();
  });

  it("shows margin label with percentage when panel is expanded", async () => {
    const breakdown: CostBreakdown = {
      input_cost: 0.01,
      output_cost: 0.02,
      margin_percent: 0.15,
      margin_total_amount: 0.005,
    };

    renderWithProviders(<CostBreakdownViewer costBreakdown={breakdown} totalSpend={0.035} />);

    await expandCostBreakdown();

    expect(screen.getByText(/Margin \(15\.00%\)/)).toBeInTheDocument();
    expect(screen.getByText("Final Calculated Cost:")).toBeInTheDocument();
  });

  describe("provider prompt-cache formula breakdown (issue #32045)", () => {
    // Numbers taken directly from the issue.
    const issueBreakdown: CostBreakdown = {
      input_cost: 0.06476666,
      cache_read_cost: 0.0645696,
      output_cost: 0.00009729,
      original_cost: 0.06486395,
    };

    it("splits the input cost into cache-miss and cache-hit token formulas", async () => {
      renderWithProviders(
        <CostBreakdownViewer
          costBreakdown={issueBreakdown}
          totalSpend={0.06486395}
          promptTokens={430798}
          completionTokens={47}
          providerCacheReadTokens={430464}
        />,
      );

      await expandCostBreakdown();

      // Assert the full formula incl. the derived per-token rate, trimmed of trailing
      // zeros and float noise, so 5.9e-7 shows as $0.00000059 (not $0.00000059000).
      expect(screen.getByText(/334 cache miss tokens \* \$0\.00000059/)).toBeInTheDocument();
      expect(screen.getByText(/430,464 cache hit tokens \* \$0\.00000015/)).toBeInTheDocument();
    });

    it("shows the output cost and original LLM cost formulas", async () => {
      renderWithProviders(
        <CostBreakdownViewer
          costBreakdown={issueBreakdown}
          totalSpend={0.06486395}
          promptTokens={430798}
          completionTokens={47}
          providerCacheReadTokens={430464}
        />,
      );

      await expandCostBreakdown();

      expect(screen.getByText(/47 completion tokens \* \$0\.00000207/)).toBeInTheDocument();
      expect(screen.getByText(/\$0\.06476666 input cost \+ \$0\.00009729 output cost/)).toBeInTheDocument();
    });

    it("does NOT show the formula when there is no provider prompt cache hit", async () => {
      renderWithProviders(
        <CostBreakdownViewer
          costBreakdown={{ input_cost: 0.001, output_cost: 0.002, original_cost: 0.003 }}
          totalSpend={0.003}
          promptTokens={100}
          completionTokens={200}
          providerCacheReadTokens={0}
        />,
      );

      await expandCostBreakdown();

      expect(screen.queryByText(/cache miss tokens \*/)).not.toBeInTheDocument();
      expect(screen.queryByText(/cache hit tokens \*/)).not.toBeInTheDocument();
    });

    it("does NOT show the formula for a LiteLLM response-cache hit (costs are $0)", async () => {
      renderWithProviders(
        <CostBreakdownViewer
          costBreakdown={issueBreakdown}
          totalSpend={0}
          promptTokens={430798}
          completionTokens={47}
          providerCacheReadTokens={430464}
          cacheHit="true"
        />,
      );

      await expandCostBreakdown();

      // Response-cache hit forces $0 and must not print a per-token cache formula.
      expect(screen.queryByText(/cache miss tokens \*/)).not.toBeInTheDocument();
    });
  });
});
