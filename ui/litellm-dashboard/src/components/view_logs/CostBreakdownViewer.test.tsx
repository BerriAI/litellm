import React from "react";
import { describe, it, expect } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import { CostBreakdownViewer } from "./CostBreakdownViewer";

async function expandCostBreakdown() {
  const user = userEvent.setup();
  await user.click(screen.getByText("Cost Breakdown"));
}

describe("CostBreakdownViewer", () => {
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
      />
    );

    expect(screen.getByText("Cost Breakdown")).toBeInTheDocument();
    await expandCostBreakdown();
    expect(screen.getByText("Input Cost:")).toBeInTheDocument();
    expect(screen.getByText("Output Cost:")).toBeInTheDocument();
    expect(screen.getByText("Final Calculated Cost:")).toBeInTheDocument();
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
      />
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
      />
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
      />
    );

    expect(screen.getByText("Cost Breakdown")).toBeInTheDocument();
    await expandCostBreakdown();
    expect(screen.getByText("Model Router Flat Cost:")).toBeInTheDocument();
    expect(container).not.toBeEmptyDOMElement();
  });

  it("returns null when no meaningful data", () => {
    const { container } = renderWithProviders(
      <CostBreakdownViewer
        costBreakdown={undefined}
        totalSpend={0}
      />
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("returns null when additional_costs are all null/zero", () => {
    const { container } = renderWithProviders(
      <CostBreakdownViewer
        costBreakdown={{
          additional_costs: {
            "Zero": 0,
            "Null": null as unknown as number,
          },
        }}
        totalSpend={0}
      />
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
      />
    );

    expect(screen.queryByText("Azure Model Router Flat Cost:")).not.toBeInTheDocument();
    await expandCostBreakdown();
    expect(screen.getByText("Azure Model Router Flat Cost:")).toBeInTheDocument();
  });
});
