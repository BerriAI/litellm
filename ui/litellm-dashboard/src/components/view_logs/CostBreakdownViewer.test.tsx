import React from "react";
import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import { CostBreakdownViewer, CostBreakdown } from "./CostBreakdownViewer";

describe("CostBreakdownViewer", () => {
  it("should render nothing when there is no meaningful data", () => {
    const { container } = renderWithProviders(
      <CostBreakdownViewer costBreakdown={null} totalSpend={0} />
    );

    expect(container.firstChild).toBeNull();
  });

  it("should render nothing when costBreakdown is undefined", () => {
    const { container } = renderWithProviders(
      <CostBreakdownViewer costBreakdown={undefined} totalSpend={0} />
    );

    expect(container.firstChild).toBeNull();
  });

  it("should render the collapse header with heading and total", () => {
    const breakdown: CostBreakdown = {
      input_cost: 0.001,
      output_cost: 0.002,
      total_cost: 0.003,
    };

    renderWithProviders(
      <CostBreakdownViewer costBreakdown={breakdown} totalSpend={0.003} />
    );

    expect(screen.getByRole("heading", { name: "Cost Breakdown" })).toBeInTheDocument();
  });

  it("should show input and output costs when the panel is expanded", async () => {
    const user = userEvent.setup();
    const breakdown: CostBreakdown = {
      input_cost: 0.001,
      output_cost: 0.002,
    };

    renderWithProviders(
      <CostBreakdownViewer
        costBreakdown={breakdown}
        totalSpend={0.003}
        promptTokens={500}
        completionTokens={200}
      />
    );

    await user.click(screen.getByRole("heading", { name: "Cost Breakdown" }));

    expect(screen.getByText("Input Cost:")).toBeVisible();
    expect(screen.getByText("Output Cost:")).toBeVisible();
    expect(screen.getByText(/500 prompt tokens/)).toBeVisible();
    expect(screen.getByText(/200 completion tokens/)).toBeVisible();
  });

  it("should show '(Cached)' in the header when cacheHit is true", () => {
    const breakdown: CostBreakdown = {
      input_cost: 0.001,
      output_cost: 0.002,
      total_cost: 0.003,
    };

    renderWithProviders(
      <CostBreakdownViewer
        costBreakdown={breakdown}
        totalSpend={0}
        cacheHit="true"
      />
    );

    expect(screen.getByText(/\(Cached\)/)).toBeInTheDocument();
  });

  it("should show discount label with percentage when panel is expanded", async () => {
    const user = userEvent.setup();
    const breakdown: CostBreakdown = {
      input_cost: 0.01,
      output_cost: 0.02,
      discount_percent: 0.1,
      discount_amount: 0.003,
    };

    renderWithProviders(
      <CostBreakdownViewer costBreakdown={breakdown} totalSpend={0.027} />
    );

    await user.click(screen.getByRole("heading", { name: "Cost Breakdown" }));

    expect(screen.getByText(/Discount \(10\.00%\)/)).toBeVisible();
  });

  it("should show margin label with percentage when panel is expanded", async () => {
    const user = userEvent.setup();
    const breakdown: CostBreakdown = {
      input_cost: 0.01,
      output_cost: 0.02,
      margin_percent: 0.15,
      margin_total_amount: 0.005,
    };

    renderWithProviders(
      <CostBreakdownViewer costBreakdown={breakdown} totalSpend={0.035} />
    );

    await user.click(screen.getByRole("heading", { name: "Cost Breakdown" }));

    expect(screen.getByText(/Margin \(15\.00%\)/)).toBeVisible();
    expect(screen.getByText("Final Calculated Cost:")).toBeVisible();
  });
});
