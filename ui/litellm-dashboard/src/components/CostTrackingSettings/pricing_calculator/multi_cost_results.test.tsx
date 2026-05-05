import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../../tests/test-utils";
import MultiCostResults from "./multi_cost_results";
import type { MultiModelResult } from "./types";
import type { CostEstimateResponse } from "../types";

vi.mock("./multi_export_utils", () => ({
  exportMultiToPDF: vi.fn(),
  exportMultiToCSV: vi.fn(),
}));

vi.mock("@/utils/dataUtils", () => ({
  formatNumberWithCommas: vi.fn((v: number, d: number = 0) =>
    Number.isFinite(v) ? v.toFixed(d) : "-"
  ),
}));

function makeCostResponse(overrides: Partial<CostEstimateResponse> = {}): CostEstimateResponse {
  return {
    model: "gpt-4",
    input_tokens: 1000,
    output_tokens: 500,
    num_requests_per_day: 100,
    num_requests_per_month: null,
    cost_per_request: 0.05,
    input_cost_per_request: 0.03,
    output_cost_per_request: 0.02,
    margin_cost_per_request: 0,
    daily_cost: 5.0,
    daily_input_cost: 3.0,
    daily_output_cost: 2.0,
    daily_margin_cost: 0,
    monthly_cost: null,
    monthly_input_cost: null,
    monthly_output_cost: null,
    monthly_margin_cost: null,
    input_cost_per_token: null,
    output_cost_per_token: null,
    provider: "openai",
    ...overrides,
  };
}

function makeMultiResult(overrides: Partial<MultiModelResult> = {}): MultiModelResult {
  return {
    entries: [
      {
        entry: { id: "e1", model: "gpt-4", input_tokens: 1000, output_tokens: 500 },
        result: makeCostResponse(),
        loading: false,
        error: null,
      },
    ],
    totals: {
      cost_per_request: 0.05,
      daily_cost: 5.0,
      monthly_cost: null,
      margin_per_request: 0,
      daily_margin: null,
      monthly_margin: null,
    },
    ...overrides,
  };
}

function emptyMultiResult(): MultiModelResult {
  return {
    entries: [
      {
        entry: { id: "e1", model: "", input_tokens: 1000, output_tokens: 500 },
        result: null,
        loading: false,
        error: null,
      },
    ],
    totals: {
      cost_per_request: 0,
      daily_cost: null,
      monthly_cost: null,
      margin_per_request: 0,
      daily_margin: null,
      monthly_margin: null,
    },
  };
}

describe("MultiCostResults", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("when no model has been selected", () => {
    it("should show a prompt to select models", () => {
      renderWithProviders(
        <MultiCostResults multiResult={emptyMultiResult()} timePeriod="month" />
      );
      expect(screen.getByText(/select models above to see cost estimates/i)).toBeInTheDocument();
    });
  });

  describe("when results are loading and no data has arrived yet", () => {
    it("should show a calculating costs spinner", () => {
      const multiResult: MultiModelResult = {
        entries: [
          {
            entry: { id: "e1", model: "gpt-4", input_tokens: 1000, output_tokens: 500 },
            result: null,
            loading: true,
            error: null,
          },
        ],
        totals: {
          cost_per_request: 0,
          daily_cost: null,
          monthly_cost: null,
          margin_per_request: 0,
          daily_margin: null,
          monthly_margin: null,
        },
      };

      renderWithProviders(<MultiCostResults multiResult={multiResult} timePeriod="month" />);
      expect(screen.getByText(/calculating costs/i)).toBeInTheDocument();
    });
  });

  describe("when there are errors but no valid results", () => {
    it("should display the error message with the model name", () => {
      const multiResult: MultiModelResult = {
        entries: [
          {
            entry: { id: "e1", model: "bad-model", input_tokens: 0, output_tokens: 0 },
            result: null,
            loading: false,
            error: "Pricing not found",
          },
        ],
        totals: {
          cost_per_request: 0,
          daily_cost: null,
          monthly_cost: null,
          margin_per_request: 0,
          daily_margin: null,
          monthly_margin: null,
        },
      };

      renderWithProviders(<MultiCostResults multiResult={multiResult} timePeriod="month" />);
      expect(screen.getByText(/bad-model/i)).toBeInTheDocument();
      expect(screen.getByText(/Pricing not found/i)).toBeInTheDocument();
    });
  });

  describe("when valid results are available", () => {
    it("should show the Cost Estimates heading", () => {
      renderWithProviders(
        <MultiCostResults multiResult={makeMultiResult()} timePeriod="day" />
      );
      expect(screen.getByText("Cost Estimates")).toBeInTheDocument();
    });

    it("should display the Total Per Request statistic", () => {
      renderWithProviders(
        <MultiCostResults multiResult={makeMultiResult()} timePeriod="day" />
      );
      expect(screen.getByText("Total Per Request")).toBeInTheDocument();
    });

    it("should display Total Daily statistic when timePeriod is day", () => {
      renderWithProviders(
        <MultiCostResults multiResult={makeMultiResult()} timePeriod="day" />
      );
      expect(screen.getByText("Total Daily")).toBeInTheDocument();
    });

    it("should display Total Monthly statistic when timePeriod is month", () => {
      renderWithProviders(
        <MultiCostResults
          multiResult={makeMultiResult({
            totals: { cost_per_request: 0.05, daily_cost: null, monthly_cost: 150.0, margin_per_request: 0, daily_margin: null, monthly_margin: null },
          })}
          timePeriod="month"
        />
      );
      expect(screen.getByText("Total Monthly")).toBeInTheDocument();
    });

    it("should show the model name in the summary table", () => {
      renderWithProviders(
        <MultiCostResults multiResult={makeMultiResult()} timePeriod="day" />
      );
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
    });

    it("should show the provider tag next to the model name", () => {
      renderWithProviders(
        <MultiCostResults multiResult={makeMultiResult()} timePeriod="day" />
      );
      expect(screen.getByText("openai")).toBeInTheDocument();
    });

    it("should show the Export button when results are available", () => {
      renderWithProviders(
        <MultiCostResults multiResult={makeMultiResult()} timePeriod="day" />
      );
      expect(screen.getByRole("button", { name: /export/i })).toBeInTheDocument();
    });

    it("should expand the model breakdown row when the expand button is clicked", async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <MultiCostResults multiResult={makeMultiResult()} timePeriod="day" />
      );

      // The expand column renders a button (RightOutlined icon) for rows without errors
      const expandButtons = screen.getAllByRole("button");
      // Find the small expand button (not the Export button)
      const expandButton = expandButtons.find(
        (btn) => !btn.textContent?.toLowerCase().includes("export")
      );
      expect(expandButton).toBeDefined();

      await user.click(expandButton!);

      // After expanding, the SingleModelBreakdown should be visible
      expect(screen.getByText("Total/Request")).toBeInTheDocument();
    });

    it("should show the collapse icon after expanding a row", async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <MultiCostResults multiResult={makeMultiResult()} timePeriod="day" />
      );

      const getExpandButton = () => {
        const allButtons = screen.getAllByRole("button");
        return allButtons.find((btn) => !btn.textContent?.toLowerCase().includes("export"));
      };

      // Before expand: button has the "down" aria-label (RightOutlined renders as down in ant icons)
      // Just verify clicking works and the breakdown content appears
      await user.click(getExpandButton()!);
      expect(screen.getByText("Total/Request")).toBeInTheDocument();

      // After a second click, the row collapses — content may be hidden or removed
      await user.click(getExpandButton()!);
      // The expanded content should no longer be visible
      expect(screen.queryByText("Total/Request")).not.toBeVisible();
    });
  });

  describe("margin section", () => {
    it("should show margin fee details when margin per request is greater than zero", () => {
      const multiResult = makeMultiResult({
        entries: [
          {
            entry: { id: "e1", model: "gpt-4", input_tokens: 1000, output_tokens: 500 },
            result: makeCostResponse({ margin_cost_per_request: 0.01, daily_margin_cost: 1.0 }),
            loading: false,
            error: null,
          },
        ],
        totals: {
          cost_per_request: 0.06,
          daily_cost: 6.0,
          monthly_cost: null,
          margin_per_request: 0.01,
          daily_margin: 1.0,
          monthly_margin: null,
        },
      });

      renderWithProviders(<MultiCostResults multiResult={multiResult} timePeriod="day" />);
      expect(screen.getByText("Margin Fee/Request")).toBeInTheDocument();
    });

    it("should not show margin fee details when margin per request is zero", () => {
      renderWithProviders(
        <MultiCostResults multiResult={makeMultiResult()} timePeriod="day" />
      );
      expect(screen.queryByText("Margin Fee/Request")).not.toBeInTheDocument();
    });
  });

  describe("when a model has zero cost", () => {
    it("should show a warning about missing pricing data", () => {
      const multiResult = makeMultiResult({
        entries: [
          {
            entry: { id: "e1", model: "custom-model", input_tokens: 1000, output_tokens: 500 },
            result: makeCostResponse({ model: "custom-model", cost_per_request: 0 }),
            loading: false,
            error: null,
          },
        ],
      });

      renderWithProviders(<MultiCostResults multiResult={multiResult} timePeriod="day" />);
      expect(screen.getByText(/no pricing data found/i)).toBeInTheDocument();
    });
  });
});
