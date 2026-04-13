import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../../tests/test-utils";
import MultiExportDropdown from "./multi_export_dropdown";
import type { MultiModelResult } from "./types";

vi.mock("./multi_export_utils", () => ({
  exportMultiToPDF: vi.fn(),
  exportMultiToCSV: vi.fn(),
}));

import { exportMultiToPDF, exportMultiToCSV } from "./multi_export_utils";

function makeMultiResult(hasResult: boolean): MultiModelResult {
  return {
    entries: [
      {
        entry: { id: "e1", model: "gpt-4", input_tokens: 1000, output_tokens: 500 },
        result: hasResult
          ? {
              model: "gpt-4",
              input_tokens: 1000,
              output_tokens: 500,
              num_requests_per_day: null,
              num_requests_per_month: null,
              cost_per_request: 0.05,
              input_cost_per_request: 0.03,
              output_cost_per_request: 0.02,
              margin_cost_per_request: 0,
              daily_cost: null,
              daily_input_cost: null,
              daily_output_cost: null,
              daily_margin_cost: null,
              monthly_cost: null,
              monthly_input_cost: null,
              monthly_output_cost: null,
              monthly_margin_cost: null,
              input_cost_per_token: null,
              output_cost_per_token: null,
              provider: "openai",
            }
          : null,
        loading: false,
        error: null,
      },
    ],
    totals: {
      cost_per_request: hasResult ? 0.05 : 0,
      daily_cost: null,
      monthly_cost: null,
      margin_per_request: 0,
      daily_margin: null,
      monthly_margin: null,
    },
  };
}

describe("MultiExportDropdown", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should not render anything when no entries have results", () => {
    const { container } = renderWithProviders(
      <MultiExportDropdown multiResult={makeMultiResult(false)} />
    );
    expect(container.firstChild).toBeNull();
  });

  it("should render the Export button when at least one entry has a result", () => {
    renderWithProviders(<MultiExportDropdown multiResult={makeMultiResult(true)} />);
    expect(screen.getByRole("button", { name: /^export$/i })).toBeInTheDocument();
  });

  it("should show the export menu when the Export button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<MultiExportDropdown multiResult={makeMultiResult(true)} />);

    await user.click(screen.getByRole("button", { name: /^export$/i }));

    expect(screen.getByText("Export as PDF")).toBeInTheDocument();
    expect(screen.getByText("Export as CSV")).toBeInTheDocument();
  });

  it("should hide the export menu when the Export button is clicked again", async () => {
    const user = userEvent.setup();
    renderWithProviders(<MultiExportDropdown multiResult={makeMultiResult(true)} />);

    await user.click(screen.getByRole("button", { name: /^export$/i }));
    expect(screen.getByText("Export as PDF")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^export$/i }));
    expect(screen.queryByText("Export as PDF")).not.toBeInTheDocument();
  });

  it("should call exportMultiToPDF and close the menu when Export as PDF is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<MultiExportDropdown multiResult={makeMultiResult(true)} />);

    await user.click(screen.getByRole("button", { name: /^export$/i }));
    await user.click(screen.getByText("Export as PDF"));

    expect(exportMultiToPDF).toHaveBeenCalledTimes(1);
    expect(screen.queryByText("Export as PDF")).not.toBeInTheDocument();
  });

  it("should call exportMultiToCSV and close the menu when Export as CSV is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<MultiExportDropdown multiResult={makeMultiResult(true)} />);

    await user.click(screen.getByRole("button", { name: /^export$/i }));
    await user.click(screen.getByText("Export as CSV"));

    expect(exportMultiToCSV).toHaveBeenCalledTimes(1);
    expect(screen.queryByText("Export as CSV")).not.toBeInTheDocument();
  });

  it("should pass the multiResult to the export functions", async () => {
    const user = userEvent.setup();
    const multiResult = makeMultiResult(true);
    renderWithProviders(<MultiExportDropdown multiResult={multiResult} />);

    await user.click(screen.getByRole("button", { name: /^export$/i }));
    await user.click(screen.getByText("Export as PDF"));

    expect(exportMultiToPDF).toHaveBeenCalledWith(multiResult);
  });

  it("should close the menu when clicking outside", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <div>
        <MultiExportDropdown multiResult={makeMultiResult(true)} />
        <div data-testid="outside">Outside</div>
      </div>
    );

    await user.click(screen.getByRole("button", { name: /^export$/i }));
    expect(screen.getByText("Export as PDF")).toBeInTheDocument();

    fireEvent.mouseDown(screen.getByTestId("outside"));
    expect(screen.queryByText("Export as PDF")).not.toBeInTheDocument();
  });
});
