import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import KeyModelUsageView from "./KeyModelUsageView";
import { TopModelData } from "../types";

describe("KeyModelUsageView", () => {
  const mockTopModels: TopModelData[] = [
    {
      model: "gpt-4",
      spend: 150.5,
      requests: 105,
      successful_requests: 100,
      failed_requests: 5,
      tokens: 50000,
    },
    {
      model: "gpt-3.5-turbo",
      spend: 75.25,
      requests: 200,
      successful_requests: 195,
      failed_requests: 5,
      tokens: 100000,
    },
  ];

  it("should render", () => {
    render(<KeyModelUsageView topModels={mockTopModels} />);
    expect(screen.getByText("Model Usage")).toBeInTheDocument();
  });

  it("should return null when topModels is empty", () => {
    const { container } = render(<KeyModelUsageView topModels={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("should display Model Usage title", () => {
    render(<KeyModelUsageView topModels={mockTopModels} />);
    expect(screen.getByText("Model Usage")).toBeInTheDocument();
  });

  it("should display Table view button", () => {
    render(<KeyModelUsageView topModels={mockTopModels} />);
    expect(screen.getByRole("button", { name: "Table" })).toBeInTheDocument();
  });

  it("should display Chart view button", () => {
    render(<KeyModelUsageView topModels={mockTopModels} />);
    expect(screen.getByRole("button", { name: "Chart" })).toBeInTheDocument();
  });

  it("should default to table view", () => {
    render(<KeyModelUsageView topModels={mockTopModels} />);
    const tableButton = screen.getByRole("button", { name: "Table" });
    expect(tableButton).toHaveClass("bg-blue-100");
  });

  it("should display all table column headers", () => {
    render(<KeyModelUsageView topModels={mockTopModels} />);
    expect(screen.getByText("Model")).toBeInTheDocument();
    expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
    expect(screen.getByText("Successful")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("Tokens")).toBeInTheDocument();
  });

  it("should display model data in table view", () => {
    render(<KeyModelUsageView topModels={mockTopModels} />);
    expect(screen.getByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByText("gpt-3.5-turbo")).toBeInTheDocument();
    expect(screen.getByText("$150.50")).toBeInTheDocument();
    expect(screen.getByText("$75.25")).toBeInTheDocument();
  });

  it("should format spend values with two decimal places", () => {
    const modelsWithDecimalSpend: TopModelData[] = [
      {
        model: "test-model",
        spend: 123.456,
        requests: 10,
        successful_requests: 10,
        failed_requests: 0,
        tokens: 1000,
      },
    ];
    render(<KeyModelUsageView topModels={modelsWithDecimalSpend} />);
    expect(screen.getByText("$123.46")).toBeInTheDocument();
  });

  it("should format large spend values with commas", () => {
    const modelsWithLargeSpend: TopModelData[] = [
      {
        model: "test-model",
        spend: 1234567.89,
        requests: 10,
        successful_requests: 10,
        failed_requests: 0,
        tokens: 1000,
      },
    ];
    render(<KeyModelUsageView topModels={modelsWithLargeSpend} />);
    expect(screen.getByText("$1,234,567.89")).toBeInTheDocument();
  });

  it("should display successful requests with green styling", () => {
    render(<KeyModelUsageView topModels={mockTopModels} />);
    const successfulElements = screen.getAllByText("100");
    const greenElement = successfulElements.find((el) => el.closest("span")?.classList.contains("text-green-600"));
    expect(greenElement).toBeDefined();
  });

  it("should display failed requests with red styling", () => {
    render(<KeyModelUsageView topModels={mockTopModels} />);
    const failedElements = screen.getAllByText("5");
    const redElement = failedElements.find((el) => el.closest("span")?.classList.contains("text-red-600"));
    expect(redElement).toBeDefined();
  });

  it("should format token numbers with commas", () => {
    const modelsWithLargeTokens: TopModelData[] = [
      {
        model: "test-model",
        spend: 100,
        requests: 10,
        successful_requests: 10,
        failed_requests: 0,
        tokens: 1234567,
      },
    ];
    render(<KeyModelUsageView topModels={modelsWithLargeTokens} />);
    expect(screen.getByText("1,234,567")).toBeInTheDocument();
  });

  it("should display dash for missing model value", () => {
    const modelsWithMissingModel: TopModelData[] = [
      {
        model: "",
        spend: 100,
        requests: 10,
        successful_requests: 10,
        failed_requests: 0,
        tokens: 1000,
      },
    ];
    render(<KeyModelUsageView topModels={modelsWithMissingModel} />);
    expect(screen.getByText("-")).toBeInTheDocument();
  });

  it("should display zero values correctly", () => {
    const modelsWithZeros: TopModelData[] = [
      {
        model: "test-model",
        spend: 0,
        requests: 0,
        successful_requests: 0,
        failed_requests: 0,
        tokens: 0,
      },
    ];
    render(<KeyModelUsageView topModels={modelsWithZeros} />);
    expect(screen.getByText("$0.00")).toBeInTheDocument();
    expect(screen.getAllByText("0").length).toBeGreaterThan(0);
  });

  it("should switch to chart view when chart button is clicked", async () => {
    const user = userEvent.setup();
    render(<KeyModelUsageView topModels={mockTopModels} />);

    const chartButton = screen.getByRole("button", { name: "Chart" });
    await user.click(chartButton);

    expect(chartButton).toHaveClass("bg-blue-100");
    const tableButton = screen.getByRole("button", { name: "Table" });
    expect(tableButton).not.toHaveClass("bg-blue-100");
  });

  it("should switch back to table view when table button is clicked", async () => {
    const user = userEvent.setup();
    render(<KeyModelUsageView topModels={mockTopModels} />);

    const chartButton = screen.getByRole("button", { name: "Chart" });
    const tableButton = screen.getByRole("button", { name: "Table" });

    await user.click(chartButton);
    await user.click(tableButton);

    expect(tableButton).toHaveClass("bg-blue-100");
    expect(chartButton).not.toHaveClass("bg-blue-100");
  });

  it("should display chart when chart view is selected", async () => {
    const user = userEvent.setup();
    render(<KeyModelUsageView topModels={mockTopModels} />);

    const chartButton = screen.getByRole("button", { name: "Chart" });
    await user.click(chartButton);

    expect(screen.queryByText("Model")).not.toBeInTheDocument();
  });

  it("should display table when table view is selected", () => {
    render(<KeyModelUsageView topModels={mockTopModels} />);
    expect(screen.getByText("Model")).toBeInTheDocument();
    expect(screen.getByText("gpt-4")).toBeInTheDocument();
  });

  it("should handle multiple model entries", () => {
    const manyModels: TopModelData[] = Array.from({ length: 10 }, (_, i) => ({
      model: `model-${i + 1}`,
      spend: 100 + i,
      requests: 50 + i,
      successful_requests: 45 + i,
      failed_requests: 5,
      tokens: 10000 + i * 1000,
    }));

    render(<KeyModelUsageView topModels={manyModels} />);
    expect(screen.getByText("model-1")).toBeInTheDocument();
    expect(screen.getByText("model-10")).toBeInTheDocument();
  });

  it("should format successful requests with toLocaleString", () => {
    const modelsWithLargeNumbers: TopModelData[] = [
      {
        model: "test-model",
        spend: 100,
        requests: 1000,
        successful_requests: 999999,
        failed_requests: 1,
        tokens: 1000,
      },
    ];
    render(<KeyModelUsageView topModels={modelsWithLargeNumbers} />);
    expect(screen.getByText("999,999")).toBeInTheDocument();
  });

  it("should format failed requests with toLocaleString", () => {
    const modelsWithLargeNumbers: TopModelData[] = [
      {
        model: "test-model",
        spend: 100,
        requests: 1000,
        successful_requests: 1,
        failed_requests: 999999,
        tokens: 1000,
      },
    ];
    render(<KeyModelUsageView topModels={modelsWithLargeNumbers} />);
    expect(screen.getByText("999,999")).toBeInTheDocument();
  });

  it("should display zero for missing successful_requests", () => {
    const modelsWithMissingFields: TopModelData[] = [
      {
        model: "test-model",
        spend: 100,
        requests: 10,
        successful_requests: undefined as any,
        failed_requests: 0,
        tokens: 1000,
      },
    ];
    render(<KeyModelUsageView topModels={modelsWithMissingFields} />);
    const zeroElements = screen.getAllByText("0");
    const successfulZero = zeroElements.find((el) => el.closest("span")?.classList.contains("text-green-600"));
    expect(successfulZero).toBeDefined();
  });

  it("should display zero for missing failed_requests", () => {
    const modelsWithMissingFields: TopModelData[] = [
      {
        model: "test-model",
        spend: 100,
        requests: 10,
        successful_requests: 10,
        failed_requests: undefined as any,
        tokens: 1000,
      },
    ];
    render(<KeyModelUsageView topModels={modelsWithMissingFields} />);
    expect(screen.getAllByText("0").length).toBeGreaterThan(0);
  });

  it("should display zero for missing tokens", () => {
    const modelsWithMissingFields: TopModelData[] = [
      {
        model: "test-model",
        spend: 100,
        requests: 10,
        successful_requests: 10,
        failed_requests: 0,
        tokens: undefined as any,
      },
    ];
    render(<KeyModelUsageView topModels={modelsWithMissingFields} />);
    expect(screen.getAllByText("0").length).toBeGreaterThan(0);
  });
});
