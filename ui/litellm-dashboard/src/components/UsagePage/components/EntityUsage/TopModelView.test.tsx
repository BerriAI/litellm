import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import TopModelView from "./TopModelView";

describe("TopModelView", () => {
  const mockSetTopModelsLimit = vi.fn();

  beforeEach(() => {
    mockSetTopModelsLimit.mockClear();
  });

  it("should render", () => {
    render(<TopModelView topModels={[]} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);
    expect(screen.getByText("Table View")).toBeInTheDocument();
  });

  it("should display table view button", () => {
    render(<TopModelView topModels={[]} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);
    expect(screen.getByRole("button", { name: "Table View" })).toBeInTheDocument();
  });

  it("should display chart view button", () => {
    render(<TopModelView topModels={[]} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);
    expect(screen.getByRole("button", { name: "Chart View" })).toBeInTheDocument();
  });

  it("should display all table column headers", () => {
    render(<TopModelView topModels={[]} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);
    expect(screen.getByText("Model")).toBeInTheDocument();
    expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
    expect(screen.getByText("Successful")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("Tokens")).toBeInTheDocument();
  });

  it("should display model data in table view", () => {
    render(
      <TopModelView
        topModels={[
          {
            key: "gpt-4",
            spend: 150.5,
            successful_requests: 100,
            failed_requests: 5,
            tokens: 50000,
          },
        ]}
        topModelsLimit={5}
        setTopModelsLimit={mockSetTopModelsLimit}
      />,
    );
    expect(screen.getByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByText("$150.50")).toBeInTheDocument();
    expect(screen.getByText("100")).toBeInTheDocument();
    const failedRequestsCell = screen
      .getAllByText("5")
      .find((el) => el.closest("span")?.classList.contains("text-red-600"));
    expect(failedRequestsCell).toBeDefined();
    expect(screen.getByText("50,000")).toBeInTheDocument();
  });

  it("should switch to chart view when chart view button is clicked", async () => {
    const user = userEvent.setup();
    render(<TopModelView topModels={[]} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);

    const chartViewButton = screen.getByRole("button", { name: "Chart View" });
    await user.click(chartViewButton);

    expect(chartViewButton).toHaveClass("bg-blue-100");
  });

  it("should switch to table view when table view button is clicked", async () => {
    const user = userEvent.setup();
    render(<TopModelView topModels={[]} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);

    const chartViewButton = screen.getByRole("button", { name: "Chart View" });
    const tableViewButton = screen.getByRole("button", { name: "Table View" });

    await user.click(chartViewButton);
    await user.click(tableViewButton);

    expect(tableViewButton).toHaveClass("bg-blue-100");
  });

  it("should call setTopModelsLimit when limit is changed via Segmented control", async () => {
    const user = userEvent.setup();
    render(<TopModelView topModels={[]} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);

    const limit10Radio = screen.getByRole("radio", { name: "10" });
    const limit10Label = limit10Radio.closest("label");
    if (limit10Label) {
      await user.click(limit10Label);
    } else {
      // Fallback: click the div with title="10"
      const limit10Div = screen.getByTitle("10");
      await user.click(limit10Div);
    }

    expect(mockSetTopModelsLimit).toHaveBeenCalledWith(10);
  });

  it("should display only top N models based on limit", () => {
    const manyModels = Array.from({ length: 10 }, (_, i) => ({
      key: `model-${i + 1}`,
      spend: 100 + i,
      successful_requests: 50 + i,
      failed_requests: 5 + i,
      tokens: 10000 + i * 1000,
    }));

    render(<TopModelView topModels={manyModels} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);

    expect(screen.getByText("model-1")).toBeInTheDocument();
    expect(screen.getByText("model-5")).toBeInTheDocument();
    expect(screen.queryByText("model-6")).not.toBeInTheDocument();
  });

  it("should display all models when limit is greater than model count", () => {
    const models = [
      {
        key: "model-1",
        spend: 100,
        successful_requests: 50,
        failed_requests: 5,
        tokens: 10000,
      },
      {
        key: "model-2",
        spend: 200,
        successful_requests: 60,
        failed_requests: 6,
        tokens: 20000,
      },
    ];

    render(<TopModelView topModels={models} topModelsLimit={10} setTopModelsLimit={mockSetTopModelsLimit} />);

    expect(screen.getByText("model-1")).toBeInTheDocument();
    expect(screen.getByText("model-2")).toBeInTheDocument();
  });

  it("should format spend values with two decimal places", () => {
    render(
      <TopModelView
        topModels={[
          {
            key: "model-1",
            spend: 123.456,
            successful_requests: 100,
            failed_requests: 5,
            tokens: 50000,
          },
        ]}
        topModelsLimit={5}
        setTopModelsLimit={mockSetTopModelsLimit}
      />,
    );
    expect(screen.getByText("$123.46")).toBeInTheDocument();
  });

  it("should display zero values correctly", () => {
    render(
      <TopModelView
        topModels={[
          {
            key: "model-1",
            spend: 0,
            successful_requests: 0,
            failed_requests: 0,
            tokens: 0,
          },
        ]}
        topModelsLimit={5}
        setTopModelsLimit={mockSetTopModelsLimit}
      />,
    );
    expect(screen.getByText("$0.00")).toBeInTheDocument();
    expect(screen.getAllByText("0").length).toBeGreaterThan(0);
  });

  it("should display successful requests with green styling", () => {
    render(
      <TopModelView
        topModels={[
          {
            key: "model-1",
            spend: 100,
            successful_requests: 50,
            failed_requests: 5,
            tokens: 10000,
          },
        ]}
        topModelsLimit={5}
        setTopModelsLimit={mockSetTopModelsLimit}
      />,
    );
    const successfulCell = screen
      .getAllByText("50")
      .find((el) => el.closest("span")?.classList.contains("text-green-600"));
    expect(successfulCell).toBeDefined();
  });

  it("should display failed requests with red styling", () => {
    render(
      <TopModelView
        topModels={[
          {
            key: "model-1",
            spend: 100,
            successful_requests: 50,
            failed_requests: 5,
            tokens: 10000,
          },
        ]}
        topModelsLimit={5}
        setTopModelsLimit={mockSetTopModelsLimit}
      />,
    );
    const failedCell = screen.getAllByText("5").find((el) => el.closest("span")?.classList.contains("text-red-600"));
    expect(failedCell).toBeDefined();
  });

  it("should format large token numbers with commas", () => {
    render(
      <TopModelView
        topModels={[
          {
            key: "model-1",
            spend: 100,
            successful_requests: 50,
            failed_requests: 5,
            tokens: 1234567,
          },
        ]}
        topModelsLimit={5}
        setTopModelsLimit={mockSetTopModelsLimit}
      />,
    );
    expect(screen.getByText("1,234,567")).toBeInTheDocument();
  });

  it("should handle empty model list", () => {
    render(<TopModelView topModels={[]} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);
    expect(screen.getByText("Model")).toBeInTheDocument();
    expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
  });

  it("should display dash for missing model key", () => {
    render(
      <TopModelView
        topModels={[
          {
            key: "",
            spend: 100,
            successful_requests: 50,
            failed_requests: 5,
            tokens: 10000,
          },
        ]}
        topModelsLimit={5}
        setTopModelsLimit={mockSetTopModelsLimit}
      />,
    );
    expect(screen.getByText("-")).toBeInTheDocument();
  });
});
