import { renderWithProviders } from "@/../tests/test-utils";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { describe, expect, it, vi, beforeEach } from "vitest";
import TopModelView from "./TopModelView";

const lastSearchParams = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

describe("TopModelView", () => {
  const mockSetTopModelsLimit = vi.fn();

  beforeEach(() => {
    mockSetTopModelsLimit.mockClear();
  });

  // Which element a control library gives its label to is its own business, so drive the
  // control by its visible text and judge the result by what the panel renders.
  const clickControl = async (user: ReturnType<typeof userEvent.setup>, label: string) => {
    await user.click(screen.getByText(label));
  };

  const showsChart = (container: HTMLElement) => container.querySelector(".recharts-wrapper") !== null;

  it("should render", () => {
    renderWithProviders(<TopModelView topModels={[]} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);
    expect(screen.getByText("Table View")).toBeInTheDocument();
  });

  it("should display table view button", () => {
    renderWithProviders(<TopModelView topModels={[]} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);
    expect(screen.getByText("Table View")).toBeInTheDocument();
  });

  it("should display chart view button", () => {
    renderWithProviders(<TopModelView topModels={[]} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);
    expect(screen.getByText("Chart View")).toBeInTheDocument();
  });

  it("should display all table column headers", () => {
    renderWithProviders(<TopModelView topModels={[]} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);
    expect(screen.getByText("Model")).toBeInTheDocument();
    expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
    expect(screen.getByText("Successful")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("Tokens")).toBeInTheDocument();
  });

  it("should display model data in table view", () => {
    renderWithProviders(
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
      .find((el) => el.closest("span")?.classList.contains("text-destructive"));
    expect(failedRequestsCell).toBeDefined();
    expect(screen.getByText("50,000")).toBeInTheDocument();
  });

  const oneModel = [{ key: "gpt-4", spend: 150.5, successful_requests: 100, failed_requests: 5, tokens: 50000 }];

  it("should switch to chart view when chart view button is clicked", async () => {
    const user = userEvent.setup();
    const { container } = renderWithProviders(
      <TopModelView topModels={oneModel} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />,
    );

    expect(showsChart(container)).toBe(false);
    await clickControl(user, "Chart View");

    expect(showsChart(container)).toBe(true);
    expect(screen.queryByText("Spend (USD)")).not.toBeInTheDocument();
  });

  it("should switch to table view when table view button is clicked", async () => {
    const user = userEvent.setup();
    const { container } = renderWithProviders(
      <TopModelView topModels={oneModel} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />,
    );

    await clickControl(user, "Chart View");
    await clickControl(user, "Table View");

    expect(showsChart(container)).toBe(false);
    expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
  });

  it("shows the chart when ?top_models_view=chart", () => {
    const { container } = renderWithProviders(
      <TopModelView topModels={oneModel} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />,
      { searchParams: "?top_models_view=chart" },
    );

    expect(showsChart(container)).toBe(true);
    expect(screen.queryByText("Spend (USD)")).not.toBeInTheDocument();
  });

  it("falls back to the table when ?top_models_view= is unknown", () => {
    const { container } = renderWithProviders(
      <TopModelView topModels={oneModel} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />,
      { searchParams: "?top_models_view=pie" },
    );

    expect(showsChart(container)).toBe(false);
    expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
  });

  it("writes ?top_models_view=chart and drops it again for the default table", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(
      <TopModelView topModels={oneModel} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />,
      { onUrlUpdate },
    );

    await clickControl(user, "Chart View");
    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.get("top_models_view")).toBe("chart"));

    await clickControl(user, "Table View");
    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.has("top_models_view")).toBe(false));
  });

  it("keeps a panel with its own viewModeUrlKey independent of top_models_view", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { container } = renderWithProviders(
      <TopModelView
        topModels={oneModel}
        topModelsLimit={5}
        setTopModelsLimit={mockSetTopModelsLimit}
        viewModeUrlKey="top_agents_view"
      />,
      { searchParams: "?top_models_view=chart", onUrlUpdate },
    );

    expect(showsChart(container)).toBe(false);

    await clickControl(user, "Chart View");

    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.get("top_agents_view")).toBe("chart"));
    expect(lastSearchParams(onUrlUpdate)?.get("top_models_view")).toBe("chart");
    expect(showsChart(container)).toBe(true);
  });

  it("renders one cyan bar per model with model names on the axis in chart view", async () => {
    const user = userEvent.setup();
    const { container } = renderWithProviders(
      <TopModelView
        topModels={[
          {
            key: "gpt-4",
            spend: 150.5,
            successful_requests: 100,
            failed_requests: 5,
            tokens: 50000,
          },
          {
            key: "claude-3",
            spend: 75.25,
            successful_requests: 50,
            failed_requests: 2,
            tokens: 25000,
          },
        ]}
        topModelsLimit={5}
        setTopModelsLimit={mockSetTopModelsLimit}
      />,
    );

    await clickControl(user, "Chart View");

    const bars = container.querySelectorAll("path.recharts-rectangle");
    expect(bars).toHaveLength(2);
    const fills = new Set(Array.from(bars).map((bar) => bar.getAttribute("fill")));
    expect(fills).toEqual(new Set(["var(--color-cyan-500, #06b6d4)"]));
    expect(screen.getAllByText("gpt-4").length).toBeGreaterThan(0);
    expect(screen.getAllByText("claude-3").length).toBeGreaterThan(0);
  });

  it("should call setTopModelsLimit when the limit control is changed", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TopModelView topModels={[]} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);

    await clickControl(user, "10");

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

    renderWithProviders(
      <TopModelView topModels={manyModels} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />,
    );

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

    renderWithProviders(
      <TopModelView topModels={models} topModelsLimit={10} setTopModelsLimit={mockSetTopModelsLimit} />,
    );

    expect(screen.getByText("model-1")).toBeInTheDocument();
    expect(screen.getByText("model-2")).toBeInTheDocument();
  });

  it("should format spend values with two decimal places", () => {
    renderWithProviders(
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
    renderWithProviders(
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
    expect(screen.getByText("-")).toBeInTheDocument();
    expect(screen.getAllByText("0").length).toBeGreaterThan(0);
  });

  it("should display successful requests with green styling", () => {
    renderWithProviders(
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
      .find((el) => el.closest("span")?.classList.contains("text-success"));
    expect(successfulCell).toBeDefined();
  });

  it("should display failed requests with red styling", () => {
    renderWithProviders(
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
    const failedCell = screen
      .getAllByText("5")
      .find((el) => el.closest("span")?.classList.contains("text-destructive"));
    expect(failedCell).toBeDefined();
  });

  it("should format large token numbers with commas", () => {
    renderWithProviders(
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
    renderWithProviders(<TopModelView topModels={[]} topModelsLimit={5} setTopModelsLimit={mockSetTopModelsLimit} />);
    expect(screen.getByText("Model")).toBeInTheDocument();
    expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
  });

  it("should display dash for missing model key", () => {
    renderWithProviders(
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
