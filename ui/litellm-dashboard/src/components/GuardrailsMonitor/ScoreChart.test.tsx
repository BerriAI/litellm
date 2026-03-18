import { render, screen } from "@testing-library/react";
import { ScoreChart } from "./ScoreChart";

vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  return {
    ...actual,
    BarChart: ({ data, categories }: { data: unknown[]; categories: string[] }) => (
      <div data-testid="bar-chart" data-categories={categories.join(",")}>
        {data.length} data points
      </div>
    ),
  };
});

describe("ScoreChart", () => {
  it("should render", () => {
    render(<ScoreChart />);
    expect(screen.getByText("Request Outcomes Over Time")).toBeInTheDocument();
  });

  it("should show empty state when no data provided", () => {
    render(<ScoreChart />);
    expect(screen.getByText("No chart data for this period")).toBeInTheDocument();
  });

  it("should show empty state when data is an empty array", () => {
    render(<ScoreChart data={[]} />);
    expect(screen.getByText("No chart data for this period")).toBeInTheDocument();
  });

  it("should render chart when data is provided", () => {
    const data = [
      { date: "2025-01-01", passed: 100, blocked: 5 },
      { date: "2025-01-02", passed: 120, blocked: 3 },
    ];
    render(<ScoreChart data={data} />);
    expect(screen.getByTestId("bar-chart")).toBeInTheDocument();
    expect(screen.getByText("2 data points")).toBeInTheDocument();
  });

  it("should pass correct categories to the chart", () => {
    const data = [{ date: "2025-01-01", passed: 100, blocked: 5 }];
    render(<ScoreChart data={data} />);
    expect(screen.getByTestId("bar-chart")).toHaveAttribute("data-categories", "passed,blocked");
  });
});
