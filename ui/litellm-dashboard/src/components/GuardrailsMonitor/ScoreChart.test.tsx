import React from "react";
import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../../../tests/test-utils";
import { ScoreChart } from "./ScoreChart";

vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  // Re-apply the global Button/Tooltip overrides from tests/setupTests.ts. A file-level
  // vi.mock fully replaces the setup-level mock, so without this the real Tremor Button
  // leaks through and its useTooltip(300) schedules a native setTimeout that can fire
  // post-teardown -> "window is not defined".
  return {
    ...actual,
    BarChart: ({ data, categories }: { data: any[]; categories: string[] }) => (
      <div data-testid="bar-chart">
        {data.map((d, i) => (
          <span key={i}>
            {d.date}: {categories.map((c) => `${c}=${d[c]}`).join(", ")}
          </span>
        ))}
      </div>
    ),
    Button: React.forwardRef<HTMLButtonElement, any>(({ children, ...props }, ref) => (
      <button {...props} ref={ref}>
        {children}
      </button>
    )),
    Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  };
});

describe("ScoreChart", () => {
  it("should render the title", () => {
    renderWithProviders(<ScoreChart />);

    expect(screen.getByText("Request Outcomes Over Time")).toBeInTheDocument();
  });

  it("should show empty state when no data is provided", () => {
    renderWithProviders(<ScoreChart />);

    expect(screen.getByText("No chart data for this period")).toBeInTheDocument();
  });

  it("should show empty state when data is an empty array", () => {
    renderWithProviders(<ScoreChart data={[]} />);

    expect(screen.getByText("No chart data for this period")).toBeInTheDocument();
  });

  it("should render the chart when data is provided", () => {
    const data = [
      { date: "2026-03-01", passed: 10, blocked: 2 },
      { date: "2026-03-02", passed: 15, blocked: 1 },
    ];

    renderWithProviders(<ScoreChart data={data} />);

    expect(screen.queryByText("No chart data for this period")).not.toBeInTheDocument();
    expect(screen.getByText(/2026-03-01/)).toBeInTheDocument();
    expect(screen.getByText(/2026-03-02/)).toBeInTheDocument();
  });
});
