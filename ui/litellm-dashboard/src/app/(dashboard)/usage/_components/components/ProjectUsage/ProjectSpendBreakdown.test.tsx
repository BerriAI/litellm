import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ProjectSpendBreakdown from "./ProjectSpendBreakdown";
import type { ProjectSpendRow } from "./projectUsageAggregations";

vi.mock("@/components/shared/chart_loader", () => ({
  ChartLoader: ({ isDateChanging }: { isDateChanging: boolean }) => (
    <div data-testid="chart-loader">{isDateChanging ? "Processing date selection..." : "Loading chart data..."}</div>
  ),
}));

const mockProjectSpend: ProjectSpendRow[] = [
  {
    project_id: "project-alpha",
    project_alias: "Project Alpha",
    spend: 150.5,
    requests: 100,
    successful_requests: 95,
    failed_requests: 5,
    tokens: 50000,
  },
  {
    project_id: "project-beta",
    project_alias: "Project Beta",
    spend: 200.75,
    requests: 120,
    successful_requests: 115,
    failed_requests: 5,
    tokens: 75000,
  },
];

describe("ProjectSpendBreakdown", () => {
  it("displays the title", () => {
    render(<ProjectSpendBreakdown loading={false} isDateChanging={false} projectSpend={[]} />);
    expect(screen.getByText("Spend by Project")).toBeInTheDocument();
  });

  it("shows the loader instead of chart content while loading", () => {
    render(<ProjectSpendBreakdown loading isDateChanging={false} projectSpend={[]} />);
    expect(screen.getByTestId("chart-loader")).toBeInTheDocument();
    expect(screen.queryByText("No project usage data")).not.toBeInTheDocument();
  });

  it("displays table headers", () => {
    render(<ProjectSpendBreakdown loading={false} isDateChanging={false} projectSpend={mockProjectSpend} />);
    expect(screen.getByText("Project")).toBeInTheDocument();
    expect(screen.getByText("Spend")).toBeInTheDocument();
    expect(screen.getByText("Successful")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("Tokens")).toBeInTheDocument();
  });

  it("displays each project's alias and formatted spend in the table", () => {
    render(<ProjectSpendBreakdown loading={false} isDateChanging={false} projectSpend={mockProjectSpend} />);
    expect(screen.getAllByText("Project Alpha").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Project Beta").length).toBeGreaterThan(0);
    expect(screen.getByText("$150.50")).toBeInTheDocument();
    expect(screen.getByText("$200.75")).toBeInTheDocument();
  });

  it("shows the empty message when no project has usage", () => {
    render(<ProjectSpendBreakdown loading={false} isDateChanging={false} projectSpend={[]} />);
    expect(screen.getByText("No project usage data")).toBeInTheDocument();
  });
});
