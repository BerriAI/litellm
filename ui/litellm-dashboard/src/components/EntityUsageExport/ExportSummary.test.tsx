import { render, screen } from "@testing-library/react";
import ExportSummary from "./ExportSummary";

describe("ExportSummary", () => {
  const dateRange = {
    from: new Date("2025-01-01"),
    to: new Date("2025-01-31"),
  };

  it("should render", () => {
    render(<ExportSummary dateRange={dateRange} selectedFilters={[]} />);
    expect(screen.getByText(/2025/)).toBeInTheDocument();
  });

  it("should display formatted date range", () => {
    render(<ExportSummary dateRange={dateRange} selectedFilters={[]} />);
    const text = screen.getByText(/\d+.*-.*\d+/);
    expect(text).toBeInTheDocument();
  });

  it("should show singular 'filter' for one filter", () => {
    render(<ExportSummary dateRange={dateRange} selectedFilters={["team-a"]} />);
    expect(screen.getByText(/1 filter$/)).toBeInTheDocument();
  });

  it("should show plural 'filters' for multiple filters", () => {
    render(<ExportSummary dateRange={dateRange} selectedFilters={["team-a", "team-b"]} />);
    expect(screen.getByText(/2 filters/)).toBeInTheDocument();
  });

  it("should not show filter text when no filters applied", () => {
    render(<ExportSummary dateRange={dateRange} selectedFilters={[]} />);
    expect(screen.queryByText(/filter/)).not.toBeInTheDocument();
  });
});
