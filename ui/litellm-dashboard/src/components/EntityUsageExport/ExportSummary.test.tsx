import { renderWithProviders, screen } from "../../../tests/test-utils";
import ExportSummary from "./ExportSummary";

describe("ExportSummary", () => {
  it("should render", () => {
    const dateRange = {
      from: new Date("2024-01-01"),
      to: new Date("2024-01-31"),
    };
    const { container } = renderWithProviders(
      <ExportSummary dateRange={dateRange} selectedFilters={[]} />
    );
    expect(container).not.toBeEmptyDOMElement();
  });

  it("should display the date range", () => {
    const from = new Date(2024, 0, 1);
    const to = new Date(2024, 0, 31);
    const dateRange = { from, to };
    renderWithProviders(
      <ExportSummary dateRange={dateRange} selectedFilters={[]} />
    );
    expect(screen.getByText(new RegExp(from.toLocaleDateString()))).toBeInTheDocument();
    expect(screen.getByText(new RegExp(to.toLocaleDateString()))).toBeInTheDocument();
  });

  it("should show filter count when filters are selected", () => {
    const dateRange = {
      from: new Date("2024-01-01"),
      to: new Date("2024-01-31"),
    };
    renderWithProviders(
      <ExportSummary dateRange={dateRange} selectedFilters={["team-a", "team-b", "team-c"]} />
    );
    expect(screen.getByText(/3 filters/)).toBeInTheDocument();
  });

  it("should show singular 'filter' for one filter", () => {
    const dateRange = {
      from: new Date("2024-01-01"),
      to: new Date("2024-01-31"),
    };
    renderWithProviders(
      <ExportSummary dateRange={dateRange} selectedFilters={["team-a"]} />
    );
    expect(screen.getByText(/1 filter$/)).toBeInTheDocument();
  });

  it("should not show filter count when no filters selected", () => {
    const dateRange = {
      from: new Date("2024-01-01"),
      to: new Date("2024-01-31"),
    };
    renderWithProviders(
      <ExportSummary dateRange={dateRange} selectedFilters={[]} />
    );
    expect(screen.queryByText(/filter/)).not.toBeInTheDocument();
  });
});
