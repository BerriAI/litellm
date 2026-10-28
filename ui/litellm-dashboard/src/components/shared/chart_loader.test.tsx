import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ChartLoader } from "./chart_loader";

describe("ChartLoader", () => {
  it("should render", () => {
    render(<ChartLoader />);
    expect(screen.getByText("Loading chart data...")).toBeInTheDocument();
  });

  it("should show loading text by default", () => {
    render(<ChartLoader />);
    expect(screen.getByText("Fetching your data")).toBeInTheDocument();
  });

  it("should show date-changing text when isDateChanging is true", () => {
    render(<ChartLoader isDateChanging />);
    expect(screen.getByText("Processing date selection...")).toBeInTheDocument();
    expect(screen.getByText("This will only take a moment")).toBeInTheDocument();
  });

  it("should not show default loading text when isDateChanging is true", () => {
    render(<ChartLoader isDateChanging />);
    expect(screen.queryByText("Loading chart data...")).not.toBeInTheDocument();
    expect(screen.queryByText("Fetching your data")).not.toBeInTheDocument();
  });
});
