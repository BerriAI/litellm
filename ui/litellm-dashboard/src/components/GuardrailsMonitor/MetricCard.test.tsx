import { render, screen } from "@testing-library/react";
import { MetricCard } from "./MetricCard";

describe("MetricCard", () => {
  it("should render", () => {
    render(<MetricCard label="Total Requests" value={1234} />);
    expect(screen.getByText("Total Requests")).toBeInTheDocument();
  });

  it("should display the numeric value", () => {
    render(<MetricCard label="Total Requests" value={1234} />);
    expect(screen.getByText("1234")).toBeInTheDocument();
  });

  it("should display a string value", () => {
    render(<MetricCard label="Pass Rate" value="95.2%" />);
    expect(screen.getByText("95.2%")).toBeInTheDocument();
  });

  it("should display subtitle when provided", () => {
    render(<MetricCard label="Blocked" value={42} subtitle="Last 7 days" />);
    expect(screen.getByText("Last 7 days")).toBeInTheDocument();
  });

  it("should not display subtitle when not provided", () => {
    render(<MetricCard label="Blocked" value={42} />);
    expect(screen.queryByText(/days/)).not.toBeInTheDocument();
  });

  it("should display icon when provided", () => {
    render(<MetricCard label="Status" value="OK" icon={<span data-testid="icon">!</span>} />);
    expect(screen.getByTestId("icon")).toBeInTheDocument();
  });
});
