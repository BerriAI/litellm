import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import EndpointUsageLineChart from "./EndpointUsageLineChart";

vi.mock("@tremor/react", async () => {
  const React = await import("react");

  function Card({ children }: any) {
    return React.createElement("div", { "data-testid": "tremor-card" }, children);
  }
  (Card as any).displayName = "Card";

  function Title({ children }: any) {
    return React.createElement("h2", { "data-testid": "tremor-title" }, children);
  }
  (Title as any).displayName = "Title";

  function LineChart(_props: any) {
    return React.createElement("div", { "data-testid": "tremor-line-chart" }, "Line Chart");
  }
  (LineChart as any).displayName = "LineChart";

  return { Card, Title, LineChart };
});

describe("EndpointUsageLineChart", () => {
  it("should render", () => {
    render(<EndpointUsageLineChart />);

    expect(screen.getByTestId("tremor-card")).toBeInTheDocument();
    expect(screen.getByText("Endpoint Usage Trends")).toBeInTheDocument();
    expect(screen.getByTestId("tremor-line-chart")).toBeInTheDocument();
  });
});
