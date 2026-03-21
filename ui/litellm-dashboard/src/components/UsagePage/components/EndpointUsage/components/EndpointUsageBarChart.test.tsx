import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import EndpointUsageBarChart from "./EndpointUsageBarChart";

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

  function BarChart(_props: any) {
    return React.createElement("div", { "data-testid": "tremor-bar-chart" }, "Bar Chart");
  }
  (BarChart as any).displayName = "BarChart";

  return { Card, Title, BarChart };
});

vi.mock("@/components/common_components/chartUtils", () => ({
  CustomLegend: ({ categories }: any) => (
    <div data-testid="custom-legend">{categories.join(", ")}</div>
  ),
  CustomTooltip: () => <div data-testid="custom-tooltip">Tooltip</div>,
}));

describe("EndpointUsageBarChart", () => {
  it("should render", () => {
    render(<EndpointUsageBarChart />);

    expect(screen.getByTestId("tremor-card")).toBeInTheDocument();
    expect(screen.getByText("Success vs Failed Requests by Endpoint")).toBeInTheDocument();
    expect(screen.getByTestId("tremor-bar-chart")).toBeInTheDocument();
  });
});
