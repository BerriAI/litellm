import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import { MetricWithMetadata } from "../../../types";
import EndpointUsageBarChart from "./EndpointUsageBarChart";

const metric = (successful: number, failed: number): MetricWithMetadata => ({
  metrics: {
    spend: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
    api_requests: successful + failed,
    successful_requests: successful,
    failed_requests: failed,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  },
  metadata: {},
  api_key_breakdown: {},
});

const endpointData = {
  "/chat/completions": metric(120, 5),
  "/embeddings": metric(40, 2),
};

describe("EndpointUsageBarChart", () => {
  it("renders the title and the header legend labels", () => {
    renderWithProviders(<EndpointUsageBarChart endpointData={endpointData} />);

    expect(screen.getByText("Success vs Failed Requests by Endpoint")).toBeInTheDocument();
    expect(screen.getByText("Successful Requests")).toBeInTheDocument();
    expect(screen.getByText("Failed Requests")).toBeInTheDocument();
  });

  it("renders stacked green and red bars per endpoint", () => {
    const { container } = renderWithProviders(<EndpointUsageBarChart endpointData={endpointData} />);

    expect(container.querySelectorAll(".recharts-bar")).toHaveLength(2);
    const rectangles = Array.from(container.querySelectorAll("path.recharts-rectangle"));
    expect(rectangles).toHaveLength(4);
    const fills = new Set(rectangles.map((rect) => rect.getAttribute("fill")));
    expect(fills).toEqual(new Set(["var(--color-green-500, #22c55e)", "var(--color-red-500, #ef4444)"]));

    const xPositions = rectangles.map((rect) => rect.getAttribute("d")?.split(",")[0]);
    expect(new Set(xPositions).size).toBe(2);
  });

  it("labels the x axis with endpoint names", () => {
    renderWithProviders(<EndpointUsageBarChart endpointData={endpointData} />);

    expect(screen.getAllByText("/chat/completions").length).toBeGreaterThan(0);
    expect(screen.getAllByText("/embeddings").length).toBeGreaterThan(0);
  });

  it("keeps the chart's own legend off; only the header legend is shown", () => {
    const { container } = renderWithProviders(<EndpointUsageBarChart endpointData={endpointData} />);

    expect(container.querySelector(".recharts-legend-wrapper")).toBeNull();
    expect(screen.queryByText("metrics.successful_requests")).not.toBeInTheDocument();
  });

  it("renders an empty chart without bars when endpointData is absent", () => {
    const { container } = renderWithProviders(<EndpointUsageBarChart />);

    expect(screen.getByText("Success vs Failed Requests by Endpoint")).toBeInTheDocument();
    expect(container.querySelectorAll("path.recharts-rectangle")).toHaveLength(0);
  });
});
