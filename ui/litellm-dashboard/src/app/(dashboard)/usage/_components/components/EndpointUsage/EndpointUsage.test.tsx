import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import EndpointUsage from "./EndpointUsage";

vi.mock("./components/EndpointUsageBarChart", () => ({
  default: () => <div>Endpoint Usage Bar Chart</div>,
}));

vi.mock("./components/EndpointUsageLineChart", () => ({
  default: () => <div>Endpoint Usage Line Chart</div>,
}));

vi.mock("./components/EndpointUsageTable", () => ({
  default: () => <div>Endpoint Usage Table</div>,
}));

describe("EndpointUsage", () => {
  it("should render", () => {
    render(<EndpointUsage />);

    expect(screen.getByText("Endpoint Usage Table")).toBeInTheDocument();
    expect(screen.getByText("Endpoint Usage Bar Chart")).toBeInTheDocument();
    expect(screen.getByText("Endpoint Usage Line Chart")).toBeInTheDocument();
  });
});
