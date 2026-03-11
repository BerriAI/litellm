import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import EndpointUsageTable from "./EndpointUsageTable";

vi.mock("antd", async () => {
  const React = await import("react");

  function Table({ columns, dataSource }: any) {
    return React.createElement(
      "div",
      { "data-testid": "antd-table" },
      columns?.map((col: any) =>
        React.createElement("div", { key: col.key, "data-testid": `column-${col.key}` }, col.title),
      ),
      dataSource?.map((row: any) =>
        React.createElement(
          "div",
          { key: row.key, "data-testid": `row-${row.key}` },
          React.createElement("div", null, row.endpoint),
        ),
      ),
    );
  }
  (Table as any).displayName = "Table";

  function Progress({ percent }: any) {
    return React.createElement("div", { "data-testid": "antd-progress", "data-percent": percent });
  }
  (Progress as any).displayName = "Progress";

  return { Table, Progress };
});

vi.mock("@/utils/dataUtils", () => ({
  formatNumberWithCommas: (value: number, decimals?: number) => {
    return value.toFixed(decimals || 0);
  },
}));

describe("EndpointUsageTable", () => {
  it("should render", () => {
    const mockEndpointData = {
      "endpoint-1": {
        metrics: {
          spend: 100.5,
          prompt_tokens: 5000,
          completion_tokens: 3000,
          total_tokens: 8000,
          api_requests: 100,
          successful_requests: 95,
          failed_requests: 5,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
        metadata: {},
        api_key_breakdown: {},
      },
    };

    render(<EndpointUsageTable endpointData={mockEndpointData} />);

    expect(screen.getByTestId("antd-table")).toBeInTheDocument();
    expect(screen.getByTestId("column-endpoint")).toBeInTheDocument();
    expect(screen.getByTestId("column-requests")).toBeInTheDocument();
    expect(screen.getByTestId("column-api_requests")).toBeInTheDocument();
    expect(screen.getByTestId("column-successRate")).toBeInTheDocument();
    expect(screen.getByTestId("column-total_tokens")).toBeInTheDocument();
    expect(screen.getByTestId("column-spend")).toBeInTheDocument();
  });
});
