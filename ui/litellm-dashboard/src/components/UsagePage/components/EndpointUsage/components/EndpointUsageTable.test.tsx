import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import EndpointUsageTable from "./EndpointUsageTable";

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

    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Endpoint/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Successful \/ Failed/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Total Request/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Success Rate/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Total Tokens/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Spend/i })).toBeInTheDocument();

    expect(screen.getByText("endpoint-1")).toBeInTheDocument();
    expect(screen.getByText("95.00%")).toBeInTheDocument();
    expect(screen.getByText("8,000")).toBeInTheDocument();
    expect(screen.getByText(/\$\s*100\.50/)).toBeInTheDocument();
  });
});
