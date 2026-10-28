import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import CloudZeroUpdateModal from "./CloudZeroUpdateModal";
import { CloudZeroSettings } from "./types";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: () => ({
    accessToken: "test-token",
  }),
}));

vi.mock("@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings", () => ({
  useCloudZeroUpdateSettings: () => ({
    mutate: vi.fn(),
    isPending: false,
  }),
}));

describe("CloudZeroUpdateModal", () => {
  let queryClient: QueryClient;
  const mockSettings: CloudZeroSettings = {
    connection_id: "test-connection-id",
    api_key_masked: "****",
    timezone: "UTC",
    status: "Active",
  };

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
  });

  it("should render", () => {
    render(
      <QueryClientProvider client={queryClient}>
        <CloudZeroUpdateModal open={true} onOk={vi.fn()} onCancel={vi.fn()} settings={mockSettings} />
      </QueryClientProvider>,
    );

    expect(screen.getByText("Edit CloudZero Integration")).toBeInTheDocument();
    expect(screen.getByLabelText("CloudZero API Key")).toBeInTheDocument();
    expect(screen.getByLabelText("Connection ID")).toBeInTheDocument();
    expect(screen.getByLabelText("Timezone")).toBeInTheDocument();
  });
});
