import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import CloudZeroCostTracking from "./CloudZeroCostTracking";

const mockUseCloudZeroSettings = vi.fn();

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: () => ({
    accessToken: "test-token",
  }),
}));

vi.mock("@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings", () => ({
  useCloudZeroSettings: () => mockUseCloudZeroSettings(),
}));

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: () => "http://test-proxy",
}));

describe("CloudZeroCostTracking", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });

    vi.clearAllMocks();
    mockUseCloudZeroSettings.mockReturnValue({
      data: null,
      isLoading: false,
      error: null,
    });
  });

  it("should render", async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <CloudZeroCostTracking />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByText("No CloudZero Integration Found")).toBeInTheDocument();
    });
  });
});
