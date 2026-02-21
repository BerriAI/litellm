import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CloudZeroIntegrationSettings } from "./CloudZeroIntegrationSettings";
import { CloudZeroSettings } from "./types";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: () => ({
    accessToken: "test-token",
  }),
}));

vi.mock("@/app/(dashboard)/hooks/cloudzero/useCloudZeroDryRun", () => ({
  useCloudZeroDryRun: () => ({
    mutate: vi.fn(),
    isPending: false,
    data: null,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/cloudzero/useCloudZeroExport", () => ({
  useCloudZeroExport: () => ({
    mutate: vi.fn(),
    isPending: false,
  }),
}));

vi.mock("antd", async () => {
  const actual = await vi.importActual("antd");
  return {
    ...actual,
    message: {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn(),
    },
  };
});

describe("CloudZeroIntegrationSettings", () => {
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
        <CloudZeroIntegrationSettings settings={mockSettings} onSettingsUpdated={vi.fn()} />
      </QueryClientProvider>,
    );

    expect(screen.getByText("CloudZero Configuration")).toBeInTheDocument();
    expect(screen.getByText("API Key (Redacted)")).toBeInTheDocument();
    expect(screen.getByText("Connection ID")).toBeInTheDocument();
    expect(screen.getByText("Timezone")).toBeInTheDocument();
  });

  it("should display the correct values from settings", () => {
    render(
      <QueryClientProvider client={queryClient}>
        <CloudZeroIntegrationSettings settings={mockSettings} onSettingsUpdated={vi.fn()} />
      </QueryClientProvider>,
    );

    expect(screen.getByText(mockSettings.api_key_masked)).toBeInTheDocument();
    expect(screen.getByText(mockSettings.connection_id)).toBeInTheDocument();
  });
});
