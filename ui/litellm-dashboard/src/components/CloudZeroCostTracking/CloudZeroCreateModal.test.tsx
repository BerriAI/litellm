import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import CloudZeroCreateModal from "./CloudZeroCreateModal";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: () => ({
    accessToken: "test-token",
  }),
}));

vi.mock("@/app/(dashboard)/hooks/cloudzero/useCloudZeroCreate", () => ({
  useCloudZeroCreate: () => ({
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
    },
  };
});

describe("CloudZeroCreateModal", () => {
  let queryClient: QueryClient;

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
        <CloudZeroCreateModal open={true} onOk={vi.fn()} onCancel={vi.fn()} />
      </QueryClientProvider>,
    );

    expect(screen.getByText("Create CloudZero Integration")).toBeInTheDocument();
    expect(screen.getByLabelText("CloudZero API Key")).toBeInTheDocument();
    expect(screen.getByLabelText("Connection ID")).toBeInTheDocument();
    expect(screen.getByLabelText("Timezone")).toBeInTheDocument();
  });
});
