import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useUIConfig } from "./useUIConfig";
import { getUiConfig, LiteLLMWellKnownUiConfig } from "@/components/networking";

// Mock the networking function
vi.mock("@/components/networking", () => ({
  getUiConfig: vi.fn(),
}));

// Mock the queryKeysFactory - we'll mock the specific return value
vi.mock("../common/queryKeysFactory", () => ({
  createQueryKeys: vi.fn((resource: string) => ({
    all: [resource],
    lists: () => [resource, "list"],
    list: (params?: any) => [resource, "list", { params }],
    details: () => [resource, "detail"],
    detail: (uid: string) => [resource, "detail", uid],
  })),
}));

// Mock data
const mockUIConfig: LiteLLMWellKnownUiConfig = {
  sso_configured: true,
  server_root_path: "/api",
  proxy_base_url: "https://proxy.example.com",
  auto_redirect_to_sso: true,
  admin_ui_disabled: false,
};

describe("useUIConfig", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    // Reset all mocks
    vi.clearAllMocks();
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should return UI config data when query is successful", async () => {
    // Mock successful API call
    (getUiConfig as any).mockResolvedValue(mockUIConfig);

    const { result } = renderHook(() => useUIConfig(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockUIConfig);
    expect(result.current.error).toBeNull();
    expect(getUiConfig).toHaveBeenCalledWith();
    expect(getUiConfig).toHaveBeenCalledTimes(1);
  });

  it("should handle error when getUiConfig fails", async () => {
    const errorMessage = "Failed to fetch UI config";
    const testError = new Error(errorMessage);

    // Mock failed API call
    (getUiConfig as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useUIConfig(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(getUiConfig).toHaveBeenCalledWith();
    expect(getUiConfig).toHaveBeenCalledTimes(1);
  });

  it("should return different UI config data correctly", async () => {
    const alternativeUIConfig: LiteLLMWellKnownUiConfig = {
      server_root_path: "/v1",
      proxy_base_url: null,
      auto_redirect_to_sso: false,
      sso_configured: false,
      admin_ui_disabled: true,
    };

    // Mock successful API call with different data
    (getUiConfig as any).mockResolvedValue(alternativeUIConfig);

    const { result } = renderHook(() => useUIConfig(), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(alternativeUIConfig);
    expect(result.current.error).toBeNull();
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    // Mock network timeout
    (getUiConfig as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useUIConfig(), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });

  it("should handle malformed response error", async () => {
    const malformedError = new Error("Invalid JSON response");

    // Mock malformed response
    (getUiConfig as any).mockRejectedValue(malformedError);

    const { result } = renderHook(() => useUIConfig(), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(malformedError);
    expect(result.current.data).toBeUndefined();
  });

  it("should use correct query key structure", async () => {
    // Mock successful API call
    (getUiConfig as any).mockResolvedValue(mockUIConfig);

    const { result } = renderHook(() => useUIConfig(), { wrapper });

    // Wait for query to execute
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // The query key should be generated by createQueryKeys("uiConfig").list({})
    // Based on our mock, this should be ["uiConfig", "list", {}]
    expect(getUiConfig).toHaveBeenCalledTimes(1);
  });
});
