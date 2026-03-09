import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useModelCostMap } from "./useModelCostMap";
import { modelCostMap } from "@/components/networking";

// Mock the networking function
vi.mock("@/components/networking", () => ({
  modelCostMap: vi.fn(),
}));

// Mock data
const mockModelCostData: Record<string, any> = {
  "gpt-3.5-turbo": {
    litellm_provider: "openai",
    input_cost_per_token: 0.0015,
    output_cost_per_token: 0.002,
  },
  "claude-3-sonnet-20240229": {
    litellm_provider: "anthropic",
    input_cost_per_token: 0.003,
    output_cost_per_token: 0.015,
  },
};

describe("useModelCostMap", () => {
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

  it("should return model cost map data when query is successful", async () => {
    // Mock successful API call
    (modelCostMap as any).mockResolvedValue(mockModelCostData);

    const { result } = renderHook(() => useModelCostMap(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockModelCostData);
    expect(result.current.error).toBeNull();
    expect(modelCostMap).toHaveBeenCalledTimes(1);
  });

  it("should handle error when modelCostMap fails", async () => {
    const errorMessage = "Failed to fetch model cost map";
    const testError = new Error(errorMessage);

    // Mock failed API call
    (modelCostMap as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useModelCostMap(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(modelCostMap).toHaveBeenCalledTimes(1);
  });

  it("should return empty object when API returns empty data", async () => {
    // Mock API returning empty object
    (modelCostMap as any).mockResolvedValue({});

    const { result } = renderHook(() => useModelCostMap(), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual({});
    expect(modelCostMap).toHaveBeenCalledTimes(1);
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    // Mock network timeout
    (modelCostMap as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useModelCostMap(), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });

  it("should have correct query configuration", async () => {
    // Mock successful API call
    (modelCostMap as any).mockResolvedValue(mockModelCostData);

    const { result } = renderHook(() => useModelCostMap(), { wrapper });

    // Wait for query to complete
    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // Verify the query was called
    expect(modelCostMap).toHaveBeenCalledTimes(1);

    // The hook should have the expected properties from useQuery
    expect(result.current).toHaveProperty("data");
    expect(result.current).toHaveProperty("isLoading");
    expect(result.current).toHaveProperty("isError");
    expect(result.current).toHaveProperty("isSuccess");
    expect(result.current).toHaveProperty("error");
  });
});
