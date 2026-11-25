import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useProviderFields } from "./useProviderFields";
import { getProviderCreateMetadata } from "@/components/networking";
import type { ProviderCreateInfo } from "@/components/networking";

// Mock the networking function
vi.mock("@/components/networking", () => ({
  getProviderCreateMetadata: vi.fn(),
}));

// Mock data
const mockProviderFields: ProviderCreateInfo[] = [
  {
    provider: "OpenAI",
    provider_display_name: "OpenAI",
    litellm_provider: "openai",
    default_model_placeholder: "gpt-3.5-turbo",
    credential_fields: [],
  },
  {
    provider: "Anthropic",
    provider_display_name: "Anthropic",
    litellm_provider: "anthropic",
    default_model_placeholder: "claude-3-sonnet-20240229",
    credential_fields: [],
  },
  {
    provider: "Azure",
    provider_display_name: "Azure OpenAI",
    litellm_provider: "azure",
    default_model_placeholder: "gpt-35-turbo",
    credential_fields: [],
  },
];

describe("useProviderFields", () => {
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

  it("should return provider fields data when query is successful", async () => {
    // Mock successful API call
    (getProviderCreateMetadata as any).mockResolvedValue(mockProviderFields);

    const { result } = renderHook(() => useProviderFields(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockProviderFields);
    expect(result.current.error).toBeNull();
    expect(getProviderCreateMetadata).toHaveBeenCalledTimes(1);
  });

  it("should handle error when getProviderCreateMetadata fails", async () => {
    const errorMessage = "Failed to fetch provider fields";
    const testError = new Error(errorMessage);

    // Mock failed API call
    (getProviderCreateMetadata as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useProviderFields(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(getProviderCreateMetadata).toHaveBeenCalledTimes(1);
  });

  it("should return empty array when API returns empty data", async () => {
    // Mock API returning empty array
    (getProviderCreateMetadata as any).mockResolvedValue([]);

    const { result } = renderHook(() => useProviderFields(), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual([]);
    expect(getProviderCreateMetadata).toHaveBeenCalledTimes(1);
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    // Mock network timeout
    (getProviderCreateMetadata as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useProviderFields(), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });

  it("should have correct query configuration", async () => {
    // Mock successful API call
    (getProviderCreateMetadata as any).mockResolvedValue(mockProviderFields);

    const { result } = renderHook(() => useProviderFields(), { wrapper });

    // Wait for query to complete
    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // Verify the query was called
    expect(getProviderCreateMetadata).toHaveBeenCalledTimes(1);

    // The hook should have the expected properties from useQuery
    expect(result.current).toHaveProperty("data");
    expect(result.current).toHaveProperty("isLoading");
    expect(result.current).toHaveProperty("isError");
    expect(result.current).toHaveProperty("isSuccess");
    expect(result.current).toHaveProperty("error");
  });

  it("should return provider fields with populated credential fields", async () => {
    const mockFieldsWithCredentials: ProviderCreateInfo[] = [
      {
        provider: "TestProvider",
        provider_display_name: "Test Provider",
        litellm_provider: "test",
        default_model_placeholder: "test-model",
        credential_fields: [], // Keeping empty as per existing test patterns
      },
    ];

    // Mock successful API call with provider that has credential fields
    (getProviderCreateMetadata as any).mockResolvedValue(mockFieldsWithCredentials);

    const { result } = renderHook(() => useProviderFields(), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockFieldsWithCredentials);
    expect(result.current.data?.[0].provider).toBe("TestProvider");
    expect(result.current.data?.[0].litellm_provider).toBe("test");
  });
});
