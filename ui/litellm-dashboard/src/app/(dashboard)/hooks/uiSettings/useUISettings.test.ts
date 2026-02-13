import { getUiSettings } from "@/components/networking";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useUISettings } from "./useUISettings";

// Mock the networking function
vi.mock("@/components/networking", () => ({
  getUiSettings: vi.fn(),
}));

// Mock data
const mockUISettings: Record<string, any> = {
  theme: "dark",
  language: "en",
  notifications: true,
  dashboard_layout: "compact",
  api_keys_visible: false,
};

describe("useUISettings", () => {
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

  it("should return UI settings data when query is successful", async () => {
    // Mock successful API call
    (getUiSettings as any).mockResolvedValue(mockUISettings);

    const { result } = renderHook(() => useUISettings(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockUISettings);
    expect(result.current.error).toBeNull();
    expect(getUiSettings).toHaveBeenCalledWith();
    expect(getUiSettings).toHaveBeenCalledTimes(1);
  });

  it("should handle error when getUiSettings fails", async () => {
    const errorMessage = "Failed to fetch UI settings";
    const testError = new Error(errorMessage);

    // Mock failed API call
    (getUiSettings as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useUISettings(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(getUiSettings).toHaveBeenCalledWith();
    expect(getUiSettings).toHaveBeenCalledTimes(1);
  });

  it("should return empty object when API returns empty settings", async () => {
    // Mock API returning empty object
    (getUiSettings as any).mockResolvedValue({});

    const { result } = renderHook(() => useUISettings(), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual({});
    expect(getUiSettings).toHaveBeenCalledWith();
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    // Mock network timeout
    (getUiSettings as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useUISettings(), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });
});
