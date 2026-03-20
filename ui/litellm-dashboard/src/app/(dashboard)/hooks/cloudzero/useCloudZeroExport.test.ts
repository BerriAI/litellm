import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useCloudZeroExport } from "./useCloudZeroExport";

const {
  mockProxyBaseUrl,
  mockAccessToken,
  mockHeaderName,
  mockGetProxyBaseUrl,
  mockGetGlobalLitellmHeaderName,
} = vi.hoisted(() => {
  const mockProxyBaseUrl = "https://proxy.example.com";
  const mockAccessToken = "test-access-token";
  const mockHeaderName = "X-LiteLLM-API-Key";
  const mockGetProxyBaseUrl = vi.fn(() => mockProxyBaseUrl);
  const mockGetGlobalLitellmHeaderName = vi.fn(() => mockHeaderName);

  return {
    mockProxyBaseUrl,
    mockAccessToken,
    mockHeaderName,
    mockGetProxyBaseUrl,
    mockGetGlobalLitellmHeaderName,
  };
});

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: mockGetProxyBaseUrl,
  getGlobalLitellmHeaderName: mockGetGlobalLitellmHeaderName,
}));

describe("useCloudZeroExport", () => {
  let queryClient: QueryClient;
  let fetchSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
        mutations: {
          retry: false,
        },
      },
    });

    vi.clearAllMocks();

    fetchSpy = vi.fn();
    global.fetch = fetchSpy;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should render", () => {
    const { result } = renderHook(() => useCloudZeroExport(mockAccessToken), { wrapper });

    expect(result.current).toBeDefined();
  });

  it("should successfully export data with custom operation", async () => {
    const mockResponse = { records_exported: 100, status: "success" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroExport(mockAccessToken), { wrapper });

    result.current.mutate({ operation: "replace_daily" });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockResponse);
    expect(fetchSpy).toHaveBeenCalledWith(`${mockProxyBaseUrl}/cloudzero/export`, {
      method: "POST",
      headers: {
        [mockHeaderName]: `Bearer ${mockAccessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        operation: "replace_daily",
      }),
    });
  });

  it("should use default operation of replace_hourly when operation is not provided", async () => {
    const mockResponse = { records_exported: 50, status: "success" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroExport(mockAccessToken), { wrapper });

    result.current.mutate({});

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockResponse);
    const callBody = JSON.parse((fetchSpy as any).mock.calls[0][1].body);
    expect(callBody.operation).toBe("replace_hourly");
  });

  it("should handle error response with error.message", async () => {
    const errorResponse = { error: { message: "Export failed" } };
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => errorResponse,
    });

    const { result } = renderHook(() => useCloudZeroExport(mockAccessToken), { wrapper });

    result.current.mutate({ operation: "replace_daily" });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Export failed");
  });

  it("should handle error response with message field", async () => {
    const errorResponse = { message: "Invalid operation" };
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => errorResponse,
    });

    const { result } = renderHook(() => useCloudZeroExport(mockAccessToken), { wrapper });

    result.current.mutate({ operation: "invalid_op" });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Invalid operation");
  });

  it("should handle error response with detail field", async () => {
    const errorResponse = { detail: "Server error occurred" };
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => errorResponse,
    });

    const { result } = renderHook(() => useCloudZeroExport(mockAccessToken), { wrapper });

    result.current.mutate({ operation: "replace_daily" });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Server error occurred");
  });

  it("should handle error response with invalid JSON", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => {
        throw new Error("Invalid JSON");
      },
    });

    const { result } = renderHook(() => useCloudZeroExport(mockAccessToken), { wrapper });

    result.current.mutate({ operation: "replace_daily" });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Failed to export data");
  });

  it("should handle network error", async () => {
    const networkError = new Error("Network request failed");
    (fetchSpy as any).mockRejectedValue(networkError);

    const { result } = renderHook(() => useCloudZeroExport(mockAccessToken), { wrapper });

    result.current.mutate({ operation: "replace_daily" });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(networkError);
  });

  it.each([
    ["empty string", ""],
    ["null", null],
  ])("should throw error when accessToken is %s", async (_, invalidToken) => {
    const { result } = renderHook(() => useCloudZeroExport(invalidToken as any), { wrapper });

    result.current.mutate({ operation: "replace_daily" });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Access token is required");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("should use relative URL when proxyBaseUrl is not set", async () => {
    mockGetProxyBaseUrl.mockReturnValue("");
    const mockResponse = { records_exported: 50 };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroExport(mockAccessToken), { wrapper });

    result.current.mutate({ operation: "replace_daily" });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(fetchSpy).toHaveBeenCalledWith("/cloudzero/export", expect.any(Object));
  });
});
