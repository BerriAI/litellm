import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useCloudZeroDryRun } from "./useCloudZeroDryRun";

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

describe("useCloudZeroDryRun", () => {
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
    const { result } = renderHook(() => useCloudZeroDryRun(mockAccessToken), { wrapper });

    expect(result.current).toBeDefined();
  });

  it("should successfully perform dry run with custom limit", async () => {
    const mockResponse = { records_processed: 5, status: "success" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroDryRun(mockAccessToken), { wrapper });

    result.current.mutate({ limit: 20 });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockResponse);
    expect(fetchSpy).toHaveBeenCalledWith(`${mockProxyBaseUrl}/cloudzero/dry-run`, {
      method: "POST",
      headers: {
        [mockHeaderName]: `Bearer ${mockAccessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        limit: 20,
      }),
    });
  });

  it("should use default limit of 10 when limit is not provided", async () => {
    const mockResponse = { records_processed: 10, status: "success" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroDryRun(mockAccessToken), { wrapper });

    result.current.mutate({});

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockResponse);
    const callBody = JSON.parse((fetchSpy as any).mock.calls[0][1].body);
    expect(callBody.limit).toBe(10);
  });

  it("should handle error response with error.message", async () => {
    const errorResponse = { error: { message: "Dry run failed" } };
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => errorResponse,
    });

    const { result } = renderHook(() => useCloudZeroDryRun(mockAccessToken), { wrapper });

    result.current.mutate({ limit: 5 });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Dry run failed");
  });

  it("should handle error response with message field", async () => {
    const errorResponse = { message: "Invalid configuration" };
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => errorResponse,
    });

    const { result } = renderHook(() => useCloudZeroDryRun(mockAccessToken), { wrapper });

    result.current.mutate({ limit: 5 });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Invalid configuration");
  });

  it("should handle error response with detail field", async () => {
    const errorResponse = { detail: "Server error" };
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => errorResponse,
    });

    const { result } = renderHook(() => useCloudZeroDryRun(mockAccessToken), { wrapper });

    result.current.mutate({ limit: 5 });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Server error");
  });

  it("should handle error response with invalid JSON", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => {
        throw new Error("Invalid JSON");
      },
    });

    const { result } = renderHook(() => useCloudZeroDryRun(mockAccessToken), { wrapper });

    result.current.mutate({ limit: 5 });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Failed to perform dry run");
  });

  it("should handle network error", async () => {
    const networkError = new Error("Network request failed");
    (fetchSpy as any).mockRejectedValue(networkError);

    const { result } = renderHook(() => useCloudZeroDryRun(mockAccessToken), { wrapper });

    result.current.mutate({ limit: 5 });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(networkError);
  });

  it.each([
    ["empty string", ""],
    ["null", null],
  ])("should throw error when accessToken is %s", async (_, invalidToken) => {
    const { result } = renderHook(() => useCloudZeroDryRun(invalidToken as any), { wrapper });

    result.current.mutate({ limit: 5 });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Access token is required");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("should use relative URL when proxyBaseUrl is not set", async () => {
    mockGetProxyBaseUrl.mockReturnValue("");
    const mockResponse = { records_processed: 10 };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroDryRun(mockAccessToken), { wrapper });

    result.current.mutate({ limit: 5 });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(fetchSpy).toHaveBeenCalledWith("/cloudzero/dry-run", expect.any(Object));
  });
});
