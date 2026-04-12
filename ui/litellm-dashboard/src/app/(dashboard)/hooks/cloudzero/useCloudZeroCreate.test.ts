import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useCloudZeroCreate } from "./useCloudZeroCreate";

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

describe("useCloudZeroCreate", () => {
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
    const { result } = renderHook(() => useCloudZeroCreate(mockAccessToken), { wrapper });

    expect(result.current).toBeDefined();
  });

  it("should successfully create CloudZero integration with all parameters", async () => {
    const mockResponse = { message: "Integration created successfully", status: "success" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroCreate(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-connection-id",
      timezone: "America/New_York",
      api_key: "test-api-key",
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockResponse);
    expect(fetchSpy).toHaveBeenCalledWith(`${mockProxyBaseUrl}/cloudzero/init`, {
      method: "POST",
      headers: {
        [mockHeaderName]: `Bearer ${mockAccessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        connection_id: "test-connection-id",
        timezone: "America/New_York",
        api_key: "test-api-key",
      }),
    });
  });

  it("should successfully create CloudZero integration with minimal parameters", async () => {
    const mockResponse = { message: "Integration created successfully", status: "success" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroCreate(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-connection-id",
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockResponse);
    expect(fetchSpy).toHaveBeenCalledWith(`${mockProxyBaseUrl}/cloudzero/init`, {
      method: "POST",
      headers: {
        [mockHeaderName]: `Bearer ${mockAccessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        connection_id: "test-connection-id",
        timezone: "UTC",
      }),
    });
  });

  it("should use default timezone when not provided", async () => {
    const mockResponse = { message: "Integration created successfully" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroCreate(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-connection-id",
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const callBody = JSON.parse((fetchSpy as any).mock.calls[0][1].body);
    expect(callBody.timezone).toBe("UTC");
  });

  it("should not include api_key in body when not provided", async () => {
    const mockResponse = { message: "Integration created successfully" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroCreate(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-connection-id",
      timezone: "UTC",
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const callBody = JSON.parse((fetchSpy as any).mock.calls[0][1].body);
    expect(callBody).not.toHaveProperty("api_key");
  });

  it("should handle error response with error.message", async () => {
    const errorResponse = { error: { message: "Connection ID already exists" } };
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => errorResponse,
    });

    const { result } = renderHook(() => useCloudZeroCreate(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-connection-id",
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Connection ID already exists");
  });

  it("should handle error response with message field", async () => {
    const errorResponse = { message: "Invalid API key" };
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => errorResponse,
    });

    const { result } = renderHook(() => useCloudZeroCreate(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-connection-id",
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Invalid API key");
  });

  it("should handle error response with detail field", async () => {
    const errorResponse = { detail: "Server error occurred" };
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => errorResponse,
    });

    const { result } = renderHook(() => useCloudZeroCreate(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-connection-id",
    });

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

    const { result } = renderHook(() => useCloudZeroCreate(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-connection-id",
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Failed to create CloudZero integration");
  });

  it("should handle network error", async () => {
    const networkError = new Error("Network request failed");
    (fetchSpy as any).mockRejectedValue(networkError);

    const { result } = renderHook(() => useCloudZeroCreate(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-connection-id",
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(networkError);
  });

  it("should throw error when accessToken is empty string", async () => {
    const { result } = renderHook(() => useCloudZeroCreate(""), { wrapper });

    result.current.mutate({
      connection_id: "test-connection-id",
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Access token is required");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("should throw error when accessToken is null", async () => {
    const { result } = renderHook(() => useCloudZeroCreate(null as any), { wrapper });

    result.current.mutate({
      connection_id: "test-connection-id",
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Access token is required");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("should use relative URL when proxyBaseUrl is not set", async () => {
    mockGetProxyBaseUrl.mockReturnValue("");
    const mockResponse = { message: "Success" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroCreate(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-connection-id",
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(fetchSpy).toHaveBeenCalledWith("/cloudzero/init", expect.any(Object));
  });
});
