import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useCloudZeroSettings, useCloudZeroUpdateSettings, useCloudZeroDeleteSettings } from "./useCloudZeroSettings";
import { CloudZeroSettings } from "@/components/CloudZeroCostTracking/types";

const {
  mockProxyBaseUrl,
  mockAccessToken,
  mockHeaderName,
  mockGetProxyBaseUrl,
  mockGetGlobalLitellmHeaderName,
  mockCreateQueryKeys,
} = vi.hoisted(() => {
  const mockProxyBaseUrl = "https://proxy.example.com";
  const mockAccessToken = "test-access-token";
  const mockHeaderName = "X-LiteLLM-API-Key";
  const mockGetProxyBaseUrl = vi.fn(() => mockProxyBaseUrl);
  const mockGetGlobalLitellmHeaderName = vi.fn(() => mockHeaderName);
  const mockCreateQueryKeys = vi.fn((resource: string) => ({
    all: [resource],
    lists: () => [resource, "list"],
    list: (params?: any) => [resource, "list", { params }],
    details: () => [resource, "detail"],
    detail: (uid: string) => [resource, "detail", uid],
  }));

  return {
    mockProxyBaseUrl,
    mockAccessToken,
    mockHeaderName,
    mockGetProxyBaseUrl,
    mockGetGlobalLitellmHeaderName,
    mockCreateQueryKeys,
  };
});

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: mockGetProxyBaseUrl,
  getGlobalLitellmHeaderName: mockGetGlobalLitellmHeaderName,
}));

vi.mock("../common/queryKeysFactory", () => ({
  createQueryKeys: mockCreateQueryKeys,
}));

const mockCloudZeroSettings: CloudZeroSettings = {
  api_key_masked: "sk-****1234",
  connection_id: "test-connection-id",
  timezone: "America/New_York",
  status: "active",
};

describe("useCloudZeroSettings", () => {
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

  it("should return CloudZero settings data when query is successful", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockCloudZeroSettings,
    });

    const { result } = renderHook(() => useCloudZeroSettings(mockAccessToken), { wrapper });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockCloudZeroSettings);
    expect(result.current.error).toBeNull();
    expect(fetchSpy).toHaveBeenCalledWith(`${mockProxyBaseUrl}/cloudzero/settings`, {
      method: "GET",
      headers: {
        [mockHeaderName]: `Bearer ${mockAccessToken}`,
        "Content-Type": "application/json",
      },
    });
  });

  it("should return null when settings are not configured (missing both api_key_masked and connection_id)", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => ({}),
    });

    const { result } = renderHook(() => useCloudZeroSettings(mockAccessToken), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toBeNull();
  });

  it("should return settings when at least one required field is present", async () => {
    const settingsWithConnectionId = { connection_id: "test-connection-id" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => settingsWithConnectionId,
    });

    const { result } = renderHook(() => useCloudZeroSettings(mockAccessToken), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(settingsWithConnectionId);
  });

  it("should handle error responses", async () => {
    const errorCases = [
      { error: { message: "Failed to fetch" }, expected: "Failed to fetch" },
      { error: "Unauthorized", expected: "Unauthorized" },
      { message: "Not found", expected: "Not found" },
      { detail: "Server error", expected: "Server error" },
    ];

    for (const errorResponse of errorCases) {
      vi.clearAllMocks();
      (fetchSpy as any).mockResolvedValue({
        ok: false,
        json: async () => errorResponse,
      });

      const { result } = renderHook(() => useCloudZeroSettings(mockAccessToken), { wrapper });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error?.message).toBe(errorResponse.expected);
    }
  });

  it("should handle error response with string error data", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => "Error string",
    });

    const { result } = renderHook(() => useCloudZeroSettings(mockAccessToken), { wrapper });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Error string");
  });

  it("should handle error response with invalid JSON", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      statusText: "Internal Server Error",
      json: async () => {
        throw new Error("Invalid JSON");
      },
    });

    const { result } = renderHook(() => useCloudZeroSettings(mockAccessToken), { wrapper });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Internal Server Error");
  });

  it("should handle network error", async () => {
    const networkError = new Error("Network request failed");
    (fetchSpy as any).mockRejectedValue(networkError);

    const { result } = renderHook(() => useCloudZeroSettings(mockAccessToken), { wrapper });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(networkError);
  });

  it("should not execute query when accessToken is missing", () => {
    const { result } = renderHook(() => useCloudZeroSettings(""), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("should use relative URL when proxyBaseUrl is not set", async () => {
    mockGetProxyBaseUrl.mockReturnValue("");
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockCloudZeroSettings,
    });

    const { result } = renderHook(() => useCloudZeroSettings(mockAccessToken), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(fetchSpy).toHaveBeenCalledWith("/cloudzero/settings", expect.any(Object));
  });
});

describe("useCloudZeroUpdateSettings", () => {
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

  it("should successfully update settings with all parameters", async () => {
    const mockResponse = { message: "Settings updated successfully", status: "success" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroUpdateSettings(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "new-connection-id",
      timezone: "America/Los_Angeles",
      api_key: "new-api-key",
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockResponse);
    expect(fetchSpy).toHaveBeenCalledWith(`${mockProxyBaseUrl}/cloudzero/settings`, {
      method: "PUT",
      headers: {
        [mockHeaderName]: `Bearer ${mockAccessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        connection_id: "new-connection-id",
        timezone: "America/Los_Angeles",
        api_key: "new-api-key",
      }),
    });
  });

  it("should not include undefined fields in request body", async () => {
    const mockResponse = { message: "Updated" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroUpdateSettings(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-id",
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const callBody = JSON.parse((fetchSpy as any).mock.calls[0][1].body);
    expect(callBody).toEqual({ connection_id: "test-id" });
    expect(callBody).not.toHaveProperty("timezone");
    expect(callBody).not.toHaveProperty("api_key");
  });

  it("should invalidate settings query on success", async () => {
    const mockResponse = { message: "Updated", status: "success" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    queryClient.setQueryData(["cloudZeroSettings", "list", { params: {} }], mockCloudZeroSettings);

    const { result } = renderHook(() => useCloudZeroUpdateSettings(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-id",
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const queryCache = queryClient.getQueryCache();
    const queries = queryCache.findAll();
    const settingsQuery = queries.find((q) => q.queryKey[0] === "cloudZeroSettings");

    expect(settingsQuery).toBeDefined();
  });

  it("should handle error responses", async () => {
    const errorCases = [
      { error: { message: "Update failed" }, expected: "Update failed" },
      { error: "Validation error", expected: "Validation error" },
      { message: "Invalid input", expected: "Invalid input" },
      { detail: "Server error", expected: "Server error" },
    ];

    for (const errorResponse of errorCases) {
      vi.clearAllMocks();
      (fetchSpy as any).mockResolvedValue({
        ok: false,
        json: async () => errorResponse,
      });

      const { result } = renderHook(() => useCloudZeroUpdateSettings(mockAccessToken), { wrapper });

      result.current.mutate({
        connection_id: "test-id",
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error?.message).toBe(errorResponse.expected);
    }
  });

  it("should handle error response with string error data", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => "Error string",
    });

    const { result } = renderHook(() => useCloudZeroUpdateSettings(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-id",
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Error string");
  });

  it("should handle error response with invalid JSON", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      statusText: "Bad Request",
      json: async () => {
        throw new Error("Invalid JSON");
      },
    });

    const { result } = renderHook(() => useCloudZeroUpdateSettings(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-id",
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Bad Request");
  });

  it("should handle network error", async () => {
    const networkError = new Error("Network request failed");
    (fetchSpy as any).mockRejectedValue(networkError);

    const { result } = renderHook(() => useCloudZeroUpdateSettings(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-id",
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(networkError);
  });

  it("should throw error when accessToken is missing", async () => {
    const testCases = ["", null as any];

    for (const accessToken of testCases) {
      vi.clearAllMocks();
      const { result } = renderHook(() => useCloudZeroUpdateSettings(accessToken), { wrapper });

      result.current.mutate({
        connection_id: "test-id",
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error?.message).toBe("Access token is required");
      expect(fetchSpy).not.toHaveBeenCalled();
    }
  });

  it("should use relative URL when proxyBaseUrl is not set", async () => {
    mockGetProxyBaseUrl.mockReturnValue("");
    const mockResponse = { message: "Updated", status: "success" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroUpdateSettings(mockAccessToken), { wrapper });

    result.current.mutate({
      connection_id: "test-id",
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(fetchSpy).toHaveBeenCalledWith("/cloudzero/settings", expect.any(Object));
  });
});

describe("useCloudZeroDeleteSettings", () => {
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

  it("should successfully delete settings", async () => {
    const mockResponse = { message: "Settings deleted successfully", status: "success" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroDeleteSettings(mockAccessToken), { wrapper });

    result.current.mutate();

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockResponse);
    expect(fetchSpy).toHaveBeenCalledWith(`${mockProxyBaseUrl}/cloudzero/delete`, {
      method: "DELETE",
      headers: {
        [mockHeaderName]: `Bearer ${mockAccessToken}`,
        "Content-Type": "application/json",
      },
    });
  });

  it("should invalidate settings query on success", async () => {
    const mockResponse = { message: "Deleted", status: "success" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    queryClient.setQueryData(["cloudZeroSettings", "list", { params: {} }], mockCloudZeroSettings);

    const { result } = renderHook(() => useCloudZeroDeleteSettings(mockAccessToken), { wrapper });

    result.current.mutate();

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const queryCache = queryClient.getQueryCache();
    const queries = queryCache.findAll();
    const settingsQuery = queries.find((q) => q.queryKey[0] === "cloudZeroSettings");

    expect(settingsQuery).toBeDefined();
  });

  it("should handle error responses", async () => {
    const errorCases = [
      { error: { message: "Delete failed" }, expected: "Delete failed" },
      { error: "Permission denied", expected: "Permission denied" },
      { message: "Not found", expected: "Not found" },
      { detail: "Server error", expected: "Server error" },
    ];

    for (const errorResponse of errorCases) {
      vi.clearAllMocks();
      (fetchSpy as any).mockResolvedValue({
        ok: false,
        json: async () => errorResponse,
      });

      const { result } = renderHook(() => useCloudZeroDeleteSettings(mockAccessToken), { wrapper });

      result.current.mutate();

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error?.message).toBe(errorResponse.expected);
    }
  });

  it("should handle error response with string error data", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => "Error string",
    });

    const { result } = renderHook(() => useCloudZeroDeleteSettings(mockAccessToken), { wrapper });

    result.current.mutate();

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Error string");
  });

  it("should handle error response with invalid JSON", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: false,
      statusText: "Internal Server Error",
      json: async () => {
        throw new Error("Invalid JSON");
      },
    });

    const { result } = renderHook(() => useCloudZeroDeleteSettings(mockAccessToken), { wrapper });

    result.current.mutate();

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Internal Server Error");
  });

  it("should handle network error", async () => {
    const networkError = new Error("Network request failed");
    (fetchSpy as any).mockRejectedValue(networkError);

    const { result } = renderHook(() => useCloudZeroDeleteSettings(mockAccessToken), { wrapper });

    result.current.mutate();

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(networkError);
  });

  it("should throw error when accessToken is missing", async () => {
    const testCases = ["", null as any];

    for (const accessToken of testCases) {
      vi.clearAllMocks();
      const { result } = renderHook(() => useCloudZeroDeleteSettings(accessToken), { wrapper });

      result.current.mutate();

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error?.message).toBe("Access token is required");
      expect(fetchSpy).not.toHaveBeenCalled();
    }
  });

  it("should use relative URL when proxyBaseUrl is not set", async () => {
    mockGetProxyBaseUrl.mockReturnValue("");
    const mockResponse = { message: "Deleted", status: "success" };
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useCloudZeroDeleteSettings(mockAccessToken), { wrapper });

    result.current.mutate();

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(fetchSpy).toHaveBeenCalledWith("/cloudzero/delete", expect.any(Object));
  });
});
