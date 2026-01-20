import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RouterFieldsResponse, useRouterFields } from "./useRouterFields";

// Mock the networking module
vi.mock("@/components/networking", () => ({
  proxyBaseUrl: null,
  getGlobalLitellmHeaderName: vi.fn(() => "Authorization"),
}));

// Mock useAuthorized hook
const mockUseAuthorized = vi.fn();
vi.mock("../useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

// Mock global fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock console methods to avoid noise in tests
vi.spyOn(console, "log").mockImplementation(() => {});
vi.spyOn(console, "error").mockImplementation(() => {});

// Mock data
const mockRouterFieldsResponse: RouterFieldsResponse = {
  fields: [
    {
      field_name: "routing_strategy",
      field_type: "String",
      field_description: "Routing strategy to use for load balancing across deployments",
      field_default: "simple-shuffle",
      options: ["simple-shuffle", "least-busy", "latency-based-routing"],
      ui_field_name: "Routing Strategy",
      link: null,
    },
    {
      field_name: "num_retries",
      field_type: "Integer",
      field_description: "Number of retries for failed requests",
      field_default: 0,
      options: null,
      ui_field_name: "Number of Retries",
      link: null,
    },
  ],
  routing_strategy_descriptions: {
    "simple-shuffle": "Randomly picks a deployment from the list. Simple and fast.",
    "least-busy": "Routes to the deployment with the lowest number of ongoing requests.",
    "latency-based-routing": "Routes to the deployment with the lowest latency over a sliding window.",
  },
};

describe("useRouterFields", () => {
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

    // Set default mock for useAuthorized (enabled state)
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userRole: "Admin",
      userId: "test-user-id",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should render", () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockRouterFieldsResponse,
    });

    const { result } = renderHook(() => useRouterFields(), { wrapper });

    expect(result.current).toBeDefined();
  });

  it("should return router fields data when query is successful", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockRouterFieldsResponse,
    });

    const { result } = renderHook(() => useRouterFields(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockRouterFieldsResponse);
    expect(result.current.error).toBeNull();
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith("/router/fields", {
      method: "GET",
      headers: {
        Authorization: "Bearer test-access-token",
        "Content-Type": "application/json",
      },
    });
  });

  it("should handle error when fetch fails", async () => {
    const errorMessage = "Failed to fetch router fields";
    const errorResponse = { error: errorMessage };

    mockFetch.mockResolvedValueOnce({
      ok: false,
      json: async () => errorResponse,
    });

    const { result } = renderHook(() => useRouterFields(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBeDefined();
    expect(result.current.data).toBeUndefined();
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("should not execute query when accessToken is missing", () => {
    mockUseAuthorized.mockReturnValue({
      accessToken: null,
      userRole: "Admin",
      userId: "test-user-id",
      token: null,
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useRouterFields(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("should not execute query when userId is missing", () => {
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userRole: "Admin",
      userId: null,
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useRouterFields(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("should not execute query when userRole is missing", () => {
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userRole: null,
      userId: "test-user-id",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useRouterFields(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("should handle network error", async () => {
    const networkError = new Error("Network error");
    mockFetch.mockRejectedValueOnce(networkError);

    const { result } = renderHook(() => useRouterFields(), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBeDefined();
    expect(result.current.data).toBeUndefined();
  });

  it("should use relative URL when proxyBaseUrl is null", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockRouterFieldsResponse,
    });

    const { result } = renderHook(() => useRouterFields(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // When proxyBaseUrl is null, should use relative URL
    expect(mockFetch).toHaveBeenCalledWith("/router/fields", {
      method: "GET",
      headers: {
        Authorization: "Bearer test-access-token",
        "Content-Type": "application/json",
      },
    });
  });

  it("should handle error response with different error formats", async () => {
    const errorFormats = [
      { error: { message: "Error message" } },
      { message: "Error message" },
      { detail: "Error detail" },
      { error: "Error string" },
      { unknown: "format" },
    ];

    for (const errorFormat of errorFormats) {
      vi.clearAllMocks();
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: async () => errorFormat,
      });

      const { result } = renderHook(() => useRouterFields(), { wrapper });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toBeDefined();
    }
  });

  it("should return empty fields array when API returns empty fields", async () => {
    const emptyResponse: RouterFieldsResponse = {
      fields: [],
      routing_strategy_descriptions: {},
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => emptyResponse,
    });

    const { result } = renderHook(() => useRouterFields(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data?.fields).toEqual([]);
    expect(result.current.data?.routing_strategy_descriptions).toEqual({});
  });

  it("should have correct query configuration", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockRouterFieldsResponse,
    });

    const { result } = renderHook(() => useRouterFields(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // Verify the query was called
    expect(mockFetch).toHaveBeenCalledTimes(1);

    // The hook should have the expected properties from useQuery
    expect(result.current).toHaveProperty("data");
    expect(result.current).toHaveProperty("isLoading");
    expect(result.current).toHaveProperty("isError");
    expect(result.current).toHaveProperty("isSuccess");
    expect(result.current).toHaveProperty("error");
  });

  it("should handle fields with null options", async () => {
    const responseWithNullOptions: RouterFieldsResponse = {
      fields: [
        {
          field_name: "timeout",
          field_type: "Float",
          field_description: "Timeout for requests in seconds",
          field_default: null,
          options: null,
          ui_field_name: "Timeout",
          link: null,
        },
      ],
      routing_strategy_descriptions: {},
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => responseWithNullOptions,
    });

    const { result } = renderHook(() => useRouterFields(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data?.fields[0].options).toBeNull();
  });

  it("should handle fields with link property", async () => {
    const responseWithLink: RouterFieldsResponse = {
      fields: [
        {
          field_name: "enable_tag_filtering",
          field_type: "Boolean",
          field_description: "Enable tag-based routing",
          field_default: false,
          options: null,
          ui_field_name: "Enable Tag Filtering",
          link: "https://docs.litellm.ai/docs/proxy/tag_routing",
        },
      ],
      routing_strategy_descriptions: {},
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => responseWithLink,
    });

    const { result } = renderHook(() => useRouterFields(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data?.fields[0].link).toBe("https://docs.litellm.ai/docs/proxy/tag_routing");
  });
});
