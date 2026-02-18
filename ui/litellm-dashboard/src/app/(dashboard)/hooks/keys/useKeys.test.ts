import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useKeys, useDeletedKeys } from "./useKeys";
import type { KeyResponse } from "@/components/key_team_helpers/key_list";

// Mock the networking utilities
vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn().mockReturnValue(""),
  getGlobalLitellmHeaderName: vi.fn().mockReturnValue("Authorization"),
  deriveErrorMessage: vi.fn((errorData: any) => {
    return (
      (errorData?.error && (errorData.error.message || errorData.error)) ||
      errorData?.message ||
      errorData?.detail ||
      errorData?.error ||
      JSON.stringify(errorData)
    );
  }),
  handleError: vi.fn(),
}));

// Mock global fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock console methods to avoid noise in tests
vi.spyOn(console, "log").mockImplementation(() => {});
vi.spyOn(console, "error").mockImplementation(() => {});

// Mock useAuthorized hook - we can override this in individual tests
const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

// Mock data
const mockKeys: KeyResponse[] = [
  {
    token: "sk-test-key-1",
    token_id: "key-1",
    key_name: "Test Key 1",
    key_alias: "test-key-1",
    spend: 10.5,
    max_budget: 100,
    expires: "2024-12-31T23:59:59Z",
    models: ["gpt-3.5-turbo"],
    aliases: {},
    config: {},
    user_id: "user-1",
    team_id: null,
    max_parallel_requests: 10,
    metadata: {},
    tpm_limit: 1000,
    rpm_limit: 100,
    duration: "30d",
    budget_duration: "1mo",
    budget_reset_at: "2024-02-01T00:00:00Z",
    allowed_cache_controls: [],
    allowed_routes: [],
    permissions: {},
    model_spend: { "gpt-3.5-turbo": 10.5 },
    model_max_budget: { "gpt-3.5-turbo": 100 },
    soft_budget_cooldown: false,
    blocked: false,
    litellm_budget_table: {},
    organization_id: null,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    team_spend: 0,
    team_alias: "",
    team_tpm_limit: 0,
    team_rpm_limit: 0,
    team_max_budget: 0,
    team_models: [],
    team_blocked: false,
    soft_budget: 0,
    team_model_aliases: {},
    team_member_spend: 0,
    team_metadata: {},
    end_user_id: "",
    end_user_tpm_limit: 0,
    end_user_rpm_limit: 0,
    end_user_max_budget: 0,
    last_refreshed_at: 0,
    api_key: "",
    user_role: "user",
    rpm_limit_per_model: {},
    tpm_limit_per_model: {},
    user_tpm_limit: 0,
    user_rpm_limit: 0,
    user_email: "",
  },
  {
    token: "sk-test-key-2",
    token_id: "key-2",
    key_name: "Test Key 2",
    key_alias: "test-key-2",
    spend: 25.0,
    max_budget: 200,
    expires: "2024-12-31T23:59:59Z",
    models: ["claude-3"],
    aliases: {},
    config: {},
    user_id: "user-2",
    team_id: "team-1",
    max_parallel_requests: 5,
    metadata: {},
    tpm_limit: 500,
    rpm_limit: 50,
    duration: "30d",
    budget_duration: "1mo",
    budget_reset_at: "2024-02-01T00:00:00Z",
    allowed_cache_controls: [],
    allowed_routes: [],
    permissions: {},
    model_spend: { "claude-3": 25.0 },
    model_max_budget: { "claude-3": 200 },
    soft_budget_cooldown: false,
    blocked: false,
    litellm_budget_table: {},
    organization_id: null,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    team_spend: 0,
    team_alias: "test-team",
    team_tpm_limit: 1000,
    team_rpm_limit: 100,
    team_max_budget: 500,
    team_models: ["claude-3"],
    team_blocked: false,
    soft_budget: 0,
    team_model_aliases: {},
    team_member_spend: 0,
    team_metadata: {},
    end_user_id: "",
    end_user_tpm_limit: 0,
    end_user_rpm_limit: 0,
    end_user_max_budget: 0,
    last_refreshed_at: 0,
    api_key: "",
    user_role: "user",
    rpm_limit_per_model: {},
    tpm_limit_per_model: {},
    user_tpm_limit: 0,
    user_rpm_limit: 0,
    user_email: "",
  },
];

const mockKeysResponse = {
  keys: mockKeys,
  total_count: 2,
  current_page: 1,
  total_pages: 1,
};

describe("useKeys", () => {
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

    // Reset fetch mock
    mockFetch.mockClear();
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should return keys data when query is successful", async () => {
    // Mock successful API call
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockKeysResponse,
    });

    const { result } = renderHook(() => useKeys(1, 10), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockKeysResponse);
    expect(result.current.error).toBeNull();
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      "/key/list?page=1&size=10&return_full_object=true&include_team_keys=true&include_created_by_keys=true",
      {
        method: "GET",
        headers: {
          Authorization: "Bearer test-access-token",
          "Content-Type": "application/json",
        },
      },
    );
  });

  it("should handle error when keyListCall fails", async () => {
    const errorMessage = "Failed to fetch keys";
    const errorResponse = { error: errorMessage };

    // Mock failed API call
    mockFetch.mockResolvedValueOnce({
      ok: false,
      json: async () => errorResponse,
    });

    const { result } = renderHook(() => useKeys(1, 10), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBeDefined();
    expect(result.current.error?.message).toBe(errorMessage);
    expect(result.current.data).toBeUndefined();
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      "/key/list?page=1&size=10&return_full_object=true&include_team_keys=true&include_created_by_keys=true",
      {
        method: "GET",
        headers: {
          Authorization: "Bearer test-access-token",
          "Content-Type": "application/json",
        },
      },
    );
  });

  it("should not execute query when accessToken is missing", async () => {
    // Mock missing accessToken
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

    const { result } = renderHook(() => useKeys(1, 10), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("should pass correct page and pageSize parameters to the API", async () => {
    // Mock successful API call
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockKeysResponse,
    });

    const page = 2;
    const pageSize = 20;

    const { result } = renderHook(() => useKeys(page, pageSize), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(mockFetch).toHaveBeenCalledWith(
      `/key/list?page=${page}&size=${pageSize}&return_full_object=true&include_team_keys=true&include_created_by_keys=true`,
      {
        method: "GET",
        headers: {
          Authorization: "Bearer test-access-token",
          "Content-Type": "application/json",
        },
      },
    );
  });

  it("should return empty keys array when API returns empty data", async () => {
    // Mock API returning empty keys array
    const emptyResponse = {
      keys: [],
      total_count: 0,
      current_page: 1,
      total_pages: 0,
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => emptyResponse,
    });

    const { result } = renderHook(() => useKeys(1, 10), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(emptyResponse);
    expect(mockFetch).toHaveBeenCalledWith(
      "/key/list?page=1&size=10&return_full_object=true&include_team_keys=true&include_created_by_keys=true",
      {
        method: "GET",
        headers: {
          Authorization: "Bearer test-access-token",
          "Content-Type": "application/json",
        },
      },
    );
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    // Mock network timeout
    mockFetch.mockRejectedValueOnce(timeoutError);

    const { result } = renderHook(() => useKeys(1, 10), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });

  it("should handle pagination correctly", async () => {
    const paginatedResponse = {
      keys: [mockKeys[0]], // Only first key
      total_count: 15,
      current_page: 2,
      total_pages: 2,
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => paginatedResponse,
    });

    const { result } = renderHook(() => useKeys(2, 10), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual(paginatedResponse);
    expect(mockFetch).toHaveBeenCalledWith(
      "/key/list?page=2&size=10&return_full_object=true&include_team_keys=true&include_created_by_keys=true",
      {
        method: "GET",
        headers: {
          Authorization: "Bearer test-access-token",
          "Content-Type": "application/json",
        },
      },
    );
  });
});

describe("useDeletedKeys", () => {
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

    // Reset fetch mock
    mockFetch.mockClear();
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should return deleted keys data when query is successful", async () => {
    // Mock successful API call
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockKeysResponse,
    });

    const { result } = renderHook(() => useDeletedKeys(1, 10), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockKeysResponse);
    expect(result.current.error).toBeNull();
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      "/key/list?page=1&size=10&status=deleted&return_full_object=true&include_team_keys=true&include_created_by_keys=true",
      {
        method: "GET",
        headers: {
          Authorization: "Bearer test-access-token",
          "Content-Type": "application/json",
        },
      },
    );
  });

  it("should pass status=deleted parameter to the API", async () => {
    // Mock successful API call
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockKeysResponse,
    });

    const { result } = renderHook(() => useDeletedKeys(1, 10), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Verify that status=deleted is included in the URL
    const callUrl = mockFetch.mock.calls[0][0];
    expect(callUrl).toContain("status=deleted");
    expect(result.current.data).toEqual(mockKeysResponse);
  });

  it("should handle error when deleted keys API call fails", async () => {
    const errorMessage = "Failed to fetch deleted keys";
    const errorResponse = { error: errorMessage };

    // Mock failed API call
    mockFetch.mockResolvedValueOnce({
      ok: false,
      json: async () => errorResponse,
    });

    const { result } = renderHook(() => useDeletedKeys(1, 10), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBeDefined();
    expect(result.current.error?.message).toBe(errorMessage);
    expect(result.current.data).toBeUndefined();
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      "/key/list?page=1&size=10&status=deleted&return_full_object=true&include_team_keys=true&include_created_by_keys=true",
      {
        method: "GET",
        headers: {
          Authorization: "Bearer test-access-token",
          "Content-Type": "application/json",
        },
      },
    );
  });

  it("should not execute query when accessToken is missing", async () => {
    // Mock missing accessToken
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

    const { result } = renderHook(() => useDeletedKeys(1, 10), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("should pass correct page and pageSize parameters to the API", async () => {
    // Mock successful API call
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockKeysResponse,
    });

    const page = 2;
    const pageSize = 20;

    const { result } = renderHook(() => useDeletedKeys(page, pageSize), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(mockFetch).toHaveBeenCalledWith(
      `/key/list?page=${page}&size=${pageSize}&status=deleted&return_full_object=true&include_team_keys=true&include_created_by_keys=true`,
      {
        method: "GET",
        headers: {
          Authorization: "Bearer test-access-token",
          "Content-Type": "application/json",
        },
      },
    );
  });

  it("should return empty deleted keys array when API returns empty data", async () => {
    // Mock API returning empty keys array
    const emptyResponse = {
      keys: [],
      total_count: 0,
      current_page: 1,
      total_pages: 0,
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => emptyResponse,
    });

    const { result } = renderHook(() => useDeletedKeys(1, 10), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(emptyResponse);
    expect(mockFetch).toHaveBeenCalledWith(
      "/key/list?page=1&size=10&status=deleted&return_full_object=true&include_team_keys=true&include_created_by_keys=true",
      {
        method: "GET",
        headers: {
          Authorization: "Bearer test-access-token",
          "Content-Type": "application/json",
        },
      },
    );
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    // Mock network timeout
    mockFetch.mockRejectedValueOnce(timeoutError);

    const { result } = renderHook(() => useDeletedKeys(1, 10), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });

  it("should handle pagination correctly", async () => {
    const paginatedResponse = {
      keys: [mockKeys[0]], // Only first key
      total_count: 15,
      current_page: 2,
      total_pages: 2,
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => paginatedResponse,
    });

    const { result } = renderHook(() => useDeletedKeys(2, 10), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual(paginatedResponse);
    expect(mockFetch).toHaveBeenCalledWith(
      "/key/list?page=2&size=10&status=deleted&return_full_object=true&include_team_keys=true&include_created_by_keys=true",
      {
        method: "GET",
        headers: {
          Authorization: "Bearer test-access-token",
          "Content-Type": "application/json",
        },
      },
    );
  });

  it("should pass additional options along with status=deleted", async () => {
    // Mock successful API call
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockKeysResponse,
    });

    const options = {
      organizationID: "org-1",
      teamID: "team-1",
      selectedKeyAlias: "test-alias",
    };

    const { result } = renderHook(() => useDeletedKeys(1, 10, options), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const callUrl = mockFetch.mock.calls[0][0];
    expect(callUrl).toContain("status=deleted");
    expect(callUrl).toContain("organization_id=org-1");
    expect(callUrl).toContain("team_id=team-1");
    expect(callUrl).toContain("key_alias=test-alias");
    expect(result.current.data).toEqual(mockKeysResponse);
  });
});
