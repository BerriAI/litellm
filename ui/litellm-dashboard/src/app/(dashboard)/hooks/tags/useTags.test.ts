import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useTags } from "./useTags";
import { tagListCall } from "@/components/networking";
import type { TagListResponse } from "@/components/tag_management/types";

// Mock the networking function
vi.mock("@/components/networking", () => ({
  tagListCall: vi.fn(),
}));

// Mock useAuthorized hook - we can override this in individual tests
const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

// Mock data
const mockTags: TagListResponse = {
  "tag-1": {
    name: "tag-1",
    description: "Test tag 1 description",
    models: ["gpt-3.5-turbo", "gpt-4"],
    model_info: { "gpt-3.5-turbo": "GPT-3.5 Turbo", "gpt-4": "GPT-4" },
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    created_by: "user-1",
    updated_by: "user-1",
    litellm_budget_table: {
      max_budget: 1000,
      soft_budget: 800,
      tpm_limit: 100000,
      rpm_limit: 1000,
      max_parallel_requests: 10,
      budget_duration: "monthly",
      model_max_budget: { "gpt-3.5-turbo": 500, "gpt-4": 500 },
    },
  },
  "tag-2": {
    name: "tag-2",
    description: "Test tag 2 description",
    models: ["claude-3"],
    model_info: { "claude-3": "Claude 3" },
    created_at: "2024-01-02T00:00:00Z",
    updated_at: "2024-01-02T00:00:00Z",
    created_by: "user-2",
    updated_by: "user-2",
    litellm_budget_table: {
      max_budget: 2000,
      soft_budget: 1500,
      tpm_limit: 200000,
      rpm_limit: 2000,
      max_parallel_requests: 20,
      budget_duration: "monthly",
      model_max_budget: { "claude-3": 2000 },
    },
  },
};

describe("useTags", () => {
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
      userId: "test-user-id",
      userRole: "Admin",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should return tags data when query is successful", async () => {
    // Mock successful API call
    (tagListCall as any).mockResolvedValue(mockTags);

    const { result } = renderHook(() => useTags(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockTags);
    expect(result.current.error).toBeNull();
    expect(tagListCall).toHaveBeenCalledWith("test-access-token");
    expect(tagListCall).toHaveBeenCalledTimes(1);
  });

  it("should handle error when tagListCall fails", async () => {
    const errorMessage = "Failed to fetch tags";
    const testError = new Error(errorMessage);

    // Mock failed API call
    (tagListCall as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useTags(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(tagListCall).toHaveBeenCalledWith("test-access-token");
    expect(tagListCall).toHaveBeenCalledTimes(1);
  });

  it("should not execute query when accessToken is missing", async () => {
    // Mock missing accessToken
    mockUseAuthorized.mockReturnValue({
      accessToken: null,
      userId: "test-user-id",
      userRole: "Admin",
      token: null,
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useTags(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(tagListCall).not.toHaveBeenCalled();
  });

  it("should not execute query when userId is missing", async () => {
    // Mock missing userId
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userId: null,
      userRole: "Admin",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useTags(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(tagListCall).not.toHaveBeenCalled();
  });

  it("should not execute query when userRole is missing", async () => {
    // Mock missing userRole
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userId: "test-user-id",
      userRole: null,
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useTags(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(tagListCall).not.toHaveBeenCalled();
  });

  it("should not execute query when all auth values are missing", async () => {
    // Mock all auth values missing
    mockUseAuthorized.mockReturnValue({
      accessToken: null,
      userId: null,
      userRole: null,
      token: null,
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useTags(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(tagListCall).not.toHaveBeenCalled();
  });

  it("should execute query when all auth values are present", async () => {
    // Mock successful API call
    (tagListCall as any).mockResolvedValue(mockTags);

    // Ensure all auth values are present (already set in beforeEach)
    const { result } = renderHook(() => useTags(), { wrapper });

    // Wait for query to execute
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(tagListCall).toHaveBeenCalledWith("test-access-token");
    expect(tagListCall).toHaveBeenCalledTimes(1);
  });

  it("should return empty object when API returns empty data", async () => {
    // Mock API returning empty object
    (tagListCall as any).mockResolvedValue({});

    const { result } = renderHook(() => useTags(), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual({});
    expect(tagListCall).toHaveBeenCalledWith("test-access-token");
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    // Mock network timeout
    (tagListCall as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useTags(), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });
});
