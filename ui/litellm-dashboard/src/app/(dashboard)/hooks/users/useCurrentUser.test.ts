import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useCurrentUser } from "./useCurrentUser";
import { userInfoCall } from "@/components/networking";
import type { UserInfo } from "@/components/view_users/types";

// Mock the networking function
vi.mock("@/components/networking", () => ({
  userInfoCall: vi.fn(),
}));

// Mock the queryKeysFactory - we'll mock the specific return value
vi.mock("../common/queryKeysFactory", () => ({
  createQueryKeys: vi.fn((resource: string) => ({
    all: [resource],
    lists: () => [resource, "list"],
    list: (params?: any) => [resource, "list", { params }],
    details: () => [resource, "detail"],
    detail: (uid: string) => [resource, "detail", uid],
  })),
}));

// Mock useAuthorized hook - we can override this in individual tests
const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

// Mock data - response from userInfoCall should have user_info property
const mockUserInfoResponse = {
  user_info: {
    user_id: "test-user-id",
    user_email: "test@example.com",
    user_alias: "Test User",
    user_role: "Admin",
    spend: 150.75,
    max_budget: 1000.0,
    key_count: 5,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    sso_user_id: null,
    budget_duration: "monthly",
  } as UserInfo,
};

describe("useCurrentUser", () => {
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

  it("should return user info data when query is successful", async () => {
    // Mock successful API call
    (userInfoCall as any).mockResolvedValue(mockUserInfoResponse);

    const { result } = renderHook(() => useCurrentUser(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockUserInfoResponse.user_info);
    expect(result.current.error).toBeNull();
    expect(userInfoCall).toHaveBeenCalledWith("test-access-token", "test-user-id", "Admin", false, null, null);
    expect(userInfoCall).toHaveBeenCalledTimes(1);
  });

  it("should handle error when userInfoCall fails", async () => {
    const errorMessage = "Failed to fetch user info";
    const testError = new Error(errorMessage);

    // Mock failed API call
    (userInfoCall as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useCurrentUser(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(userInfoCall).toHaveBeenCalledWith("test-access-token", "test-user-id", "Admin", false, null, null);
    expect(userInfoCall).toHaveBeenCalledTimes(1);
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

    const { result } = renderHook(() => useCurrentUser(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(userInfoCall).not.toHaveBeenCalled();
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

    const { result } = renderHook(() => useCurrentUser(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(userInfoCall).not.toHaveBeenCalled();
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

    const { result } = renderHook(() => useCurrentUser(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(userInfoCall).not.toHaveBeenCalled();
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

    const { result } = renderHook(() => useCurrentUser(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(userInfoCall).not.toHaveBeenCalled();
  });

  it("should execute query when all auth values are present", async () => {
    // Mock successful API call
    (userInfoCall as any).mockResolvedValue(mockUserInfoResponse);

    // Ensure all auth values are present (already set in beforeEach)
    const { result } = renderHook(() => useCurrentUser(), { wrapper });

    // Wait for query to execute
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(userInfoCall).toHaveBeenCalledWith("test-access-token", "test-user-id", "Admin", false, null, null);
    expect(userInfoCall).toHaveBeenCalledTimes(1);
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    // Mock network timeout
    (userInfoCall as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useCurrentUser(), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });
});
