import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useInfiniteUsers } from "./useUsers";
import { userListCall } from "@/components/networking";
import type { UserListResponse } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  userListCall: vi.fn(),
}));

vi.mock("../common/queryKeysFactory", () => ({
  createQueryKeys: vi.fn((resource: string) => ({
    all: [resource],
    lists: () => [resource, "list"],
    list: (params?: any) => [resource, "list", { params }],
    details: () => [resource, "detail"],
    detail: (uid: string) => [resource, "detail", uid],
  })),
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const DEFAULT_AUTH = {
  accessToken: "test-access-token",
  userId: "test-user-id",
  userRole: "Admin",
  token: "test-token",
  userEmail: "test@example.com",
  premiumUser: false,
  disabledPersonalKeyCreation: null,
  showSSOBanner: false,
};

const buildUserListResponse = (
  page: number,
  totalPages: number,
  userCount = 2,
): UserListResponse => ({
  page,
  page_size: 50,
  total: totalPages * userCount,
  total_pages: totalPages,
  users: Array.from({ length: userCount }, (_, i) => ({
    user_id: `user-${page}-${i}`,
    user_email: `user-${page}-${i}@example.com`,
    user_alias: null,
    user_role: "Internal User",
    spend: 0,
    max_budget: null,
    key_count: 0,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    sso_user_id: null,
    budget_duration: null,
  })),
});

describe("useInfiniteUsers", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue(DEFAULT_AUTH);
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should return paginated user data when query is successful", async () => {
    const mockResponse = buildUserListResponse(1, 2);
    (userListCall as any).mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data?.pages).toHaveLength(1);
    expect(result.current.data?.pages[0]).toEqual(mockResponse);
    expect(userListCall).toHaveBeenCalledWith(
      "test-access-token",
      null,
      1,
      50,
      null,
    );
  });

  it("should use the default page size of 50", async () => {
    const mockResponse = buildUserListResponse(1, 1);
    (userListCall as any).mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(userListCall).toHaveBeenCalledWith(
      "test-access-token",
      null,
      1,
      50,
      null,
    );
  });

  it("should use a custom page size when provided", async () => {
    const customPageSize = 25;
    const mockResponse = buildUserListResponse(1, 1, 5);
    (userListCall as any).mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useInfiniteUsers(customPageSize), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(userListCall).toHaveBeenCalledWith(
      "test-access-token",
      null,
      1,
      customPageSize,
      null,
    );
  });

  it("should pass searchEmail to userListCall when provided", async () => {
    const searchEmail = "search@example.com";
    const mockResponse = buildUserListResponse(1, 1, 1);
    (userListCall as any).mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useInfiniteUsers(50, searchEmail), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(userListCall).toHaveBeenCalledWith(
      "test-access-token",
      null,
      1,
      50,
      searchEmail,
    );
  });

  it("should pass null for searchEmail when not provided", async () => {
    const mockResponse = buildUserListResponse(1, 1);
    (userListCall as any).mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useInfiniteUsers(50, undefined), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(userListCall).toHaveBeenCalledWith(
      "test-access-token",
      null,
      1,
      50,
      null,
    );
  });

  it("should fetch the next page when more pages are available", async () => {
    const page1 = buildUserListResponse(1, 3);
    const page2 = buildUserListResponse(2, 3);
    let callCount = 0;
    (userListCall as any).mockImplementation(async () => {
      callCount++;
      return callCount === 1 ? page1 : page2;
    });

    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.hasNextPage).toBe(true);

    result.current.fetchNextPage();

    await waitFor(() => {
      expect(result.current.isFetchingNextPage).toBe(false);
      expect(result.current.data?.pages).toHaveLength(2);
    });

    expect(result.current.data?.pages[1]).toEqual(page2);
    expect(userListCall).toHaveBeenCalledTimes(2);
    expect(userListCall).toHaveBeenLastCalledWith(
      "test-access-token",
      null,
      2,
      50,
      null,
    );
  });

  it("should not have a next page when on the last page", async () => {
    const lastPage = buildUserListResponse(2, 2);
    (userListCall as any).mockResolvedValue(lastPage);

    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.hasNextPage).toBe(false);
  });

  it("should not execute query when accessToken is missing", async () => {
    mockUseAuthorized.mockReturnValue({
      ...DEFAULT_AUTH,
      accessToken: null,
    });

    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(userListCall).not.toHaveBeenCalled();
  });

  it("should not execute query when userRole is not an admin role", async () => {
    mockUseAuthorized.mockReturnValue({
      ...DEFAULT_AUTH,
      userRole: "Internal User",
    });

    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(userListCall).not.toHaveBeenCalled();
  });

  it("should not execute query when both accessToken and userRole are invalid", async () => {
    mockUseAuthorized.mockReturnValue({
      ...DEFAULT_AUTH,
      accessToken: null,
      userRole: "App User",
    });

    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(userListCall).not.toHaveBeenCalled();
  });

  it("should execute query for each admin role", async () => {
    const adminRoles = [
      "Admin",
      "Admin Viewer",
      "proxy_admin",
      "proxy_admin_viewer",
      "org_admin",
    ];

    for (const role of adminRoles) {
      vi.clearAllMocks();
      queryClient = new QueryClient({
        defaultOptions: { queries: { retry: false } },
      });
      const mockResponse = buildUserListResponse(1, 1);
      (userListCall as any).mockResolvedValue(mockResponse);
      mockUseAuthorized.mockReturnValue({ ...DEFAULT_AUTH, userRole: role });

      const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(userListCall).toHaveBeenCalledTimes(1);
    }
  });

  it("should handle error when userListCall fails", async () => {
    const testError = new Error("Failed to fetch users");
    (userListCall as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
  });

  it("should pass empty string searchEmail as null", async () => {
    const mockResponse = buildUserListResponse(1, 1);
    (userListCall as any).mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useInfiniteUsers(50, ""), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(userListCall).toHaveBeenCalledWith(
      "test-access-token",
      null,
      1,
      50,
      null,
    );
  });
});
