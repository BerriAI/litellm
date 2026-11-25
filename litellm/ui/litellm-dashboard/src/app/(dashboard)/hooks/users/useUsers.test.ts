import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { keepPreviousData, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import {
  useInfiniteUsers,
  useUserEmailLookup,
  useUserLookup,
  useUserIdentityLookup,
  userIdentityLabel,
} from "./useUsers";
import { userGetInfoV2, userListCall } from "@/components/networking";
import type { UserInfoV2Response, UserListResponse } from "@/components/networking";
import { ApiError } from "@/lib/http/client";
import type { CurrentUser } from "../currentUser/useCurrentUser";

vi.mock("@/components/networking", () => ({
  userGetInfoV2: vi.fn(),
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

const mockUseCurrentUser = vi.fn<(accessToken: string | null) => { data?: CurrentUser; isPlaceholderData?: boolean }>();
vi.mock("../currentUser/useCurrentUser", () => ({
  useCurrentUser: (accessToken: string | null) => mockUseCurrentUser(accessToken),
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

const buildUserListResponse = (page: number, totalPages: number, userCount = 2): UserListResponse => ({
  page,
  page_size: 50,
  total: totalPages * userCount,
  total_pages: totalPages,
  users: Array.from({ length: userCount }, (_, i) => ({
    user_id: `user-${page}-${i}`,
    user_email: `user-${page}-${i}@example.com`,
    user_alias: null,
    user_role: "Internal User",
    allowed_cache_controls: [],
    model_max_budget: {},
    model_spend: {},
    models: [],
    policies: [],
    teams: [],
    spend: 0,
    max_budget: null,
    key_count: 0,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    sso_user_id: null,
    budget_duration: null,
  })),
});

const buildUserInfoV2Response = (userId: string): UserInfoV2Response => ({
  user_id: userId,
  user_email: "alice@example.com",
  user_alias: "Alice",
  user_role: "Internal User",
  spend: 0,
  max_budget: null,
  models: [],
  budget_duration: null,
  budget_reset_at: null,
  metadata: null,
  created_at: null,
  updated_at: null,
  sso_user_id: null,
  teams: [],
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
    expect(userListCall).toHaveBeenCalledWith("test-access-token", null, 1, 50, null);
  });

  it("should use the default page size of 50", async () => {
    const mockResponse = buildUserListResponse(1, 1);
    (userListCall as any).mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(userListCall).toHaveBeenCalledWith("test-access-token", null, 1, 50, null);
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

    expect(userListCall).toHaveBeenCalledWith("test-access-token", null, 1, customPageSize, null);
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

    expect(userListCall).toHaveBeenCalledWith("test-access-token", null, 1, 50, searchEmail);
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

    expect(userListCall).toHaveBeenCalledWith("test-access-token", null, 1, 50, null);
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
    expect(userListCall).toHaveBeenLastCalledWith("test-access-token", null, 2, 50, null);
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
    const adminRoles = ["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer", "org_admin", "Org Admin"];

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

    expect(userListCall).toHaveBeenCalledWith("test-access-token", null, 1, 50, null);
  });
});

describe("useUserLookup", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue(DEFAULT_AUTH);
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("fetches exactly the requested user by id", async () => {
    const exactUser = buildUserInfoV2Response("user-1-0");
    vi.mocked(userGetInfoV2).mockResolvedValue(exactUser);

    const { result } = renderHook(() => useUserLookup("user-1-0"), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(userGetInfoV2).toHaveBeenCalledWith("test-access-token", "user-1-0");
    expect(userListCall).not.toHaveBeenCalled();
    expect(result.current.data).toMatchObject(exactUser);
    expect(result.current.data?.user_id).toBe("user-1-0");
  });

  it("resolves to no user when user info returns a different user id", async () => {
    vi.mocked(userGetInfoV2).mockResolvedValue(buildUserInfoV2Response("a+b"));

    const { result } = renderHook(() => useUserLookup("a b"), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toBeNull();
    expect(userListCall).not.toHaveBeenCalled();
  });

  it("resolves to no user when the user info endpoint returns a 404", async () => {
    vi.mocked(userGetInfoV2).mockRejectedValue(new ApiError("Not Found", 404, {}));

    const { result } = renderHook(() => useUserLookup("bob"), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toBeNull();
    expect(userListCall).not.toHaveBeenCalled();
  });

  it("rethrows errors other than 404", async () => {
    const serverError = new ApiError("Internal Server Error", 500, {});
    vi.mocked(userGetInfoV2).mockRejectedValue(serverError);

    const { result } = renderHook(() => useUserLookup("user-1-0"), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error).toBe(serverError);
    expect(userListCall).not.toHaveBeenCalled();
  });

  it("does not query without a user id", async () => {
    const { result } = renderHook(() => useUserLookup(null), { wrapper });

    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(result.current.fetchStatus).toBe("idle");
    expect(userGetInfoV2).not.toHaveBeenCalled();
  });

  it("does not query for a non-admin role", async () => {
    mockUseAuthorized.mockReturnValue({ ...DEFAULT_AUTH, userRole: "Internal User" });

    const { result } = renderHook(() => useUserLookup("user-1-0"), { wrapper });

    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(result.current.fetchStatus).toBe("idle");
    expect(userGetInfoV2).not.toHaveBeenCalled();
  });
});

describe("useUserEmailLookup", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue(DEFAULT_AUTH);
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("fetches the distinct ids in one call and maps each id to its email", async () => {
    const response = buildUserListResponse(1, 1, 2);
    vi.mocked(userListCall).mockResolvedValue(response);

    const { result } = renderHook(() => useUserEmailLookup(["user-1-1", "user-1-0", "user-1-1", ""]), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(userListCall).toHaveBeenCalledTimes(1);
    expect(userListCall).toHaveBeenCalledWith("test-access-token", ["user-1-0", "user-1-1"], 1, 2);
    expect(result.current.data).toEqual({
      "user-1-0": "user-1-0@example.com",
      "user-1-1": "user-1-1@example.com",
    });
  });

  it("omits users that have no email so callers fall back to the id", async () => {
    const response = buildUserListResponse(1, 1, 2);
    vi.mocked(userListCall).mockResolvedValue({
      ...response,
      users: [{ ...response.users[0], user_email: "" }, response.users[1]],
    });

    const { result } = renderHook(() => useUserEmailLookup(["user-1-0", "user-1-1"]), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ "user-1-1": "user-1-1@example.com" });
  });

  it("does not query with no ids", async () => {
    const { result } = renderHook(() => useUserEmailLookup([]), { wrapper });

    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(result.current.fetchStatus).toBe("idle");
    expect(userListCall).not.toHaveBeenCalled();
  });

  it("queries for the formatted Org Admin session role", async () => {
    mockUseAuthorized.mockReturnValue({ ...DEFAULT_AUTH, userRole: "Org Admin" });
    vi.mocked(userListCall).mockResolvedValue(buildUserListResponse(1, 1, 1));

    const { result } = renderHook(() => useUserEmailLookup(["user-1-0"]), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ "user-1-0": "user-1-0@example.com" });
  });

  it("does not query for a non-admin role", async () => {
    mockUseAuthorized.mockReturnValue({ ...DEFAULT_AUTH, userRole: "Internal User" });

    const { result } = renderHook(() => useUserEmailLookup(["user-1-0"]), { wrapper });

    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(result.current.fetchStatus).toBe("idle");
    expect(userListCall).not.toHaveBeenCalled();
  });
});

describe("useUserIdentityLookup", () => {
  let queryClient: QueryClient;
  const adminScope = { accessToken: "admin-token", userRole: "Admin" };
  const admin: CurrentUser = { user_id: "admin-id", user_email: "admin@example.com", user_role: "proxy_admin" };
  const member: CurrentUser = { user_id: "member-id", user_email: "member@example.com", user_role: "internal_user" };
  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
  const lookup = (
    ids: Parameters<typeof useUserIdentityLookup>[0],
    scope: Parameters<typeof useUserIdentityLookup>[1] = adminScope,
  ) => renderHook(() => useUserIdentityLookup(ids, scope), { wrapper });

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, placeholderData: keepPreviousData } } });
    vi.clearAllMocks();
    mockUseCurrentUser.mockReturnValue({ data: admin });
    vi.mocked(userGetInfoV2).mockRejectedValue(new ApiError("Not Found", 404, {}));
    vi.mocked(userListCall).mockRejectedValue(new ApiError("Forbidden", 403, {}));
  });

  afterEach(() => queryClient.clear());

  it.each([
    [" alice@example.com ", "Alice", "alice@example.com"],
    [null, " Alice ", "Alice"],
    [" ", " ", "id"],
  ])("labels by email, alias, then id", (user_email, user_alias, expected) => {
    expect(userIdentityLabel("id", { id: { user_email, user_alias } })).toBe(expected);
  });

  it("matches singleton by exact id, never email", async () => {
    vi.mocked(userGetInfoV2).mockResolvedValue(buildUserInfoV2Response("different-id"));
    const { result } = lookup(["alice@example.com"]);
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    expect(result.current).toEqual({});
    expect(userGetInfoV2).toHaveBeenCalledExactlyOnceWith("admin-token", "alice@example.com");
  });

  it.each([undefined, 403, 404])("resolves exact ids across batches, retaining successes after %s", async (status) => {
    const ids = Array.from({ length: 201 }, (_, index) => `user-${String(index).padStart(3, "0")}`);
    const template = buildUserListResponse(1, 1, 1);
    vi.mocked(userListCall).mockImplementation(async (_token, batch) => ({
      ...template,
      users: [...(batch ?? []), "unrequested-id"].map((user_id) => ({ ...template.users[0], user_id })),
    }));
    if (status) vi.mocked(userListCall).mockRejectedValueOnce(new ApiError("Unavailable", status, {}));
    vi.mocked(userGetInfoV2).mockResolvedValue(buildUserInfoV2Response(ids[200]));
    const { result } = lookup([...ids, ids[0]]);
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    expect(Object.keys(result.current)).toEqual(status ? ids.slice(100) : ids);
    expect(userIdentityLabel(ids[0], result.current)).toBe(status ? ids[0] : template.users[0].user_email);
    expect(userListCall).toHaveBeenCalledTimes(2);
    expect(userListCall).toHaveBeenNthCalledWith(1, "admin-token", ids.slice(0, 100), 1, 100);
    expect(userListCall).toHaveBeenNthCalledWith(2, "admin-token", ids.slice(100, 200), 1, 100);
    expect(userGetInfoV2).toHaveBeenCalledExactlyOnceWith("admin-token", ids[200]);
  });

  it.each([200, 403])("enriches only self, keeping authenticated email after status %s", async (status) => {
    mockUseCurrentUser.mockReturnValue({ data: member });
    if (status === 200) vi.mocked(userGetInfoV2).mockResolvedValue(buildUserInfoV2Response("member-id"));
    const { result } = lookup(["member-id", "other-id"], { accessToken: "member-token" });
    expect(userIdentityLabel("member-id", result.current)).toBe(member.user_email);
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    expect(result.current).toEqual({
      "member-id": { user_email: member.user_email, user_alias: status === 200 ? "Alice" : null },
    });
    expect(userGetInfoV2).toHaveBeenCalledExactlyOnceWith("member-token", "member-id");
    expect(userListCall).not.toHaveBeenCalled();
  });

  it.each([
    { ...adminScope, userRole: "Internal User", ids: ["other-id"] },
    { accessToken: "member-token", userRole: undefined, ids: ["member@example.com", "other-id"] },
    { ...adminScope, ids: [null, undefined, ""] },
    { ...adminScope, accessToken: null, ids: ["admin-id", "other-id"] },
  ])("does not query outside scope $accessToken / $userRole / $ids", ({ ids, ...scope }) => {
    if (scope.accessToken === "member-token") mockUseCurrentUser.mockReturnValue({ data: member });
    const { result } = lookup(ids, scope);
    expect(result.current).toEqual({});
    expect(mockUseCurrentUser).toHaveBeenCalledWith(scope.accessToken);
    expect(mockUseAuthorized).not.toHaveBeenCalled();
    expect(userListCall).not.toHaveBeenCalled();
    expect(userGetInfoV2).not.toHaveBeenCalled();
  });

  it.each(["supplied-role", "server-role", "read-only", "ids", "account", "logout"])(
    "clears privileged data immediately after a %s transition, including placeholders",
    async (change) => {
      const id = change === "account" ? "admin-id" : "other-id";
      vi.mocked(userGetInfoV2).mockResolvedValue(buildUserInfoV2Response(id));
      const initialProps: Parameters<typeof useUserIdentityLookup>[1] & { ids: string[] } = {
        ...adminScope,
        userRole: change === "account" ? undefined : adminScope.userRole,
        isViewOnly: false,
        ids: [id],
      };
      const { result, rerender } = renderHook(({ ids, ...scope }) => useUserIdentityLookup(ids, scope), {
        wrapper,
        initialProps,
      });
      await waitFor(() => expect(queryClient.isFetching()).toBe(0));
      expect(userIdentityLabel(id, result.current)).toBe(change === "account" ? admin.user_email : "alice@example.com");
      vi.mocked(userGetInfoV2).mockRejectedValue(new ApiError("Forbidden", 403, {}));
      if (change === "server-role")
        mockUseCurrentUser.mockReturnValue({ data: { ...admin, user_role: "internal_user" } });
      if (change === "account") mockUseCurrentUser.mockReturnValue({ data: admin, isPlaceholderData: true });
      const nextToken = change === "account" ? "member-token" : initialProps.accessToken;
      const nextProps = {
        accessToken: change === "logout" ? null : nextToken,
        userRole: change === "supplied-role" ? "Internal User" : initialProps.userRole,
        isViewOnly: change === "read-only",
        ids: change === "ids" ? ["new-id"] : initialProps.ids,
      };
      rerender(nextProps);
      expect(result.current).toEqual({});
      await waitFor(() => expect(queryClient.isFetching()).toBe(0));
      mockUseCurrentUser.mockReturnValue({ data: member });
      rerender(nextProps);
      expect(result.current).toEqual({});
      expect(userListCall).not.toHaveBeenCalled();
      if (change === "logout") expect(userGetInfoV2).toHaveBeenCalledTimes(1);
    },
  );
});
