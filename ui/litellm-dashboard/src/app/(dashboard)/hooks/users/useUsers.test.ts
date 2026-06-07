import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useInfiniteUsers } from "./useUsers";
import type { paths } from "@/lib/http/schema";

type UserListResponse = paths["/user/list"]["get"]["responses"][200]["content"]["application/json"];

// openapi-fetch captures globalThis.fetch when the client is created (at import time), so the
// mock must be installed before imports run — hence vi.hoisted. Per test we reconfigure its
// implementation; the captured reference stays the same.
const { fetchMock } = vi.hoisted(() => {
  const fetchMock = vi.fn();
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  return { fetchMock };
});

// api.ts only needs the base URL + auth header name from networking; mock those so the
// test doesn't pull in the whole networking module.
vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: () => "http://localhost:4000",
  getGlobalLitellmHeaderName: () => "Authorization",
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

const buildUserListResponse = (page: number, totalPages: number, userCount = 2): UserListResponse =>
  ({
    page,
    page_size: 50,
    total: totalPages * userCount,
    total_pages: totalPages,
    users: Array.from({ length: userCount }, (_, i) => ({
      user_id: `user-${page}-${i}`,
      user_email: `user-${page}-${i}@example.com`,
      user_role: "internal_user",
      key_count: 0,
    })),
  }) as UserListResponse;

const requestOf = (arg: unknown): Request => arg as Request;

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

/** Serve buildUserListResponse for whatever `page` the request asks for. */
const stubPagedFetch = (totalPages: number, userCount = 2) => {
  fetchMock.mockImplementation(async (arg: unknown) => {
    const url = new URL(requestOf(arg).url);
    const page = Number(url.searchParams.get("page") ?? "1");
    return jsonResponse(buildUserListResponse(page, totalPages, userCount));
  });
  return fetchMock;
};

const lastRequestUrl = (mock: typeof fetchMock): URL => new URL(requestOf(mock.mock.calls.at(-1)![0]).url);

describe("useInfiniteUsers", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    fetchMock.mockReset();
    mockUseAuthorized.mockReset();
    mockUseAuthorized.mockReturnValue(DEFAULT_AUTH);
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("returns the first page of typed user data on success", async () => {
    stubPagedFetch(2);
    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.pages).toHaveLength(1);
    expect(result.current.data?.pages[0].users[0].user_id).toBe("user-1-0");
  });

  it("requests page 1 with the default page size and the bearer auth header", async () => {
    const mock = stubPagedFetch(1);
    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const url = lastRequestUrl(mock);
    expect(url.pathname).toBe("/user/list");
    expect(url.searchParams.get("page")).toBe("1");
    expect(url.searchParams.get("page_size")).toBe("50");
    expect(requestOf(mock.mock.calls[0][0]).headers.get("Authorization")).toBe("Bearer test-access-token");
  });

  it("sends a custom page size when provided", async () => {
    const mock = stubPagedFetch(1, 5);
    const { result } = renderHook(() => useInfiniteUsers(25), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(lastRequestUrl(mock).searchParams.get("page_size")).toBe("25");
  });

  it("maps searchEmail to the user_email query param", async () => {
    const mock = stubPagedFetch(1, 1);
    const { result } = renderHook(() => useInfiniteUsers(50, "search@example.com"), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(lastRequestUrl(mock).searchParams.get("user_email")).toBe("search@example.com");
  });

  it("omits user_email when no searchEmail is given", async () => {
    const mock = stubPagedFetch(1);
    const { result } = renderHook(() => useInfiniteUsers(50), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(lastRequestUrl(mock).searchParams.has("user_email")).toBe(false);
  });

  it("fetches the next page with an incremented page param", async () => {
    const mock = stubPagedFetch(3);
    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.hasNextPage).toBe(true);

    result.current.fetchNextPage();

    await waitFor(() => {
      expect(result.current.isFetchingNextPage).toBe(false);
      expect(result.current.data?.pages).toHaveLength(2);
    });

    expect(result.current.data?.pages[1].page).toBe(2);
    expect(mock).toHaveBeenCalledTimes(2);
    expect(lastRequestUrl(mock).searchParams.get("page")).toBe("2");
  });

  it("has no next page on the last page", async () => {
    stubPagedFetch(1);
    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.hasNextPage).toBe(false);
  });

  it("surfaces an error when the request fails", async () => {
    fetchMock.mockImplementation(async () => jsonResponse({ detail: "boom" }, 500));

    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.data).toBeUndefined();
  });

  it.each([
    ["null token", { accessToken: null }],
    ["non-admin role", { userRole: "Internal User" }],
  ])("does not fire a request when gated by %s", async (_label, override) => {
    const mock = stubPagedFetch(1);
    mockUseAuthorized.mockReturnValue({ ...DEFAULT_AUTH, ...override });

    const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.isFetched).toBe(false);
    expect(mock).not.toHaveBeenCalled();
  });

  it("runs for each admin role", async () => {
    for (const userRole of ["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer", "org_admin"]) {
      fetchMock.mockReset();
      queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
      const mock = stubPagedFetch(1);
      mockUseAuthorized.mockReturnValue({ ...DEFAULT_AUTH, userRole });

      const { result } = renderHook(() => useInfiniteUsers(), { wrapper });

      await waitFor(() => expect(result.current.isSuccess).toBe(true));
      expect(mock).toHaveBeenCalledTimes(1);
    }
  });
});
