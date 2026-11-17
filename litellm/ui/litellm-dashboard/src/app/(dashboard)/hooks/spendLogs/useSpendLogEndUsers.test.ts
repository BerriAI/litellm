import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const useInfiniteQuery = vi.fn();
vi.mock("@/lib/http/api", () => ({ $api: { useInfiniteQuery: (...args: unknown[]) => useInfiniteQuery(...args) } }));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

import { nextPageFromLinks, useInfiniteSpendLogEndUsers } from "./useSpendLogEndUsers";

const WINDOW = { start_date: "2026-07-23 00:00:00", end_date: "2026-07-24 00:00:00" };

const page = (next: string | null) => ({
  data: ["cust-a"],
  meta: { page: 1, page_size: 50, has_more: next !== null },
  links: { self: "/management/v1/spend_logs/end_users?page=1", prev: null, next },
});

describe("useInfiniteSpendLogEndUsers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token" });
  });

  it("calls the control plane path", () => {
    renderHook(() => useInfiniteSpendLogEndUsers(WINDOW, 50));

    expect(useInfiniteQuery.mock.calls[0][1]).toBe("/management/v1/spend_logs/end_users");
  });

  it("sends the window as filter params and the page size as page_size", () => {
    renderHook(() => useInfiniteSpendLogEndUsers(WINDOW, 25));

    const query = useInfiniteQuery.mock.calls[0][2].params.query;
    expect(query).toEqual({
      "filter[startTime][gte]": "2026-07-23 00:00:00",
      "filter[startTime][lte]": "2026-07-24 00:00:00",
      page_size: 25,
    });
    expect(query).not.toHaveProperty("start_date");
    expect(query).not.toHaveProperty("end_date");
    expect(query).not.toHaveProperty("size");
  });

  it("sends free text as q, not search", () => {
    renderHook(() => useInfiniteSpendLogEndUsers(WINDOW, 50, "acme"));

    const query = useInfiniteQuery.mock.calls[0][2].params.query;
    expect(query.q).toBe("acme");
    expect(query).not.toHaveProperty("search");
  });

  it("omits q entirely when the search box is empty", () => {
    renderHook(() => useInfiniteSpendLogEndUsers(WINDOW, 50, ""));

    expect(useInfiniteQuery.mock.calls[0][2].params.query).not.toHaveProperty("q");
  });

  it("derives the next page from the server's links.next", () => {
    renderHook(() => useInfiniteSpendLogEndUsers(WINDOW, 50));

    const { getNextPageParam } = useInfiniteQuery.mock.calls[0][3];
    expect(getNextPageParam(page("/management/v1/spend_logs/end_users?page_size=50&page=7"))).toBe(7);
  });
});

describe("nextPageFromLinks", () => {
  it("reads the page the server pointed at rather than incrementing", () => {
    /* An endpoint that later switches to cursor pagination changes links.next and
       nothing else; a client that computed page+1 would silently break. */
    expect(nextPageFromLinks(page("/management/v1/spend_logs/end_users?page_size=50&page=7"))).toBe(7);
  });

  it("stops paging when the server omits links.next", () => {
    expect(nextPageFromLinks(page(null))).toBeUndefined();
  });
});
