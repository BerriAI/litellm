import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useInfiniteKeyAliases } from "./useKeyAliases";
import type { PaginatedKeyAliasResponse } from "@/components/networking";

// Mock networking module
vi.mock("@/components/networking", () => ({
  keyAliasesCall: vi.fn(),
}));

// Mock useAuthorized hook
const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

// Mock console methods to avoid noise
vi.spyOn(console, "log").mockImplementation(() => {});
vi.spyOn(console, "error").mockImplementation(() => {});

import { keyAliasesCall } from "@/components/networking";

const mockKeyAliasesCall = vi.mocked(keyAliasesCall);

const mockPage1: PaginatedKeyAliasResponse = {
  aliases: ["alias-1", "alias-2"],
  total_count: 3,
  current_page: 1,
  total_pages: 2,
  size: 2,
};

const mockPage2: PaginatedKeyAliasResponse = {
  aliases: ["alias-3"],
  total_count: 3,
  current_page: 2,
  total_pages: 2,
  size: 2,
};

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
};

describe("useInfiniteKeyAliases", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token" });
    mockKeyAliasesCall.mockResolvedValue(mockPage1);
  });

  it("should fetch the first page of key aliases", async () => {
    const wrapper = createWrapper();
    const { result } = renderHook(() => useInfiniteKeyAliases(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(mockKeyAliasesCall).toHaveBeenCalledWith("test-token", 1, 50, undefined);
    expect(result.current.data?.pages[0]).toEqual(mockPage1);
  });

  it("should pass custom size parameter", async () => {
    const wrapper = createWrapper();
    renderHook(() => useInfiniteKeyAliases(25), { wrapper });

    await waitFor(() => {
      expect(mockKeyAliasesCall).toHaveBeenCalledWith("test-token", 1, 25, undefined);
    });
  });

  it("should pass search parameter when provided", async () => {
    const wrapper = createWrapper();
    renderHook(() => useInfiniteKeyAliases(50, "my-alias"), { wrapper });

    await waitFor(() => {
      expect(mockKeyAliasesCall).toHaveBeenCalledWith("test-token", 1, 50, "my-alias");
    });
  });

  it("should not fetch when accessToken is not available", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: null });
    const wrapper = createWrapper();
    const { result } = renderHook(() => useInfiniteKeyAliases(), { wrapper });

    expect(result.current.isFetching).toBe(false);
    expect(mockKeyAliasesCall).not.toHaveBeenCalled();
  });

  it("should expose hasNextPage when more pages are available", async () => {
    const wrapper = createWrapper();
    const { result } = renderHook(() => useInfiniteKeyAliases(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.hasNextPage).toBe(true);
  });

  it("should return hasNextPage false when on last page", async () => {
    const singlePage: PaginatedKeyAliasResponse = {
      aliases: ["alias-1"],
      total_count: 1,
      current_page: 1,
      total_pages: 1,
      size: 50,
    };
    mockKeyAliasesCall.mockResolvedValue(singlePage);

    const wrapper = createWrapper();
    const { result } = renderHook(() => useInfiniteKeyAliases(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.hasNextPage).toBe(false);
  });

  it("should fetch the next page when fetchNextPage is called", async () => {
    mockKeyAliasesCall
      .mockResolvedValueOnce(mockPage1)
      .mockResolvedValueOnce(mockPage2);

    const wrapper = createWrapper();
    const { result } = renderHook(() => useInfiniteKeyAliases(2), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    result.current.fetchNextPage();

    await waitFor(() => {
      expect(result.current.data?.pages).toHaveLength(2);
    });

    expect(mockKeyAliasesCall).toHaveBeenCalledWith("test-token", 2, 2, undefined);
    expect(result.current.data?.pages[1]).toEqual(mockPage2);
  });

  it("should include search in query key so search changes refetch from page 1", async () => {
    const wrapper = createWrapper();
    const { result, rerender } = renderHook(
      ({ search }: { search?: string }) => useInfiniteKeyAliases(50, search),
      { wrapper, initialProps: { search: undefined } },
    );

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    mockKeyAliasesCall.mockResolvedValue({
      aliases: ["search-result"],
      total_count: 1,
      current_page: 1,
      total_pages: 1,
      size: 50,
    });

    rerender({ search: "search-result" });

    await waitFor(() => {
      expect(mockKeyAliasesCall).toHaveBeenCalledWith("test-token", 1, 50, "search-result");
    });
  });
});
