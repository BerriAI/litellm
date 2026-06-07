import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useTags } from "./useTags";
import type { TagListResponse } from "@/components/tag_management/types";

const mockGet = vi.fn();
vi.mock("@/lib/http/api", () => ({
  fetchClient: { GET: (...args: unknown[]) => mockGet(...args) },
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

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

    vi.clearAllMocks();

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

  it("fetches /tag/list and returns the unwrapped body on success", async () => {
    mockGet.mockResolvedValue({ data: mockTags });

    const { result } = renderHook(() => useTags(), { wrapper });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockTags);
    expect(result.current.error).toBeNull();
    expect(mockGet).toHaveBeenCalledWith("/tag/list");
    expect(mockGet).toHaveBeenCalledTimes(1);
  });

  it("surfaces an error when the request rejects", async () => {
    const testError = new Error("Failed to fetch tags");
    mockGet.mockRejectedValue(testError);

    const { result } = renderHook(() => useTags(), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(mockGet).toHaveBeenCalledWith("/tag/list");
    expect(mockGet).toHaveBeenCalledTimes(1);
  });

  it("does not fetch when accessToken is missing", () => {
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

    expect(result.current.isLoading).toBe(false);
    expect(result.current.isFetched).toBe(false);
    expect(mockGet).not.toHaveBeenCalled();
  });

  it("does not fetch when userId is missing", () => {
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

    expect(result.current.isFetched).toBe(false);
    expect(mockGet).not.toHaveBeenCalled();
  });

  it("does not fetch when userRole is missing", () => {
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

    expect(result.current.isFetched).toBe(false);
    expect(mockGet).not.toHaveBeenCalled();
  });

  it("returns an empty object when the API returns empty data", async () => {
    mockGet.mockResolvedValue({ data: {} });

    const { result } = renderHook(() => useTags(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual({});
    expect(mockGet).toHaveBeenCalledWith("/tag/list");
  });

  it("falls back to an empty object when the response has no body", async () => {
    mockGet.mockResolvedValue({ data: undefined });

    const { result } = renderHook(() => useTags(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual({});
  });
});
