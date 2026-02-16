/* @vitest-environment jsdom */
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useAccessGroups, AccessGroupResponse } from "./useAccessGroups";
import * as networking from "@/components/networking";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "http://proxy.example"),
  getGlobalLitellmHeaderName: vi.fn(() => "Authorization"),
  deriveErrorMessage: vi.fn((data: unknown) => (data as { detail?: string })?.detail ?? "Unknown error"),
  handleError: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(() => ({
    accessToken: "test-token-123",
    userRole: "Admin",
  })),
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = createQueryClient();
  return React.createElement(QueryClientProvider, { client: queryClient }, children);
};

const mockAccessToken = "test-token-123";
const mockAccessGroups: AccessGroupResponse[] = [
  {
    access_group_id: "ag-1",
    access_group_name: "Group One",
    description: "First group",
    access_model_names: [],
    access_mcp_server_ids: [],
    access_agent_ids: [],
    assigned_team_ids: [],
    assigned_key_ids: [],
    created_at: "2025-01-01T00:00:00Z",
    created_by: "user-1",
    updated_at: "2025-01-01T00:00:00Z",
    updated_by: "user-1",
  },
];

const fetchMock = vi.fn();

describe("useAccessGroups", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    vi.mocked(networking.getProxyBaseUrl).mockReturnValue("http://proxy.example");
    vi.mocked(networking.getGlobalLitellmHeaderName).mockReturnValue("Authorization");

    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: mockAccessToken,
      userRole: "Admin",
    } as any);

    global.fetch = fetchMock;
  });

  it("should return hook result without errors", () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve([]),
    } as Response);

    const { result } = renderHook(() => useAccessGroups(), { wrapper });

    expect(result.current).toBeDefined();
    expect(result.current).toHaveProperty("data");
    expect(result.current).toHaveProperty("isSuccess");
    expect(result.current).toHaveProperty("isError");
    expect(result.current).toHaveProperty("status");
  });

  it("should return access groups when access token and admin role are present", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockAccessGroups),
    } as Response);

    const { result } = renderHook(() => useAccessGroups(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://proxy.example/v1/access_group",
      expect.objectContaining({
        method: "GET",
        headers: expect.objectContaining({
          Authorization: `Bearer ${mockAccessToken}`,
          "Content-Type": "application/json",
        }),
      }),
    );
    expect(result.current.data).toEqual(mockAccessGroups);
  });

  it("should not fetch when access token is null", async () => {
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: null,
      userRole: "Admin",
    } as any);

    const { result } = renderHook(() => useAccessGroups(), { wrapper });

    expect(result.current.isFetching).toBe(false);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("should not fetch when access token is empty string", async () => {
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: "",
      userRole: "Admin",
    } as any);

    const { result } = renderHook(() => useAccessGroups(), { wrapper });

    expect(result.current.isFetching).toBe(false);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("should not fetch when user role is not an admin role", async () => {
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: mockAccessToken,
      userRole: "Viewer",
    } as any);

    const { result } = renderHook(() => useAccessGroups(), { wrapper });

    expect(result.current.isFetching).toBe(false);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("should not fetch when user role is null", async () => {
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: mockAccessToken,
      userRole: null,
    } as any);

    const { result } = renderHook(() => useAccessGroups(), { wrapper });

    expect(result.current.isFetching).toBe(false);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("should fetch when user role is proxy_admin", async () => {
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: mockAccessToken,
      userRole: "proxy_admin",
    } as any);

    fetchMock.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockAccessGroups),
    } as Response);

    const { result } = renderHook(() => useAccessGroups(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(fetchMock).toHaveBeenCalled();
    expect(result.current.data).toEqual(mockAccessGroups);
  });

  it("should expose error state when fetch fails", async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      json: () => Promise.resolve({ detail: "Forbidden" }),
    } as Response);
    vi.mocked(networking.deriveErrorMessage).mockReturnValue("Forbidden");

    const { result } = renderHook(() => useAccessGroups(), { wrapper });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect((result.current.error as Error).message).toBe("Forbidden");
    expect(result.current.data).toBeUndefined();
    expect(networking.handleError).toHaveBeenCalledWith("Forbidden");
  });

  it("should return empty array when API returns empty list", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve([]),
    } as Response);

    const { result } = renderHook(() => useAccessGroups(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual([]);
  });

  it("should propagate network errors", async () => {
    const networkError = new Error("Network failure");
    fetchMock.mockRejectedValue(networkError);

    const { result } = renderHook(() => useAccessGroups(), { wrapper });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(networkError);
    expect(result.current.data).toBeUndefined();
  });
});
