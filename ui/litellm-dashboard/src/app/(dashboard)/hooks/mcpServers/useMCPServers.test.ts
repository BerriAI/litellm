/* @vitest-environment jsdom */
import React from "react";
import { renderHook, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useMCPServers, useToggleMCPServerEnabled } from "./useMCPServers";
import * as networking from "@/components/networking";

vi.mock("@/components/networking", () => ({
  fetchMCPServers: vi.fn(),
  enableMCPServerCall: vi.fn(),
  disableMCPServerCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(() => ({
    accessToken: "test-token-123",
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
const mockServers = [
  {
    server_id: "server-1",
    server_name: "Server One",
    url: "http://localhost:4000",
    created_at: "2025-01-01T00:00:00Z",
    created_by: "user-1",
    updated_at: "2025-01-01T00:00:00Z",
    updated_by: "user-1",
  },
];

describe("useMCPServers", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: mockAccessToken,
    } as any);
  });

  it("should return hook result without errors", () => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);

    const { result } = renderHook(() => useMCPServers(), { wrapper });

    expect(result.current).toBeDefined();
    expect(result.current).toHaveProperty("data");
    expect(result.current).toHaveProperty("isSuccess");
    expect(result.current).toHaveProperty("isError");
    expect(result.current).toHaveProperty("status");
  });

  it("should return MCP servers when access token is present", async () => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers);

    const { result } = renderHook(() => useMCPServers(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(networking.fetchMCPServers).toHaveBeenCalledWith(mockAccessToken, undefined, undefined, undefined);
    expect(result.current.data).toEqual(mockServers);
  });

  it("should not fetch when access token is null", async () => {
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: null,
    } as any);

    const { result } = renderHook(() => useMCPServers(), { wrapper });

    expect(result.current.isFetching).toBe(false);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(networking.fetchMCPServers).not.toHaveBeenCalled();
  });

  it("should not fetch when access token is empty string", async () => {
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: "",
    } as any);

    const { result } = renderHook(() => useMCPServers(), { wrapper });

    expect(result.current.isFetching).toBe(false);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(networking.fetchMCPServers).not.toHaveBeenCalled();
  });

  it("should expose error state when fetch fails", async () => {
    const mockError = new Error("Failed to fetch MCP servers");
    vi.mocked(networking.fetchMCPServers).mockRejectedValue(mockError);

    const { result } = renderHook(() => useMCPServers(), { wrapper });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(mockError);
    expect(result.current.data).toBeUndefined();
  });

  it("forwards the user scope to the server list call", async () => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers);

    const { result } = renderHook(() => useMCPServers(null, "user-7"), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(networking.fetchMCPServers).toHaveBeenCalledWith(mockAccessToken, null, undefined, "user-7");
  });

  it("should return empty array when API returns empty list", async () => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);

    const { result } = renderHook(() => useMCPServers(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual([]);
  });
});

describe("useToggleMCPServerEnabled", () => {
  const sharedWrapper = () => {
    const queryClient = createQueryClient();
    const w = ({ children }: { children: React.ReactNode }) =>
      React.createElement(QueryClientProvider, { client: queryClient }, children);
    return { w, queryClient };
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls enableMCPServerCall when enabling and refetches the list", async () => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers as any);
    vi.mocked(networking.enableMCPServerCall).mockResolvedValue({} as any);
    const { w } = sharedWrapper();
    const { result } = renderHook(() => ({ list: useMCPServers(), toggle: useToggleMCPServerEnabled() }), {
      wrapper: w,
    });
    await waitFor(() => expect(result.current.list.isSuccess).toBe(true));
    expect(networking.fetchMCPServers).toHaveBeenCalledTimes(1);

    await act(async () => {
      await result.current.toggle.mutateAsync({ serverId: "server-1", enabled: true });
    });

    expect(networking.enableMCPServerCall).toHaveBeenCalledWith(mockAccessToken, "server-1");
    expect(networking.disableMCPServerCall).not.toHaveBeenCalled();
    await waitFor(() => expect(networking.fetchMCPServers).toHaveBeenCalledTimes(2));
  });

  it("calls disableMCPServerCall when disabling", async () => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers as any);
    vi.mocked(networking.disableMCPServerCall).mockResolvedValue({} as any);
    const { w } = sharedWrapper();
    const { result } = renderHook(() => useToggleMCPServerEnabled(), { wrapper: w });

    await act(async () => {
      await result.current.mutateAsync({ serverId: "server-1", enabled: false });
    });

    expect(networking.disableMCPServerCall).toHaveBeenCalledWith(mockAccessToken, "server-1");
    expect(networking.enableMCPServerCall).not.toHaveBeenCalled();
  });

  it("optimistically updates cached list and rolls back on error", async () => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([{ ...mockServers[0], enabled: true }] as any);
    let reject: (e: Error) => void = () => {};
    vi.mocked(networking.disableMCPServerCall).mockImplementation(() => new Promise((_, rej) => (reject = rej)) as any);
    const { w } = sharedWrapper();
    const { result } = renderHook(() => ({ list: useMCPServers(), toggle: useToggleMCPServerEnabled() }), {
      wrapper: w,
    });
    await waitFor(() => expect(result.current.list.isSuccess).toBe(true));

    let p: Promise<unknown> = Promise.resolve();
    act(() => {
      p = result.current.toggle.mutateAsync({ serverId: "server-1", enabled: false }).catch(() => undefined);
    });
    await waitFor(() => expect(result.current.list.data?.[0].enabled).toBe(false));
    await act(async () => {
      reject(new Error("boom"));
      await p;
    });
    await waitFor(() => expect(result.current.toggle.isError).toBe(true));
    await waitFor(() => expect(result.current.list.data?.[0].enabled).toBe(true));
  });
});
