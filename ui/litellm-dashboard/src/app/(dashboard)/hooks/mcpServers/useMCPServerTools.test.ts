/* @vitest-environment jsdom */
import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { mcpServerToolKeys, useMCPServerTools } from "./useMCPServerTools";
import * as networking from "@/components/networking";

vi.mock("@/components/networking", () => ({
  listMCPTools: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(() => ({ accessToken: "test-token-123" })),
}));

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return React.createElement(QueryClientProvider, { client: queryClient }, children);
};

const toolsFor = (serverId: string) => ({
  tools: [{ name: `${serverId}_tool`, description: `from ${serverId}` }],
  error: null,
});

describe("useMCPServerTools", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("fetches only the enabled servers and leaves the rest idle", async () => {
    vi.mocked(networking.listMCPTools).mockImplementation(async (_token, serverId) => toolsFor(serverId));

    const { result } = renderHook(() => useMCPServerTools(["a", "b"], new Set(["a"])), { wrapper });

    await waitFor(() => expect(result.current.get("a")?.tools).toEqual([{ name: "a_tool", description: "from a" }]));
    expect(result.current.get("b")).toEqual({ tools: [], status: "idle" });
    expect(networking.listMCPTools).toHaveBeenCalledTimes(1);
    expect(networking.listMCPTools).toHaveBeenCalledWith("test-token-123", "a");
  });

  it("isolates a failing server from the others", async () => {
    vi.mocked(networking.listMCPTools).mockImplementation(async (_token, serverId) =>
      serverId === "bad" ? { tools: [], error: "http_500", message: "upstream down" } : toolsFor(serverId),
    );

    const { result } = renderHook(() => useMCPServerTools(["good", "bad"], new Set(["good", "bad"])), { wrapper });

    await waitFor(() => expect(result.current.get("bad")).toEqual({ tools: [], status: "error" }));
    expect(result.current.get("good")).toEqual({
      tools: [{ name: "good_tool", description: "from good" }],
      status: "success",
    });
  });

  it("reports a server that is being fetched as loading", () => {
    vi.mocked(networking.listMCPTools).mockReturnValue(new Promise(() => {}));

    const { result } = renderHook(() => useMCPServerTools(["a"], new Set(["a"])), { wrapper });

    expect(result.current.get("a")).toEqual({ tools: [], status: "loading" });
  });

  it("normalizes a bare array response, tools without descriptions, and drops tools without a name", async () => {
    vi.mocked(networking.listMCPTools).mockResolvedValue([
      { tool_name: "legacy" },
      "plain",
      { description: "nameless" },
    ]);

    const { result } = renderHook(() => useMCPServerTools(["a"], new Set(["a"])), { wrapper });

    await waitFor(() =>
      expect(result.current.get("a")?.tools).toEqual([
        { name: "legacy", description: "" },
        { name: "plain", description: "" },
      ]),
    );
  });

  it("reports an error when a refetch fails for a server that had loaded no tools", async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const sharedWrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(QueryClientProvider, { client: queryClient }, children);
    vi.mocked(networking.listMCPTools)
      .mockResolvedValueOnce({ tools: [], error: null })
      .mockResolvedValueOnce({ tools: [], error: "http_502", message: "upstream down" });

    const { result } = renderHook(() => useMCPServerTools(["a"], new Set(["a"])), { wrapper: sharedWrapper });
    await waitFor(() => expect(result.current.get("a")?.status).toBe("success"));

    await act(() => queryClient.refetchQueries({ queryKey: mcpServerToolKeys.detail("a") }));

    await waitFor(() => expect(result.current.get("a")).toEqual({ tools: [], status: "error" }));
  });

  it("keeps showing the loaded tools when a later refetch fails", async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const sharedWrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(QueryClientProvider, { client: queryClient }, children);
    vi.mocked(networking.listMCPTools)
      .mockResolvedValueOnce(toolsFor("a"))
      .mockResolvedValueOnce({ tools: [], error: "http_502", message: "upstream down" });

    const { result } = renderHook(() => useMCPServerTools(["a"], new Set(["a"])), { wrapper: sharedWrapper });
    await waitFor(() => expect(result.current.get("a")?.status).toBe("success"));

    await act(() => queryClient.refetchQueries({ queryKey: mcpServerToolKeys.detail("a") }));

    await waitFor(() => expect(queryClient.getQueryState(mcpServerToolKeys.detail("a"))?.status).toBe("error"));
    expect(networking.listMCPTools).toHaveBeenCalledTimes(2);
    expect(result.current.get("a")).toEqual({
      tools: [{ name: "a_tool", description: "from a" }],
      status: "success",
    });
  });
});
