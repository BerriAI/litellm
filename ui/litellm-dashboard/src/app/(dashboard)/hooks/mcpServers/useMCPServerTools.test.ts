/* @vitest-environment jsdom */
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useMCPServerTools } from "./useMCPServerTools";
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

  it("normalizes a bare array response and tools without descriptions", async () => {
    vi.mocked(networking.listMCPTools).mockResolvedValue([{ tool_name: "legacy" }, "plain"]);

    const { result } = renderHook(() => useMCPServerTools(["a"], new Set(["a"])), { wrapper });

    await waitFor(() =>
      expect(result.current.get("a")?.tools).toEqual([
        { name: "legacy", description: "" },
        { name: "plain", description: "" },
      ]),
    );
  });
});
