import { getUiSettings } from "@/components/networking";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useStdioMcpEnabled } from "./useStdioMcpEnabled";

vi.mock("@/components/networking", () => ({
  getUiSettings: vi.fn(),
}));

describe("useStdioMcpEnabled", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    vi.clearAllMocks();
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("returns true only when the proxy reports stdio enabled", async () => {
    vi.mocked(getUiSettings).mockResolvedValue({ values: { enable_stdio_mcp: true } });

    const { result } = renderHook(() => useStdioMcpEnabled(), { wrapper });

    await waitFor(() => {
      expect(result.current).toBe(true);
    });
  });

  it.each([{ values: { enable_stdio_mcp: false } }, { values: {} }, {}])(
    "returns false unless the proxy explicitly enables stdio",
    async (settings) => {
      vi.mocked(getUiSettings).mockResolvedValue(settings);

      const { result } = renderHook(() => useStdioMcpEnabled(), { wrapper });

      await waitFor(() => {
        expect(getUiSettings).toHaveBeenCalled();
      });
      expect(result.current).toBe(false);
    },
  );

  it("returns false while the UI settings request is unresolved", () => {
    vi.mocked(getUiSettings).mockReturnValue(new Promise(() => {}));

    const { result } = renderHook(() => useStdioMcpEnabled(), { wrapper });

    expect(result.current).toBe(false);
  });

  it("returns false when the UI settings request fails", async () => {
    vi.mocked(getUiSettings).mockRejectedValue(new Error("failed"));

    const { result } = renderHook(() => useStdioMcpEnabled(), { wrapper });

    await waitFor(() => {
      expect(getUiSettings).toHaveBeenCalled();
    });
    expect(result.current).toBe(false);
  });
});
