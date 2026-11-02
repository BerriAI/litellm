import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useClaudeCodeMarketplaceRequiresKey } from "./useClaudeCodeMarketplaceRequiresKey";

const renderRequiresKey = (respond: () => Promise<Response>) => {
  const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(respond);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const { result } = renderHook(() => useClaudeCodeMarketplaceRequiresKey(), {
    wrapper: ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    ),
  });
  const settled = () =>
    waitFor(() =>
      expect(
        client
          .getQueryCache()
          .getAll()
          .map((query) => query.state.status),
      ).toEqual(["success"]),
    );
  return { result, fetchSpy, settled };
};

describe("useClaudeCodeMarketplaceRequiresKey", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it.each([
    [{ claude_code_marketplace_auth_required: false }, false],
    [{ claude_code_marketplace_auth_required: true }, true],
    [{}, true],
  ])("reads %o from the UI config as requires key=%s", async (config, requiresKey) => {
    const { result, fetchSpy, settled } = renderRequiresKey(
      async () => new Response(JSON.stringify(config), { status: 200 }),
    );

    await settled();
    expect(result.current).toBe(requiresKey);
    expect(fetchSpy.mock.calls.map(([request]) => new URL((request as Request).url).pathname)).toEqual([
      "/.well-known/litellm-ui-config",
    ]);
  });

  it("assumes a key is needed while the proxy has not answered yet", async () => {
    const { result, fetchSpy } = renderRequiresKey(() => new Promise<Response>(() => {}));

    await waitFor(() => expect(fetchSpy).toHaveBeenCalled());
    expect(result.current).toBe(true);
  });
});
