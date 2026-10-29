import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchClaudeCodeMarketplaceRequiresKey } from "./useClaudeCodeMarketplaceRequiresKey";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => ""),
}));

describe("fetchClaudeCodeMarketplaceRequiresKey", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("reports a key requirement when the marketplace refuses a request sent without credentials", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: {} }), { status: 401 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchClaudeCodeMarketplaceRequiresKey()).resolves.toBe(true);
    expect(fetchMock).toHaveBeenCalledWith(
      "/claude-code/marketplace.json",
      expect.objectContaining({ credentials: "omit" }),
    );
  });

  it("reports no key requirement when the marketplace serves its catalog", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ plugins: [] }), { status: 200 })));

    await expect(fetchClaudeCodeMarketplaceRequiresKey()).resolves.toBe(false);
  });

  it("fails the query instead of guessing when the marketplace answers another error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("{}", { status: 500 })));

    await expect(fetchClaudeCodeMarketplaceRequiresKey()).rejects.toThrow();
  });
});
