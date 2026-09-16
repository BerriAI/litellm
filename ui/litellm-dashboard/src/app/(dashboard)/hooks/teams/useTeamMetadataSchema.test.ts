import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchTeamMetadataSchema } from "./useTeamMetadataSchema";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => ""),
  getGlobalLitellmHeaderName: vi.fn(() => "Authorization"),
}));

describe("fetchTeamMetadataSchema", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("should return the declared fields from the response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: async () => JSON.stringify({ fields: [{ key: "cost_center", label: "Cost Center", required: true }] }),
      }),
    );

    await expect(fetchTeamMetadataSchema("sk-test")).resolves.toEqual([
      { key: "cost_center", label: "Cost Center", required: true },
    ]);
  });

  it("should return an empty list when the response has no fields array", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, text: async () => "{}" }));

    await expect(fetchTeamMetadataSchema("sk-test")).resolves.toEqual([]);
  });

  it("should throw on a non-ok response so the query can retry and fail open", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 404, text: async () => "" }));

    await expect(fetchTeamMetadataSchema("sk-test")).rejects.toThrow("404");
  });
});
