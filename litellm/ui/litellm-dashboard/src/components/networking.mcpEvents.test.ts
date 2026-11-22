import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchMCPEventSubscriptions } from "./networking";

const originalFetch = global.fetch;

afterEach(() => {
  global.fetch = originalFetch;
});

const page = (ids: string[], nextOffset: number | null) =>
  new Response(JSON.stringify({ subscriptions: ids.map((id) => ({ id })), next_offset: nextOffset }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });

describe("fetchMCPEventSubscriptions", () => {
  it("follows next_offset until the last page", async () => {
    const mockFetch = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(page(["sub_1"], 500))
      .mockResolvedValueOnce(page(["sub_2"], null));
    global.fetch = mockFetch;
    const result = await fetchMCPEventSubscriptions("sk-admin");
    expect(result.subscriptions.map((item) => item.id)).toEqual(["sub_1", "sub_2"]);
    const offsets = mockFetch.mock.calls.map(([url]) =>
      new URL(String(url), "http://example.com").searchParams.get("offset"),
    );
    expect(offsets).toEqual(["0", "500"]);
  });

  it("stops after the page budget and reports where it stopped", async () => {
    global.fetch = vi.fn<typeof fetch>().mockImplementation(async () => page(["sub"], 500));
    const result = await fetchMCPEventSubscriptions("sk-admin", 0, 2);
    expect(result.next_offset).toBe(500);
    expect(result.subscriptions).toHaveLength(2);
  });
});
