import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useScopedDailyActivityRange } from "./useDailyActivityRange";

// The sibling unit test mocks networking, so it pins the positional args array against itself and
// would still pass if the array and the two networking signatures drifted apart. Nothing else runs
// the real callers, which is what makes the ordering a comment-enforced rule. This drives the hook
// through the actual query serializer instead, so an argument appended or inserted on one side only
// lands its value on the wrong filter and fails here.
describe("useScopedDailyActivityRange wiring", () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("lands each scope field on its own query param through the real networking callers", async () => {
    const mockFetch = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ results: [], metadata: {} }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    global.fetch = mockFetch;

    renderHook(() => useScopedDailyActivityRange("sk-token", { userId: "u1", apiKey: "hash-abc" }));

    await waitFor(() => expect(mockFetch).toHaveBeenCalled());

    const url = String(mockFetch.mock.calls[0][0]);
    expect(url).toContain("user_id=u1");
    expect(url).toContain("api_key=hash-abc");
  });
});
