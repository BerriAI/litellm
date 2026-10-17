import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useDailyActivityRange, useScopedDailyActivityRange } from "./useDailyActivityRange";

// The sibling unit test mocks networking, so it pins the positional args array against itself and
// would still pass if the array and the two networking signatures drifted apart. Nothing else runs
// the real callers, which is what makes the ordering a comment-enforced rule. This drives the hook
// through the actual query serializer instead, so an argument appended or inserted on one side only
// lands its value on the wrong filter and fails here.
describe("useScopedDailyActivityRange wiring", () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
    vi.useRealTimers();
  });

  it("requests seven days including today on mount and preserves a wider selected range", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-03-09T12:00:00"));
    const mockFetch = vi.fn<typeof fetch>().mockImplementation(
      async () =>
        new Response(JSON.stringify({ results: [], metadata: {} }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    );
    global.fetch = mockFetch;
    const { result } = renderHook(() => useDailyActivityRange("sk-token", "u1", "proxy_admin"));

    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(1));
    const initialUrl = new URL(String(mockFetch.mock.calls[0][0]));
    expect(initialUrl.pathname).toBe("/user/daily/activity/aggregated");
    expect(initialUrl.searchParams.get("start_date")).toBe("2026-03-03");
    expect(initialUrl.searchParams.get("end_date")).toBe("2026-03-09");
    expect(initialUrl.searchParams.get("include_current_utc_day")).toBe("true");
    expect(initialUrl.searchParams.get("timezone")).toBe(String(new Date().getTimezoneOffset()));

    const selectedRange = { from: new Date(2026, 0, 1), to: new Date(2026, 2, 9, 23, 59, 59, 999) };
    act(() => result.current.onDateChange(selectedRange));

    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(2));
    const selectedUrl = new URL(String(mockFetch.mock.calls[1][0]));
    expect(selectedUrl.searchParams.get("start_date")).toBe("2026-01-01");
    expect(selectedUrl.searchParams.get("end_date")).toBe("2026-03-09");
    expect(result.current.dateValue).toEqual(selectedRange);
  });

  it("lands each scope field on its own query param through the real networking callers", async () => {
    const mockFetch = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ results: [], metadata: {} }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    global.fetch = mockFetch;

    const activity = {
      dateValue: { from: new Date(2026, 7, 1), to: new Date(2026, 7, 10) },
      onDateChange: vi.fn(),
    };
    renderHook(() => useScopedDailyActivityRange("sk-token", { userId: "u1", apiKey: "hash-abc" }, activity));

    await waitFor(() => expect(mockFetch).toHaveBeenCalled());

    const url = String(mockFetch.mock.calls[0][0]);
    expect(url).toContain("user_id=u1");
    expect(url).toContain("api_key=hash-abc");
    expect(url).toContain("start_date=2026-08-01");
    expect(url).toContain("end_date=2026-08-10");
  });
});
