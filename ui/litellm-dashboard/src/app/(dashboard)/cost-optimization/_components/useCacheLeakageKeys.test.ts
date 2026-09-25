import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/networking", () => ({ formatDate: vi.fn((d: Date) => d.toISOString().slice(0, 10)) }));
vi.mock("@/lib/http/api", () => ({ $api: { useQuery: vi.fn() } }));

import { $api } from "@/lib/http/api";

import { useCacheLeakageKeys } from "./useCacheLeakageKeys";

const lastCall = () =>
  vi.mocked($api.useQuery).mock.calls.at(-1) as unknown as [
    string,
    string,
    { params: { query: Record<string, unknown> } },
    { enabled: boolean; retry: boolean },
  ];

const range = { from: new Date("2026-07-06T19:00:00Z"), to: new Date("2026-08-05T19:00:00Z") };

describe("useCacheLeakageKeys", () => {
  it("queries the cache-leakage route with the picked range, the caller's UTC offset and the live-day extension", () => {
    useCacheLeakageKeys("sk-test", range, "user-9");

    const [, path, init, options] = lastCall();
    expect(path).toBe("/user/daily/activity/cache_leakage");
    expect(init.params.query).toEqual({
      start_date: "2026-07-06",
      end_date: "2026-08-05",
      timezone: new Date().getTimezoneOffset(),
      include_current_utc_day: true,
      user_id: "user-9",
    });
    expect(options.retry).toBe(false);
  });

  it("leaves the read deployment-wide when the caller has a proxy-wide spend view", () => {
    useCacheLeakageKeys("sk-test", range, null);

    const [, , init] = lastCall();
    expect(init.params.query.user_id).toBeUndefined();
  });

  it("stays disabled until a token and both range ends exist", () => {
    useCacheLeakageKeys(null, range, null);
    expect(lastCall()[3].enabled).toBe(false);

    useCacheLeakageKeys("sk-test", { from: range.from }, null);
    expect(lastCall()[3].enabled).toBe(false);

    useCacheLeakageKeys("sk-test", range, null);
    expect(lastCall()[3].enabled).toBe(true);
  });
});
