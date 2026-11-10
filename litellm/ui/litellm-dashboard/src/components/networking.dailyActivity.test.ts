import { afterEach, describe, expect, it, vi } from "vitest";

import {
  cacheLeakageKeysCall,
  dailyActivityAggregatedCall,
  dailyActivityExportCall,
  dailyActivityKeySearchCall,
  dailyActivityModelTopKeysCall,
} from "./networking";
import type { DailyActivityEntity, DailyActivityRequest } from "./UsagePage/dailyActivityApi";

const originalFetch = global.fetch;

const captureFetch = () => {
  const mockFetch = vi.fn<typeof fetch>().mockImplementation(
    async () =>
      new Response(JSON.stringify({ results: [], metadata: {}, api_keys: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  );
  global.fetch = mockFetch;
  return mockFetch;
};

const requestedUrl = (mockFetch: ReturnType<typeof captureFetch>, callIndex = 0): URL =>
  new URL(String(mockFetch.mock.calls[callIndex][0]), "http://example.com");

const start = new Date("2025-01-05T00:00:00Z");
const end = new Date("2025-01-31T00:00:00Z");

const req = (overrides: Partial<DailyActivityRequest> = {}): DailyActivityRequest => ({
  accessToken: "sk-key",
  startTime: start,
  endTime: end,
  ...overrides,
});

afterEach(() => {
  global.fetch = originalFetch;
});

describe("dailyActivityAggregatedCall", () => {
  it.each<[DailyActivityEntity, string]>([
    ["user", "user_id"],
    ["team", "team_ids"],
    ["tag", "tags"],
    ["organization", "organization_ids"],
    ["customer", "end_user_ids"],
    ["agent", "agent_ids"],
  ])("GETs /%s/daily/activity/aggregated with %s filters", async (entity, param) => {
    const mockFetch = captureFetch();

    await dailyActivityAggregatedCall(entity, req({ entityIds: ["e1", "e2"] }));

    const url = requestedUrl(mockFetch);
    expect(url.pathname).toBe(`/${entity}/daily/activity/aggregated`);
    const expected = entity === "user" ? "e1" : "e1,e2";
    expect(url.searchParams.get(param)).toBe(expected);
    expect(url.searchParams.get("start_date")).toBe("2025-01-05");
    expect(url.searchParams.get("end_date")).toBe("2025-01-31");
    expect(url.searchParams.has("timezone")).toBe(true);
    expect(url.searchParams.has("page")).toBe(false);
    expect(url.searchParams.has("page_size")).toBe(false);
  });

  it("omits the entity filter when no ids are given", async () => {
    const mockFetch = captureFetch();

    await dailyActivityAggregatedCall("team", req({ entityIds: null }));

    const url = requestedUrl(mockFetch);
    expect(url.searchParams.has("team_ids")).toBe(false);
    expect(url.searchParams.has("exclude_team_ids")).toBe(false);
  });

  it.each<[DailyActivityEntity, string]>([
    ["team", "exclude_team_ids"],
    ["organization", "exclude_organization_ids"],
    ["customer", "exclude_end_user_ids"],
    ["agent", "exclude_agent_ids"],
  ])("sends exclude ids comma-joined under %s", async (entity, param) => {
    const mockFetch = captureFetch();

    await dailyActivityAggregatedCall(
      entity,
      req({ entityIds: ["e1"], excludeEntityIds: ["litellm-dashboard", "other"] }),
    );

    expect(requestedUrl(mockFetch).searchParams.get(param)).toBe("litellm-dashboard,other");
  });

  it.each<[DailyActivityEntity]>([["tag"], ["user"]])("emits no exclude param for %s", async (entity) => {
    const mockFetch = captureFetch();

    await dailyActivityAggregatedCall(entity, req({ entityIds: ["e1"], excludeEntityIds: ["litellm-dashboard"] }));

    const params = [...requestedUrl(mockFetch).searchParams.keys()];
    expect(params.some((key) => key.startsWith("exclude_"))).toBe(false);
  });

  it("keeps an empty api_key as a filter rather than widening the read", async () => {
    const mockFetch = captureFetch();

    await dailyActivityAggregatedCall("user", req({ apiKey: "" }));

    expect(requestedUrl(mockFetch).searchParams.get("api_key")).toBe("");
  });

  it("omits api_key entirely when none is given", async () => {
    const mockFetch = captureFetch();

    await dailyActivityAggregatedCall("user", req({ apiKey: null }));

    expect(requestedUrl(mockFetch).searchParams.has("api_key")).toBe(false);
  });

  it("sends include_current_utc_day only when set", async () => {
    const mockFetch = captureFetch();

    await dailyActivityAggregatedCall("user", req({ includeCurrentUtcDay: true }));

    expect(requestedUrl(mockFetch).searchParams.get("include_current_utc_day")).toBe("true");
  });

  it("sends api_key_limit only when provided", async () => {
    const mockFetch = captureFetch();

    await dailyActivityAggregatedCall("user", req());
    await dailyActivityAggregatedCall("user", req({ apiKeyLimit: 250 }));

    expect(requestedUrl(mockFetch, 0).searchParams.has("api_key_limit")).toBe(false);
    expect(requestedUrl(mockFetch, 1).searchParams.get("api_key_limit")).toBe("250");
  });
});

describe("dailyActivityKeySearchCall", () => {
  it("GETs the search route with the search term", async () => {
    const mockFetch = captureFetch();

    await dailyActivityKeySearchCall("team", req({ entityIds: ["t1"] }), "alice");

    const url = requestedUrl(mockFetch);
    expect(url.pathname).toBe("/team/daily/activity/aggregated/search");
    expect(url.searchParams.get("search")).toBe("alice");
    expect(url.searchParams.get("team_ids")).toBe("t1");
    expect(url.searchParams.has("limit")).toBe(false);
  });

  it("sends the optional limit", async () => {
    const mockFetch = captureFetch();

    await dailyActivityKeySearchCall("team", req(), "alice", 50);

    expect(requestedUrl(mockFetch).searchParams.get("limit")).toBe("50");
  });
});

describe("dailyActivityModelTopKeysCall", () => {
  it("GETs model_top_keys with model_group and by_model_group", async () => {
    const mockFetch = captureFetch();

    await dailyActivityModelTopKeysCall("user", req(), "gpt-4o", true);

    const url = requestedUrl(mockFetch);
    expect(url.pathname).toBe("/user/daily/activity/aggregated/model_top_keys");
    expect(url.searchParams.get("model_group")).toBe("gpt-4o");
    expect(url.searchParams.get("by_model_group")).toBe("true");
    expect(url.searchParams.has("limit")).toBe(false);
  });

  it("sends by_model_group=false for the models view", async () => {
    const mockFetch = captureFetch();

    await dailyActivityModelTopKeysCall("user", req(), "gpt-4o", false);

    expect(requestedUrl(mockFetch).searchParams.get("by_model_group")).toBe("false");
  });

  it("sends the optional limit", async () => {
    const mockFetch = captureFetch();

    await dailyActivityModelTopKeysCall("user", req(), "gpt-4o", true, 25);

    expect(requestedUrl(mockFetch).searchParams.get("limit")).toBe("25");
  });
});

describe("dailyActivityExportCall", () => {
  it("GETs the export route with export_type and format", async () => {
    const mockFetch = captureFetch();

    await dailyActivityExportCall("organization", req({ entityIds: ["o1"] }), "daily_with_keys", "csv");

    const url = requestedUrl(mockFetch);
    expect(url.pathname).toBe("/organization/daily/activity/export");
    expect(url.searchParams.get("export_type")).toBe("daily_with_keys");
    expect(url.searchParams.get("format")).toBe("csv");
    expect(url.searchParams.get("organization_ids")).toBe("o1");
  });
});

describe("cacheLeakageKeysCall", () => {
  it("GETs the user cache_leakage_keys route with the user scope", async () => {
    const mockFetch = captureFetch();

    await cacheLeakageKeysCall(req({ entityIds: ["u1"], apiKey: "hash-1", includeCurrentUtcDay: true }));

    const url = requestedUrl(mockFetch);
    expect(url.pathname).toBe("/user/daily/activity/aggregated/cache_leakage_keys");
    expect(url.searchParams.get("user_id")).toBe("u1");
    expect(url.searchParams.get("api_key")).toBe("hash-1");
    expect(url.searchParams.get("include_current_utc_day")).toBe("true");
    expect(url.searchParams.has("limit")).toBe(false);
  });

  it("sends the optional limit", async () => {
    const mockFetch = captureFetch();

    await cacheLeakageKeysCall(req(), 75);

    expect(requestedUrl(mockFetch).searchParams.get("limit")).toBe("75");
  });
});
