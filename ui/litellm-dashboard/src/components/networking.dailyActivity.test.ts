import { afterEach, describe, expect, it, vi } from "vitest";

import {
  cacheLeakageKeysCall,
  dailyActivityAggregatedCall,
  dailyActivityExportCall,
  dailyActivityKeySearchCall,
  dailyActivityModelTopKeysCall,
  tagListCall,
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

  it("emits team_ids and group_by for tag", async () => {
    const mockFetch = captureFetch();

    await dailyActivityAggregatedCall("tag", req({ entityIds: ["shared"], teamIds: ["t1", "t2"], groupBy: "team" }));

    const url = requestedUrl(mockFetch);
    expect(url.searchParams.get("team_ids")).toBe("t1,t2");
    expect(url.searchParams.get("group_by")).toBe("team");
    expect(url.searchParams.get("tags")).toBe("shared");
  });

  it("emits tags for team", async () => {
    const mockFetch = captureFetch();

    await dailyActivityAggregatedCall("team", req({ entityIds: ["t1"], tags: ["shared", "x"], groupBy: "tag" }));

    const url = requestedUrl(mockFetch);
    expect(url.searchParams.get("tags")).toBe("shared,x");
    expect(url.searchParams.get("group_by")).toBe("tag");
    expect(url.searchParams.get("team_ids")).toBe("t1");
  });

  it("does not send the teamIds field on team routes or the tags field on tag routes", async () => {
    const mockFetch = captureFetch();

    await dailyActivityAggregatedCall("team", req({ teamIds: ["t1"], tags: ["shared"] }));
    await dailyActivityAggregatedCall("tag", req({ teamIds: ["t1"], tags: ["shared"] }));

    const teamCall = requestedUrl(mockFetch, 0);
    expect(teamCall.searchParams.has("team_ids")).toBe(false);
    expect(teamCall.searchParams.get("tags")).toBe("shared");
    const tagCall = requestedUrl(mockFetch, 1);
    expect(tagCall.searchParams.get("team_ids")).toBe("t1");
    expect(tagCall.searchParams.has("tags")).toBe(false);
    expect(tagCall.searchParams.has("exclude_tags")).toBe(false);
  });

  it("emits none of the new params when unset", async () => {
    const mockFetch = captureFetch();

    await dailyActivityAggregatedCall("tag", req());
    await dailyActivityAggregatedCall("team", req());

    for (const url of [requestedUrl(mockFetch, 0), requestedUrl(mockFetch, 1)]) {
      const keys = [...url.searchParams.keys()].sort();
      expect(keys).toEqual(["end_date", "start_date", "timezone"]);
    }
  });

  it("omits empty team_ids and tags lists", async () => {
    const mockFetch = captureFetch();

    await dailyActivityAggregatedCall("tag", req({ teamIds: [] }));
    await dailyActivityAggregatedCall("team", req({ tags: [] }));

    expect(requestedUrl(mockFetch, 0).searchParams.has("team_ids")).toBe(false);
    expect(requestedUrl(mockFetch, 1).searchParams.has("tags")).toBe(false);
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

describe("tagListCall", () => {
  const respondWith = (body: unknown) => {
    global.fetch = vi
      .fn<typeof fetch>()
      .mockResolvedValue(
        new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } }),
      );
  };

  it("returns the /tag/list array as-is", async () => {
    respondWith([{ name: "cc-shared" }, { name: "Café ☕" }]);

    expect(await tagListCall("sk-key")).toEqual([{ name: "cc-shared" }, { name: "Café ☕" }]);
  });

  it("returns an empty list rather than a non-array body, so callers can always .map()", async () => {
    respondWith({ "cc-shared": { name: "cc-shared" } });

    expect(await tagListCall("sk-key")).toEqual([]);
  });

  it("GETs /tag/list with no params when nothing is given", async () => {
    const mockFetch = captureFetch();

    await tagListCall("sk-key");

    const url = requestedUrl(mockFetch);
    expect(url.pathname).toBe("/tag/list");
    expect([...url.searchParams.keys()]).toEqual([]);
  });

  it("sends start and end dates when both are given", async () => {
    const mockFetch = captureFetch();

    await tagListCall("sk-key", start, end);

    const url = requestedUrl(mockFetch);
    expect(url.searchParams.get("start_date")).toBe("2025-01-05");
    expect(url.searchParams.get("end_date")).toBe("2025-01-31");
  });

  it("sends team_ids and usage_only with dates", async () => {
    const mockFetch = captureFetch();

    await tagListCall("sk-key", start, end, { teamIds: ["t1", "t2"], usageOnly: true });

    const url = requestedUrl(mockFetch);
    expect(url.searchParams.get("team_ids")).toBe("t1,t2");
    expect(url.searchParams.get("usage_only")).toBe("true");
    expect(url.searchParams.get("start_date")).toBe("2025-01-05");
    expect(url.searchParams.get("end_date")).toBe("2025-01-31");
  });

  it("sends team_ids and usage_only without dates", async () => {
    const mockFetch = captureFetch();

    await tagListCall("sk-key", null, null, { teamIds: ["t1"], usageOnly: true });

    const url = requestedUrl(mockFetch);
    expect(url.searchParams.get("team_ids")).toBe("t1");
    expect(url.searchParams.get("usage_only")).toBe("true");
    expect(url.searchParams.has("start_date")).toBe(false);
    expect(url.searchParams.has("end_date")).toBe(false);
  });
});
