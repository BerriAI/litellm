import { describe, expect, it } from "vitest";

import { getResponseCacheStatus } from "./responseCache";

describe("getResponseCacheStatus", () => {
  it.each([
    { cache_hit: "True", cache_key: undefined, expected: "hit" },
    { cache_hit: "true", cache_key: "response-key", expected: "hit" },
    { cache_hit: "TRUE", cache_key: "Cache OFF", expected: "hit" },
    { cache_hit: "False", cache_key: undefined, expected: "miss" },
    { cache_hit: "false", cache_key: "response-key", expected: "miss" },
    { cache_hit: "FALSE", cache_key: "Cache OFF", expected: "miss" },
    { cache_hit: "None", cache_key: "response-key", expected: "miss" },
    { cache_hit: "", cache_key: "response-key", expected: "miss" },
    { cache_hit: "None", cache_key: "Cache OFF", expected: undefined },
    { cache_hit: "None", cache_key: undefined, expected: undefined },
    { cache_hit: "None", cache_key: "", expected: undefined },
    { cache_hit: "", cache_key: undefined, expected: undefined },
  ])("returns $expected for cache_hit=$cache_hit and cache_key=$cache_key", ({ cache_hit, cache_key, expected }) => {
    expect(getResponseCacheStatus({ cache_hit, cache_key })).toBe(expected);
  });
});
