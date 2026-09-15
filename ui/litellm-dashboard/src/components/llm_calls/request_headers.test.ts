import { describe, expect, it } from "vitest";
import { buildPlaygroundHeaders, customHeadersFromPairs, parseStoredHeaderPairs } from "./request_headers";

describe("customHeadersFromPairs", () => {
  it("trims header names and drops rows without a name", () => {
    expect(
      customHeadersFromPairs([
        [" anthropic-beta ", "context-1m-2025-08-07"],
        ["", "orphan value"],
        ["   ", "whitespace name"],
        ["x-empty", ""],
      ]),
    ).toEqual({ "anthropic-beta": "context-1m-2025-08-07", "x-empty": "" });
  });
});

describe("parseStoredHeaderPairs", () => {
  it("round-trips pairs persisted as JSON", () => {
    const pairs = [["anthropic-beta", "context-1m-2025-08-07"]] as const;
    expect(parseStoredHeaderPairs(JSON.stringify(pairs))).toEqual(pairs);
  });

  it("returns no pairs for missing, malformed, or wrongly shaped storage", () => {
    expect(parseStoredHeaderPairs(null)).toEqual([]);
    expect(parseStoredHeaderPairs("not json")).toEqual([]);
    expect(parseStoredHeaderPairs(JSON.stringify({ "anthropic-beta": "x" }))).toEqual([]);
    expect(parseStoredHeaderPairs(JSON.stringify([["ok", "pair"], ["one"], [1, 2], "str"]))).toEqual([["ok", "pair"]]);
  });
});

describe("buildPlaygroundHeaders", () => {
  it("joins tags into x-litellm-tags and lets custom headers override it", () => {
    expect(buildPlaygroundHeaders(["a", "b"], { "x-custom": "1" })).toEqual({
      "x-litellm-tags": "a,b",
      "x-custom": "1",
    });
    expect(buildPlaygroundHeaders(["a"], { "x-litellm-tags": "b" })).toEqual({ "x-litellm-tags": "b" });
  });

  it("omits x-litellm-tags when there are no tags", () => {
    expect(buildPlaygroundHeaders([], { "x-custom": "1" })).toEqual({ "x-custom": "1" });
    expect(buildPlaygroundHeaders(undefined, undefined)).toEqual({});
  });
});
