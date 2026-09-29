import { describe, expect, it } from "vitest";

import { filterKeyActivity, keyActivityMatches, parseKeyQuery } from "./keyActivityFilter";
import type { KeyMetadata, ModelActivityData } from "./types";

function activity(label: string, key_metadata?: KeyMetadata): ModelActivityData {
  return {
    label,
    key_metadata,
    total_requests: 1,
    total_successful_requests: 1,
    total_failed_requests: 0,
    total_cache_read_input_tokens: 0,
    total_cache_creation_input_tokens: 0,
    total_tokens: 10,
    prompt_tokens: 5,
    completion_tokens: 5,
    total_spend: 0.01,
    top_api_keys: [],
    top_models: [],
    daily_data: [],
  };
}

const aliceMeta: KeyMetadata = {
  key_alias: "alice-batch",
  team_id: "team-research",
  user_id: "user-alice-1234",
  user_email: "alice@example.com",
};
const bobMeta: KeyMetadata = {
  key_alias: null,
  team_id: "team-research",
  user_id: "user-bob-5678",
  user_email: "bob@example.com",
};
const alice = activity("alice-batch (team: research)", aliceMeta);
const bob = activity("bob@example.com (team: research)", bobMeta);
const orphan = activity("key-hash-deadbeef", { key_alias: null, team_id: null });

const keyMetrics: Record<string, ModelActivityData> = {
  "hash-alice": alice,
  "hash-bob": bob,
  deadbeef: orphan,
};

describe("keyActivityMatches", () => {
  it("matches every key on an empty or whitespace query", () => {
    expect(keyActivityMatches("deadbeef", orphan, "")).toBe(true);
    expect(keyActivityMatches("deadbeef", orphan, "   ")).toBe(true);
  });

  it("matches key alias case-insensitively", () => {
    expect(keyActivityMatches("hash-alice", alice, "ALICE-batch")).toBe(true);
    expect(keyActivityMatches("hash-bob", bob, "alice-batch")).toBe(false);
  });

  it("matches user email", () => {
    expect(keyActivityMatches("hash-bob", bob, "bob@example")).toBe(true);
    expect(keyActivityMatches("hash-alice", alice, "bob@example")).toBe(false);
  });

  it("matches user id", () => {
    expect(keyActivityMatches("hash-alice", alice, "user-alice-1234")).toBe(true);
    expect(keyActivityMatches("hash-bob", bob, "user-alice-1234")).toBe(false);
  });

  it("matches the key hash when the key has no alias or user metadata", () => {
    expect(keyActivityMatches("deadbeef", orphan, "dead")).toBe(true);
    expect(keyActivityMatches("deadbeef", orphan, "alice")).toBe(false);
  });

  it("trims surrounding whitespace from the query", () => {
    expect(keyActivityMatches("hash-alice", alice, "  alice@example.com  ")).toBe(true);
  });
});

describe("parseKeyQuery and keyActivityMatches", () => {
  it("matches anchored, case-insensitive globs", () => {
    expect(keyActivityMatches("hash-alice", alice, "alice-*")).toBe(true);
    expect(keyActivityMatches("hash-alice", alice, "batch-*")).toBe(false);
    expect(keyActivityMatches("hash-alice", activity("production batch-key", aliceMeta), "batch-*")).toBe(false);
    expect(keyActivityMatches("hash-alice", alice, "*-batch")).toBe(true);
    expect(keyActivityMatches("hash-alice", alice, "ALICE-*")).toBe(true);
  });

  it("treats regex punctuation literally in globs", () => {
    expect(keyActivityMatches("hash-alice", alice, "alice.batch*")).toBe(false);
  });

  it("matches the key hash with a glob", () => {
    expect(keyActivityMatches("deadbeef", orphan, "dead*")).toBe(true);
  });

  it("matches regular expressions with standard case sensitivity", () => {
    expect(keyActivityMatches("hash-bob", bob, "/^user-bob-\\d+$/")).toBe(true);
    expect(keyActivityMatches("hash-alice", alice, "/^user-bob-\\d+$/")).toBe(false);
    expect(keyActivityMatches("hash-alice", alice, "/^ALICE/")).toBe(false);
    expect(keyActivityMatches("hash-alice", alice, "/^ALICE/i")).toBe(true);
  });

  it("marks malformed regular expressions invalid and matches nothing", () => {
    expect(parseKeyQuery("/[/")).toEqual({ kind: "invalid", source: "/[/" });
    expect(keyActivityMatches("hash-alice", alice, "/[/")).toBe(false);
  });

  it("strips stateful flags from a regular expression reused for multiple matches", () => {
    const query = parseKeyQuery("/^alice/g");
    expect(query.kind).toBe("pattern");
    if (query.kind !== "pattern") return;

    expect(query.regex.flags).not.toContain("g");
    expect(query.regex.test("alice-batch")).toBe(true);
    expect(query.regex.test("alice-batch")).toBe(true);
    expect(
      Object.keys(
        filterKeyActivity({ "hash-alice": alice, "hash-alice-2": activity("alice-secondary", aliceMeta) }, "/^alice/g"),
      ),
    ).toEqual(["hash-alice", "hash-alice-2"]);
  });

  it("keeps a lone slash and a slash within text as substring queries", () => {
    expect(parseKeyQuery("/")).toEqual({ kind: "substring", needle: "/" });
    expect(parseKeyQuery("a/b")).toEqual({ kind: "substring", needle: "a/b" });
  });
});

describe("filterKeyActivity", () => {
  it("returns the same object when the query is blank", () => {
    expect(filterKeyActivity(keyMetrics, "")).toBe(keyMetrics);
  });

  it("keeps only the keys matching the query, preserving their hashes", () => {
    expect(Object.keys(filterKeyActivity(keyMetrics, "example.com"))).toEqual(["hash-alice", "hash-bob"]);
    expect(filterKeyActivity(keyMetrics, "user-bob")).toEqual({ "hash-bob": bob });
  });

  it("returns an empty record when nothing matches", () => {
    expect(filterKeyActivity(keyMetrics, "nobody")).toEqual({});
  });

  it("keeps the hashes matching a glob", () => {
    expect(filterKeyActivity(keyMetrics, "user-bob-*")).toEqual({ "hash-bob": bob });
  });
});
