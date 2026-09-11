import { describe, expect, it } from "vitest";

import { filterKeyActivity, keyActivityMatches } from "./keyActivityFilter";
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
});
