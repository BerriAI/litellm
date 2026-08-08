/**
 * Unit tests for the snake_case ↔ camelCase wire transform.
 *
 * The SDK's public API is camelCase but the LiteLLM proxy speaks snake_case.
 * `snakeToCamel` runs over response JSON; `camelToSnake` runs over request
 * bodies. Only object keys are rewritten — values pass through unchanged.
 */

import { describe, expect, it } from "vitest";
import { camelToSnake, snakeToCamel } from "../src/client/http.js";

describe("snakeToCamel", () => {
  it("renames top-level snake_case keys", () => {
    expect(snakeToCamel({ agent_id: "x", created_at: "t" })).toEqual({
      agentId: "x",
      createdAt: "t",
    });
  });

  it("passes single-word keys through unchanged", () => {
    expect(snakeToCamel({ id: "x", type: "delta", data: 1, seq: 7 })).toEqual({
      id: "x",
      type: "delta",
      data: 1,
      seq: 7,
    });
  });

  it("recurses into nested objects", () => {
    expect(
      snakeToCamel({
        outer_field: { inner_field: { leaf_key: "v" } },
      }),
    ).toEqual({
      outerField: { innerField: { leafKey: "v" } },
    });
  });

  it("recurses into arrays of objects", () => {
    expect(
      snakeToCamel({
        items_list: [
          { item_id: 1, sub_field: "a" },
          { item_id: 2, sub_field: "b" },
        ],
      }),
    ).toEqual({
      itemsList: [
        { itemId: 1, subField: "a" },
        { itemId: 2, subField: "b" },
      ],
    });
  });

  it("preserves values verbatim (does not transform string values that look snake_case)", () => {
    expect(snakeToCamel({ system_prompt: "hello_world" })).toEqual({
      systemPrompt: "hello_world",
    });
  });

  it("handles keys with embedded digits", () => {
    expect(snakeToCamel({ agent_v2_id: "x" })).toEqual({ agentV2Id: "x" });
  });

  it("returns primitives unchanged", () => {
    expect(snakeToCamel(null)).toBe(null);
    expect(snakeToCamel(undefined)).toBe(undefined);
    expect(snakeToCamel("snake_string")).toBe("snake_string");
    expect(snakeToCamel(42)).toBe(42);
    expect(snakeToCamel(true)).toBe(true);
  });

  it("handles arrays at the top level", () => {
    expect(snakeToCamel([{ a_b: 1 }, { a_b: 2 }])).toEqual([
      { aB: 1 },
      { aB: 2 },
    ]);
  });
});

describe("camelToSnake", () => {
  it("renames top-level camelCase keys", () => {
    expect(camelToSnake({ agentId: "x", createdAt: "t" })).toEqual({
      agent_id: "x",
      created_at: "t",
    });
  });

  it("passes single-word keys through unchanged", () => {
    expect(camelToSnake({ id: "x", type: "delta", data: 1, seq: 7 })).toEqual({
      id: "x",
      type: "delta",
      data: 1,
      seq: 7,
    });
  });

  it("recurses into nested objects", () => {
    expect(
      camelToSnake({
        outerField: { innerField: { leafKey: "v" } },
      }),
    ).toEqual({
      outer_field: { inner_field: { leaf_key: "v" } },
    });
  });

  it("recurses into arrays of objects", () => {
    expect(
      camelToSnake({
        itemsList: [
          { itemId: 1, subField: "a" },
          { itemId: 2, subField: "b" },
        ],
      }),
    ).toEqual({
      items_list: [
        { item_id: 1, sub_field: "a" },
        { item_id: 2, sub_field: "b" },
      ],
    });
  });

  it("preserves string values verbatim", () => {
    expect(camelToSnake({ systemPrompt: "helloWorld" })).toEqual({
      system_prompt: "helloWorld",
    });
  });

  it("handles keys with embedded digits", () => {
    expect(camelToSnake({ agentV2Id: "x" })).toEqual({ agent_v2_id: "x" });
  });

  it("returns primitives unchanged", () => {
    expect(camelToSnake(null)).toBe(null);
    expect(camelToSnake(undefined)).toBe(undefined);
    expect(camelToSnake("camelString")).toBe("camelString");
    expect(camelToSnake(42)).toBe(42);
    expect(camelToSnake(false)).toBe(false);
  });
});

describe("round-trip", () => {
  it("camelToSnake → snakeToCamel preserves key names for typical SDK shapes", () => {
    const original = {
      agentId: "agt_1",
      createdAt: "2026-05-06",
      systemPrompt: "be helpful",
      nestedObject: { innerField: "v", anotherField: 7 },
      itemsList: [{ itemId: 1 }, { itemId: 2 }],
      // Single-word keys stay stable.
      id: "x",
      type: "delta",
      data: { foo: "bar" },
      seq: 0,
      status: "ready",
    };
    const round = snakeToCamel(camelToSnake(original));
    expect(round).toEqual(original);
  });

  it("snakeToCamel → camelToSnake preserves key names for typical wire shapes", () => {
    const original = {
      agent_id: "agt_1",
      created_at: "2026-05-06",
      system_prompt: "be helpful",
      nested_object: { inner_field: "v", another_field: 7 },
      items_list: [{ item_id: 1 }, { item_id: 2 }],
      next_cursor: null,
      id: "x",
      type: "delta",
      data: { foo: "bar" },
      seq: 0,
      status: "ready",
    };
    const round = camelToSnake(snakeToCamel(original));
    expect(round).toEqual(original);
  });
});
