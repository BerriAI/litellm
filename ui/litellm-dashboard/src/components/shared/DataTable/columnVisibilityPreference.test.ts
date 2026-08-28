import { describe, expect, it } from "vitest";

import { parseColumnVisibility } from "./columnVisibilityPreference";

const COLUMN_IDS = ["startTime", "model", "key_hash"] as const;

const parse = (raw: string | null) => parseColumnVisibility(raw, COLUMN_IDS);

describe("parseColumnVisibility", () => {
  it("treats a missing preference as every column visible", () => {
    expect(parse(null)).toEqual({});
  });

  it("falls back to every column visible when the stored value is not valid JSON", () => {
    expect(parse("{not json")).toEqual({});
  });

  it.each([
    ["an array", "[1,2]"],
    ["a JSON null", "null"],
    ["a bare number", "3"],
    ["a bare string", '"nope"'],
  ])("ignores %s, which cannot describe column visibility", (_label, raw) => {
    expect(parse(raw)).toEqual({});
  });

  it("keeps a hidden column that the user actually chose to hide", () => {
    expect(parse('{"key_hash":false}')).toEqual({ key_hash: false });
  });

  it("keeps an explicitly visible column", () => {
    expect(parse('{"model":true}')).toEqual({ model: true });
  });

  it("drops an id that is no longer a column, so a renamed column cannot resurrect stale state", () => {
    expect(parse('{"column_that_never_existed":false}')).toEqual({});
  });

  it("drops a non-boolean value instead of coercing it", () => {
    expect(parse('{"model":"false"}')).toEqual({});
  });

  it("keeps the known entries and drops the unknown ones from the same object", () => {
    expect(parse('{"model":true,"gone":false,"key_hash":false}')).toEqual({ model: true, key_hash: false });
  });
});
