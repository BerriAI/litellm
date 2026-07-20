import { describe, expect, it } from "vitest";
import { PROTECTED_MODEL_INFO_KEYS, withNullsForRemovedKeys } from "./modelJsonPatchUtils";

describe("withNullsForRemovedKeys", () => {
  it("nulls keys removed from the edited JSON so PATCH merge deletes them", () => {
    const previous = { abc: 123, someKey: true, keep: "yes" };
    const next = { abc: 123, keep: "yes" };

    expect(withNullsForRemovedKeys(previous, next)).toEqual({
      abc: 123,
      keep: "yes",
      someKey: null,
    });
  });

  it("does not null keys listed in skip (masked secrets / credential)", () => {
    const previous = { api_key: "sk-****abcd", custom: 1 };
    const next = { custom: 1 };

    expect(withNullsForRemovedKeys(previous, next, new Set(["api_key"]))).toEqual({
      custom: 1,
    });
  });

  it("returns a copy of next when previous is missing", () => {
    const next = { a: 1 };
    expect(withNullsForRemovedKeys(undefined, next)).toEqual({ a: 1 });
    expect(withNullsForRemovedKeys(null, next)).toEqual({ a: 1 });
  });

  it("protects model_info identity fields via PROTECTED_MODEL_INFO_KEYS", () => {
    const previous = {
      id: "dep-1",
      team_id: "team-keep",
      someKey: true,
      abc: 1,
    };
    const next = { id: "dep-1", abc: 1 };

    expect(withNullsForRemovedKeys(previous, next, PROTECTED_MODEL_INFO_KEYS)).toEqual({
      id: "dep-1",
      abc: 1,
      someKey: null,
    });
  });
});
