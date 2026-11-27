import { describe, expect, it } from "vitest";
import { appendFallbacks, replaceFallbacks } from "./mapper";

describe("fallback replacement lists", () => {
  it("appends groups in draft order without merging an existing primary", () => {
    const existing = [{ primary: ["original"] }, { unrelated: ["backup"] }];

    expect(
      appendFallbacks(existing, {
        groups: [
          { id: "1", primaryModel: "primary", fallbackModels: ["second", "first"] },
          { id: "2", primaryModel: "new", fallbackModels: ["backup"] },
        ],
      }),
    ).toEqual([...existing, { primary: ["second", "first"] }, { new: ["backup"] }]);
  });

  it("replaces the selected key without removing other keys or inserting a missing primary", () => {
    const group = { id: "edit", primaryModel: "primary", fallbackModels: ["new"] };

    expect(replaceFallbacks([{ primary: ["old"], sibling: ["keep"] }, { primary: ["other"] }], { group })).toEqual([
      { primary: ["new"], sibling: ["keep"] },
      { primary: ["new"] },
    ]);
    expect(replaceFallbacks([], { group })).toEqual([]);
  });
});
