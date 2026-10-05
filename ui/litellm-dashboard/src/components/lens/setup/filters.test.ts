import { describe, expect, it } from "vitest";
import { normalizeFilters } from "./filters";

describe("Lens selection and findings", () => {
  it("preserves literal equals signs in a metadata value", () => {
    expect(normalizeFilters([{ key: " swarm ", value: " research=v2 " }])).toEqual([
      { key: "swarm", value: "research=v2" },
    ]);
  });
  it("rejects an incomplete condition instead of broadening the scan", () => {
    expect(() => normalizeFilters([{ key: "swarm", value: " " }])).toThrow("Choose a key and value");
  });
});
