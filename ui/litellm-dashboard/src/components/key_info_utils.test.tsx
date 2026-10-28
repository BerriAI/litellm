import { describe, it, expect } from "vitest";
import {
  filterSensitiveMetadata,
  extractLoggingSettings,
  formatMetadataForDisplay,
  stripTagsFromMetadata,
} from "./key_info_utils";

describe("filterSensitiveMetadata", () => {
  it("removes sensitive top-level fields like 'logging' while preserving others", () => {
    const input = {
      a: 1,
      logging: [{ level: "info" }],
      nested: { c: 2 },
      tags: ["x"],
    };
    const result = filterSensitiveMetadata(input);
    expect(result).toEqual({
      a: 1,
      nested: { c: 2 },
      tags: ["x"],
    });
    expect((result as any).logging).toBeUndefined();
  });
});

describe("extractLoggingSettings", () => {
  it("returns the logging array when present; returns the same reference", () => {
    const loggingRef = [{ enabled: true, destination: "s3" }];
    const input = { logging: loggingRef, other: 42 };
    const extracted = extractLoggingSettings(input);
    expect(extracted).toBe(loggingRef);
    expect(extracted).toEqual([{ enabled: true, destination: "s3" }]);
  });
});

describe("formatMetadataForDisplay", () => {
  it("stringifies metadata without sensitive fields like 'logging'", () => {
    const input = {
      logging: [{ level: "error" }],
      visible: "ok",
    };
    const output = formatMetadataForDisplay(input); // default indent = 2
    const expected = JSON.stringify({ visible: "ok" }, null, 2);
    expect(output).toBe(expected);
    expect(output).not.toContain("logging");
  });
});

describe("stripTagsFromMetadata", () => {
  it("removes top-level 'tags' but leaves other properties intact and does not mutate input", () => {
    const input = { tags: ["a", "b"], keep: { x: 1 } };
    const originalCopy = JSON.parse(JSON.stringify(input));
    const result = stripTagsFromMetadata(input);
    expect(result).toEqual({ keep: { x: 1 } });
    // Ensure original input is not mutated
    expect(input).toEqual(originalCopy);
  });
});
