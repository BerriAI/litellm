import { describe, expect, it } from "vitest";
import { effectiveLensEnabled, readLensPolicy } from "./lensPolicy";

describe("Lens logging policy", () => {
  it.each([
    { key: null, team: null, configured: false, expected: false },
    { key: null, team: null, configured: true, expected: true },
    { key: null, team: false, configured: true, expected: false },
    { key: true, team: false, configured: true, expected: true },
    { key: false, team: true, configured: true, expected: false },
    { key: null, team: true, configured: false, expected: true },
  ])("resolves key=$key, team=$team, configured=$configured to $expected", ({ key, team, configured, expected }) => {
    expect(effectiveLensEnabled(key, team, configured)).toBe(expected);
  });

  it.each([
    { metadata: { lens_enabled: false }, expected: false },
    { metadata: { lens_enabled: true }, expected: true },
    { metadata: { lens_enabled: null }, expected: null },
    { metadata: {}, expected: null },
    { metadata: null, expected: null },
    { metadata: { lens_enabled: "false" }, expected: null },
  ])("reads $metadata as $expected", ({ metadata, expected }) => {
    expect(readLensPolicy(metadata)).toBe(expected);
  });
});
