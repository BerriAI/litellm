import { describe, expect, it } from "vitest";
import { isNewerVersion, parseReleaseVersion } from "./versionUtils";

describe("parseReleaseVersion", () => {
  it("reads the numeric components with or without a leading v", () => {
    expect(parseReleaseVersion("1.102.0")).toEqual([1, 102, 0]);
    expect(parseReleaseVersion("v1.102.0")).toEqual([1, 102, 0]);
  });

  it("ignores a prerelease suffix", () => {
    expect(parseReleaseVersion("1.102.0-dev.3")).toEqual([1, 102, 0]);
    expect(parseReleaseVersion("1.102.0.rc1")).toEqual([1, 102, 0]);
  });

  it("returns null for strings that are not a release version", () => {
    expect(parseReleaseVersion("")).toBeNull();
    expect(parseReleaseVersion("latest")).toBeNull();
    expect(parseReleaseVersion("1.102")).toBeNull();
  });
});

describe("isNewerVersion", () => {
  it("is false when the versions are equal", () => {
    expect(isNewerVersion("1.102.0", "1.102.0")).toBe(false);
    expect(isNewerVersion("1.102.0", "v1.102.0")).toBe(false);
  });

  it("is true when the latest version is ahead on any component", () => {
    expect(isNewerVersion("1.102.0", "1.102.1")).toBe(true);
    expect(isNewerVersion("1.102.5", "1.103.0")).toBe(true);
    expect(isNewerVersion("1.999.9", "2.0.0")).toBe(true);
  });

  it("is false when the running version is already ahead", () => {
    expect(isNewerVersion("1.103.0", "1.102.9")).toBe(false);
    expect(isNewerVersion("2.0.0", "1.999.9")).toBe(false);
  });

  it("compares components numerically rather than as strings", () => {
    expect(isNewerVersion("1.9.0", "1.10.0")).toBe(true);
    expect(isNewerVersion("1.10.0", "1.9.0")).toBe(false);
  });

  it("treats a prerelease of the latest version as not behind", () => {
    expect(isNewerVersion("1.102.0-dev.1", "1.102.0")).toBe(false);
    expect(isNewerVersion("1.101.0-dev.1", "1.102.0")).toBe(true);
  });

  it("is false when either version cannot be parsed", () => {
    expect(isNewerVersion("unknown", "1.102.0")).toBe(false);
    expect(isNewerVersion("1.102.0", "")).toBe(false);
  });
});
