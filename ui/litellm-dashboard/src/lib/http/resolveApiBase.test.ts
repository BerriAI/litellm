import { describe, expect, it } from "vitest";
import { resolveApiBase } from "./resolveApiBase";

describe("resolveApiBase", () => {
  describe("same-origin (no explicit base)", () => {
    it("returns an empty base so requests stay relative", () => {
      expect(resolveApiBase({})).toBe("");
      expect(resolveApiBase({ explicitBase: null, serverRootPath: "/" })).toBe("");
      expect(resolveApiBase({ explicitBase: "", serverRootPath: undefined })).toBe("");
    });

    it("prefixes a relative base with the server root path", () => {
      expect(resolveApiBase({ serverRootPath: "/litellm" })).toBe("/litellm");
    });
  });

  describe("explicit base (split-origin / dev)", () => {
    it("uses the explicit base verbatim when root path is mounted at /", () => {
      expect(resolveApiBase({ explicitBase: "http://localhost:4000", serverRootPath: "/" })).toBe(
        "http://localhost:4000",
      );
    });

    it("appends the server root path to the explicit base", () => {
      expect(resolveApiBase({ explicitBase: "http://localhost:4000", serverRootPath: "/litellm" })).toBe(
        "http://localhost:4000/litellm",
      );
    });
  });

  describe("dedup — base already includes the root path", () => {
    it("does not double the root path", () => {
      expect(resolveApiBase({ explicitBase: "http://localhost:4000/litellm", serverRootPath: "/litellm" })).toBe(
        "http://localhost:4000/litellm",
      );
    });
  });

  describe("normalization", () => {
    it("trims a trailing slash from the explicit base before appending", () => {
      expect(resolveApiBase({ explicitBase: "http://localhost:4000/", serverRootPath: "/litellm" })).toBe(
        "http://localhost:4000/litellm",
      );
    });

    it("trims a trailing slash from the explicit base when there is no root path", () => {
      expect(resolveApiBase({ explicitBase: "http://localhost:4000/", serverRootPath: "/" })).toBe(
        "http://localhost:4000",
      );
    });

    it("adds a leading slash to a root path that lacks one", () => {
      expect(resolveApiBase({ explicitBase: "http://localhost:4000", serverRootPath: "litellm" })).toBe(
        "http://localhost:4000/litellm",
      );
    });

    it("trims a trailing slash from the root path", () => {
      expect(resolveApiBase({ explicitBase: "http://localhost:4000", serverRootPath: "/litellm/" })).toBe(
        "http://localhost:4000/litellm",
      );
    });

    it("ignores whitespace around inputs", () => {
      expect(resolveApiBase({ explicitBase: "  http://localhost:4000  ", serverRootPath: "  /litellm  " })).toBe(
        "http://localhost:4000/litellm",
      );
    });
  });

  describe("multi-segment root paths", () => {
    it("appends a nested root path", () => {
      expect(resolveApiBase({ explicitBase: "http://localhost:4000", serverRootPath: "/team/litellm" })).toBe(
        "http://localhost:4000/team/litellm",
      );
    });

    it("dedups a nested root path", () => {
      expect(
        resolveApiBase({ explicitBase: "http://localhost:4000/team/litellm", serverRootPath: "/team/litellm" }),
      ).toBe("http://localhost:4000/team/litellm");
    });
  });
});
