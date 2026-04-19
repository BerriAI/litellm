/**
 * Unit coverage for the Playwright `guardedPage` fixture's URL classifier.
 *
 * The fixture itself needs a running Playwright test to be exercised, so this
 * test isolates the pure `isForbiddenRequestUrl` helper and pins its behavior
 * against the set of patterns that the double-prefix incident motivated.
 *
 * Axes covered:
 *   - double-prefix `/ui/ui/` regardless of resource type
 *   - API-verb-under-/ui/ only for xhr/fetch requests
 *   - document navigations to `/ui/<verb>` (pages like /ui/guardrails) must
 *     NOT be flagged
 *   - legitimate API paths at the root, cross-origin, and malformed URLs are
 *     never flagged
 */

import { describe, it, expect } from "vitest";
import { isForbiddenRequestUrl } from "../e2e_tests/fixtures/guarded-page";

const ORIGIN = "http://localhost:4000";

describe("isForbiddenRequestUrl", () => {
  describe("double-prefix /ui/ui/", () => {
    it("should flag a /ui/ui/ XHR", () => {
      const r = isForbiddenRequestUrl(`${ORIGIN}/ui/ui/project/list`, "xhr");
      expect(r.forbidden).toBe(true);
      expect(r.reason).toMatch(/double-prefix/);
    });

    it("should flag a /ui/ui/ document navigation too", () => {
      const r = isForbiddenRequestUrl(`${ORIGIN}/ui/ui/`, "document");
      expect(r.forbidden).toBe(true);
      expect(r.reason).toMatch(/double-prefix/);
    });

    it("should flag a /ui/ui/ script load", () => {
      const r = isForbiddenRequestUrl(`${ORIGIN}/ui/ui/foo.js`, "script");
      expect(r.forbidden).toBe(true);
    });

    it("should flag /ui/ui/ deep in path with query string", () => {
      const r = isForbiddenRequestUrl(`${ORIGIN}/ui/ui/key/info?a=1`, "xhr");
      expect(r.forbidden).toBe(true);
    });
  });

  describe("API verb nested under /ui/ (XHR/fetch only)", () => {
    const forbiddenApiPaths = [
      "/ui/key/info",
      "/ui/key/list",
      "/ui/team/list",
      "/ui/user/info?user_id=abc",
      "/ui/model/info",
      "/ui/models",
      "/ui/global/spend/logs",
      "/ui/spend/calculate",
      "/ui/customer/info",
      "/ui/organization/list",
      "/ui/health",
      "/ui/sso/key/generate",
      "/ui/config/update",
      "/ui/budget/settings",
      "/ui/tag/list",
      "/ui/public/model_hub/info",
      "/ui/invitation/new",
      "/ui/callbacks/configs",
    ];

    it.each(forbiddenApiPaths)("should flag XHR %s", (path) => {
      const r = isForbiddenRequestUrl(`${ORIGIN}${path}`, "xhr");
      expect(r.forbidden).toBe(true);
      expect(r.reason).toMatch(/API endpoint nested under/);
    });

    it.each(forbiddenApiPaths)("should flag fetch %s", (path) => {
      const r = isForbiddenRequestUrl(`${ORIGIN}${path}`, "fetch");
      expect(r.forbidden).toBe(true);
    });

    it.each(["document", "script", "stylesheet", "image", "font", "other"])(
      "should NOT flag non-data resource type %s for /ui/key/info (page navigation, not an API call)",
      (resourceType) => {
        const r = isForbiddenRequestUrl(`${ORIGIN}/ui/key/info`, resourceType);
        expect(r.forbidden).toBe(false);
      },
    );
  });

  describe("legitimate UI routes under /ui/ (never flagged)", () => {
    const routes = [
      "/",
      "/ui",
      "/ui/",
      "/ui/login",
      "/ui/virtual-keys",
      "/ui/models-and-endpoints",
      "/ui/teams",
      "/ui/organizations",
      "/ui/test-key",
      "/ui/chat",
      "/ui/logs",
      "/ui/model-hub",
      "/ui/api-reference",
    ];

    it.each(routes)("should allow document request %s", (path) => {
      const r = isForbiddenRequestUrl(`${ORIGIN}${path}`, "document");
      expect(r.forbidden).toBe(false);
    });

    // `/ui/guardrails`, `/ui/usage`, `/ui/prompts` are real UI pages whose
    // first segment collides with API verb names. They must not be flagged
    // as document navigations.
    it.each(["/ui/guardrails", "/ui/guardrails/edit", "/ui/usage", "/ui/prompts"])(
      "should allow API-verb-named document route %s",
      (path) => {
        const r = isForbiddenRequestUrl(`${ORIGIN}${path}`, "document");
        expect(r.forbidden).toBe(false);
      },
    );
  });

  describe("static assets under /ui/", () => {
    it.each([
      "/ui/_next/static/chunks/abc.js",
      "/ui/_next/data/abc.json",
      "/ui/assets/logos/openai.svg",
      "/ui/favicon.ico",
      "/ui/login/index.html",
      "/ui/foo.png",
      "/ui/anything.css",
      "/ui/anything.map",
      "/ui/anything.woff2",
    ])("should allow static asset %s", (path) => {
      const r = isForbiddenRequestUrl(`${ORIGIN}${path}`, "script");
      expect(r.forbidden).toBe(false);
    });
  });

  describe("legitimate API paths at the root (XHR/fetch)", () => {
    it.each([
      "/project/list",
      "/key/info",
      "/team/list",
      "/user/info",
      "/model/info",
      "/global/spend/logs",
      "/health",
      "/sso/key/generate",
      "/callbacks/configs",
    ])("should allow %s", (path) => {
      const r = isForbiddenRequestUrl(`${ORIGIN}${path}`, "xhr");
      expect(r.forbidden).toBe(false);
    });
  });

  describe("edge cases", () => {
    it("should allow cross-origin requests", () => {
      const r = isForbiddenRequestUrl("https://api.example.com/project/list", "xhr");
      expect(r.forbidden).toBe(false);
    });

    it("should not throw on malformed URL", () => {
      const r = isForbiddenRequestUrl("not a url", "xhr");
      expect(r.forbidden).toBe(false);
    });

    it("should default to XHR semantics when resourceType is omitted", () => {
      const r = isForbiddenRequestUrl(`${ORIGIN}/ui/key/info`);
      expect(r.forbidden).toBe(true);
    });
  });
});
