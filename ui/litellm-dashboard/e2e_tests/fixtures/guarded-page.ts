/**
 * Playwright fixture: `guardedPage`.
 *
 * This module re-exports Playwright's `test` with the default `page`
 * fixture wrapped in a URL guard. Specs only need to change their import
 * source — no per-test code changes. The guard installs a
 * `page.on('request')` listener that records any browser request whose URL
 * matches a forbidden pattern (indicating a UI-routing regression such as the
 * double-prefix bug where a bundle built with `NEXT_PUBLIC_BASE_URL="ui/"`
 * causes API calls to resolve under `/ui/ui/...` or API endpoints to be
 * nested under `/ui/...`).
 *
 * After each test, the fixture fails the test with the full list of bad URLs
 * so the regression is obvious and actionable.
 *
 * Why this exists
 * ---------------
 * The Playwright specs in this suite mostly assert on visible UI text. With
 * client-rendered state, that can pass even when every API call 404s. This
 * fixture bridges that gap by asserting on the *wire format* directly.
 *
 * Patterns
 * --------
 * Forbidden for *any* request (including document navigation):
 *   - URL path containing `/ui/ui/`. This is the exact signature of the
 *     double-prefix bug.
 *
 * Forbidden only for XHR/fetch API requests (i.e. actual data loads):
 *   - A request to a *known* API path nested under `/ui/`, e.g.
 *     `/ui/key/info`, `/ui/team/list`. The list below is pulled from the
 *     verb prefixes observed in `networking.tsx` as of this commit.
 *
 * We intentionally do NOT flag document-type requests to `/ui/<verb>`
 * because several verbs collide with real UI routes (e.g. `/ui/guardrails`,
 * `/ui/usage`, `/ui/prompts` are legitimate dashboard pages). The
 * distinction is made by inspecting `request.resourceType()` and only
 * applying the API-verb list to `xhr` and `fetch` requests.
 *
 * Allowed:
 *   - `/ui`, `/ui/`, `/ui/login`, any HTML export route (document loads).
 *   - `/ui/_next/...` (Next.js client bundles + data files).
 *   - `/ui/assets/...`, `/ui/favicon.ico`, etc.
 *
 * Extending
 * ---------
 * When a new top-level API prefix is added to `networking.tsx`, update the
 * `FORBIDDEN_API_VERBS` array below. The goal is that *every* API path the
 * UI can call is reflected here. False positives (legit UI routes whose
 * first segment collides with an API verb) are possible in principle but
 * currently unused.
 */

import { test as base, expect, Page, TestInfo } from "@playwright/test";

/**
 * Top-level API path segments that should *never* appear under `/ui/...`.
 * Kept in sync with the `${proxyBaseUrl}/<segment>` calls in
 * `ui/litellm-dashboard/src/components/networking.tsx`.
 */
const FORBIDDEN_API_VERBS: string[] = [
  "add",
  "alerting",
  "audit",
  "budget",
  "cache",
  "callbacks",
  "claude-code",
  "cloudzero",
  "compliance",
  "config",
  "config_overrides",
  "cost",
  "credentials",
  "customer",
  "delete",
  "email",
  "get",
  "global",
  "guardrails",
  "health",
  "in_product_nudges",
  "invitation",
  "key",
  "mcp",
  "mcp-rest",
  "model",
  "model_group",
  "model_hub",
  "models",
  "onboarding",
  "openapi",
  "organization",
  "policies",
  "policy",
  "prompts",
  "public",
  "rag",
  "reload",
  "router",
  "schedule",
  "search_tools",
  "spend",
  "sso",
  "tag",
  "team",
  "toolset",
  "update",
  "usage",
  "user",
  "utils",
  "v",
  "vector_store",
];

/**
 * Paths under `/ui/...` that *are* legitimate UI assets and must not trip
 * the guard. Matched against the pathname with a leading `/`.
 */
const ALLOWED_UI_SUBPATHS: RegExp[] = [
  /^\/ui\/?$/,                    // /ui, /ui/
  /^\/ui\/_next\//,               // Next.js bundles + _next/data
  /^\/ui\/assets\//,              // static assets (logos, etc.)
  /^\/ui\/favicon\.ico$/,
  /\.(html|txt|js|css|map|png|jpe?g|svg|ico|gif|webp|woff2?|ttf|eot)$/i,
];

const DOUBLE_PREFIX_RE = /\/ui\/ui\//;
const FORBIDDEN_API_RE = new RegExp(
  `^/ui/(${FORBIDDEN_API_VERBS.map((v) => v.replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&")).join("|")})(?:/|$|\\?)`,
);

/**
 * Return true iff the request URL matches a forbidden pattern.
 *
 * `resourceType` is the Playwright `request.resourceType()` value. For
 * document/asset requests we only flag the `/ui/ui/` double-prefix; for
 * `xhr` / `fetch` we also flag any `/ui/<api-verb>/...` request since
 * that means the frontend is calling an API under its own static-asset
 * namespace.
 */
export function isForbiddenRequestUrl(
  urlString: string,
  resourceType: string = "xhr",
): {
  forbidden: boolean;
  reason?: string;
} {
  let parsed: URL;
  try {
    parsed = new URL(urlString);
  } catch {
    return { forbidden: false };
  }
  const path = parsed.pathname;

  // Absolute fail: the double-prefix regression, regardless of where it
  // appears in the path. Applies to every resource type.
  if (DOUBLE_PREFIX_RE.test(path)) {
    return { forbidden: true, reason: "double-prefix (/ui/ui/)" };
  }

  // Only the fetch/xhr resource types are subject to the API-verb check.
  // Document / stylesheet / script / image / font / other requests are
  // legitimate hits of the static export under `/ui/*` (including pages
  // like `/ui/guardrails` that collide with API verb names).
  const isDataRequest = resourceType === "xhr" || resourceType === "fetch";
  if (!isDataRequest) {
    return { forbidden: false };
  }

  // Only inspect `/ui/...` requests for the second class of violation.
  if (!path.startsWith("/ui/") && path !== "/ui") {
    return { forbidden: false };
  }

  // Allow-list UI-only subpaths.
  if (ALLOWED_UI_SUBPATHS.some((re) => re.test(path))) {
    return { forbidden: false };
  }

  if (FORBIDDEN_API_RE.test(path)) {
    return {
      forbidden: true,
      reason: `API endpoint nested under /ui/ (${path})`,
    };
  }

  return { forbidden: false };
}

/**
 * Re-exports a `test` whose `page` fixture is wrapped with the URL guard.
 *
 * Usage: change the import in a spec from
 *
 *     import { test, expect } from "@playwright/test";
 *
 * to
 *
 *     import { test, expect } from "../../fixtures/guarded-page";
 *
 * and every existing `({ page })` callback is automatically checked. No
 * other spec-level changes are required.
 */
export const test = base.extend<{}>({
  page: async ({ page }, use, testInfo: TestInfo) => {
    const violations: Array<{ url: string; reason: string; method: string }> = [];

    const onRequest = (req: ReturnType<Page["on"]> extends never ? never : any) => {
      const url: string = req.url();
      const method: string = req.method();
      const resourceType: string = req.resourceType();
      const check = isForbiddenRequestUrl(url, resourceType);
      if (check.forbidden) {
        violations.push({ url, method, reason: check.reason ?? "forbidden" });
      }
    };

    page.on("request", onRequest);

    try {
      await use(page);
    } finally {
      page.off("request", onRequest);
    }

    if (violations.length > 0) {
      const report = violations
        .map((v) => `  ${v.method} ${v.url}\n    → ${v.reason}`)
        .join("\n");
      await testInfo.attach("forbidden-request-urls.txt", {
        body: report,
        contentType: "text/plain",
      });
      // eslint-disable-next-line playwright/no-standalone-expect
      expect(
        violations,
        `Browser made ${violations.length} forbidden request(s). This usually means the UI ` +
          `bundle was built with a bogus NEXT_PUBLIC_BASE_URL and every API call is being ` +
          `routed under /ui/ (see PR #25109 for the double-prefix incident).\n\n` +
          report,
      ).toEqual([]);
    }
  },
});

export { expect };
