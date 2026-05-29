import { expect, test } from "@playwright/test";

// Driven by the SERVER_ROOT_PATH env var injected by the workflow; the container
// is booted with the same value, so the asset paths and the runtime config it
// serves at /litellm/.well-known/litellm-ui-config will both reflect it.
const ROOT_PATH = process.env.SERVER_ROOT_PATH ?? "";

test.skip(!ROOT_PATH, "Requires SERVER_ROOT_PATH env var");

// Contract: an unauthenticated visit must redirect to a login URL that preserves
// the SERVER_ROOT_PATH prefix. The redirect URL is built client-side from
// `proxyBaseUrl`, which is populated by an async fetch of the runtime UI config.
// If the redirect fires before that fetch resolves, the URL is missing the
// prefix and the user lands on a 404. To make the race deterministic across
// runners, the config endpoint is intentionally delayed.
test("unauth redirect preserves SERVER_ROOT_PATH prefix", async ({ page }) => {
  // Matches both `/litellm/.well-known/litellm-ui-config` and
  // `${SERVER_ROOT_PATH}/.well-known/litellm-ui-config` (the proxy rewrites the
  // bundle at boot when a root path is set).
  await page.route("**/.well-known/litellm-ui-config", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 500));
    await route.continue();
  });

  await page.context().clearCookies();

  await page.goto(`http://localhost:4000${ROOT_PATH}/ui/?page=virtual-keys`);

  await page.waitForURL((url) => url.pathname.endsWith("/ui/login"), { timeout: 15_000 });

  expect(page.url()).toContain(`${ROOT_PATH}/ui/login`);
});
