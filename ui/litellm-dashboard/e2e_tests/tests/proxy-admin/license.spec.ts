import { test, expect } from "@playwright/test";
import * as fs from "fs";
import { ADMIN_STORAGE_PATH } from "../../constants";

/**
 * Sanity check that LITELLM_LICENSE is being forwarded to the proxy when set
 * in the environment (e.g. CircleCI's `e2e_ui_testing` job). The login JWT's
 * `premium_user` claim is the same value the dashboard reads to enable
 * premium-gated UI surfaces (Team-BYOK switch, etc.), so asserting it here
 * catches any future regression where the env var stops being plumbed
 * through `run_e2e.sh` / `.circleci/config.yml`.
 *
 * Skips locally when no license is configured.
 */
test.describe("Premium license wiring", () => {
  test("admin session JWT carries premium_user=true when LITELLM_LICENSE is set", () => {
    test.skip(
      !process.env.LITELLM_LICENSE,
      "LITELLM_LICENSE not set in test env — proxy is running unlicensed",
    );

    const storage = JSON.parse(fs.readFileSync(ADMIN_STORAGE_PATH, "utf-8"));
    const tokenCookie = storage.cookies?.find((c: { name: string }) => c.name === "token");
    expect(tokenCookie, "token cookie missing from admin storage state").toBeDefined();

    // Decode the JWT payload (no signature check — we trust globalSetup ran
    // against our own proxy). Payload is the middle base64url segment.
    const jwtParts = tokenCookie.value.split(".");
    expect(jwtParts.length, "token cookie is not a 3-part JWT").toBe(3);
    const [, payloadB64] = jwtParts;
    const payload = JSON.parse(
      Buffer.from(payloadB64, "base64url").toString("utf-8"),
    );

    expect(payload.premium_user).toBe(true);
  });
});
