import { describe, expect, it } from "vitest";
import { getLiteAskSession } from "./session";

const now = 1_000_000;
const token = (claims: Record<string, unknown>) =>
  `e30.${Buffer.from(JSON.stringify(claims)).toString("base64url")}.sig`;
const claims = { user_role: "proxy_admin", user_id: "admin-one", key: "session-one", exp: now / 1000 + 60 };
const auth = {
  token: token(claims),
  accessToken: "session-one",
  userID: "admin-one",
  authLoading: false,
  passwordResetRequired: false,
};

describe("LiteAsk session visibility", () => {
  it("accepts the matching unexpired proxy admin session", () => {
    expect(getLiteAskSession(auth, now)).toEqual({ expiresAt: now + 60_000 });
  });

  it.each(["proxy_admin_viewer", "Admin", "org_admin", "internal_user", "internal_user_viewer", undefined])(
    "rejects raw role %s even when the dashboard presents it as Admin",
    (role) => {
      expect(getLiteAskSession({ ...auth, token: token({ ...claims, user_role: role }) }, now)).toBeNull();
    },
  );

  it.each([
    { token: null },
    { token: "invalid" },
    { accessToken: null },
    { userID: null },
    { accessToken: "previous-session" },
    { userID: "previous-admin" },
    { authLoading: true },
    { passwordResetRequired: true },
    { token: token({ ...claims, exp: now / 1000 }) },
    { token: token({ ...claims, exp: "later" }) },
    { token: token({ ...claims, password_reset_required: true }) },
  ])("rejects expired, inconsistent, or incomplete session %j", (change) => {
    expect(getLiteAskSession({ ...auth, ...change }, now)).toBeNull();
  });

  it("leaves legacy sessions without JWT expiry subject to backend authorization", () => {
    expect(getLiteAskSession({ ...auth, token: token({ ...claims, exp: undefined }) }, now)).toEqual({
      expiresAt: null,
    });
  });
});
