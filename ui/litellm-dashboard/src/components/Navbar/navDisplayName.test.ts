import { describe, expect, it } from "vitest";
import { navAccountDisplayName } from "./navDisplayName";

describe("navAccountDisplayName", () => {
  it("should prefer email when present", () => {
    expect(navAccountDisplayName("x@y.com", "ignored")).toBe("x@y.com");
  });

  it("should map default_user_id placeholder to Account", () => {
    expect(navAccountDisplayName(null, "default_user_id")).toBe("Account");
    expect(navAccountDisplayName(null, "DEFAULT_USER_ID")).toBe("Account");
  });

  it("should show a sensible token when user id is non-placeholder", () => {
    expect(navAccountDisplayName(null, "user-uuid-123")).toBe("user-uuid-123");
  });
});

it("localizes only the placeholder account label, preserving real account identities", () => {
  expect(navAccountDisplayName(null, "default_user_id", "账户")).toBe("账户");
  expect(navAccountDisplayName(null, "Account", "账户")).toBe("Account");
  expect(navAccountDisplayName("admin@example.com", null, "账户")).toBe("admin@example.com");
});
