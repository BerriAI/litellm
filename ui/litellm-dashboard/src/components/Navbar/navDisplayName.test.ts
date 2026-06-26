import { describe, expect, it } from "vitest";
import { navAccountDisplayName } from "./navDisplayName";

describe("navAccountDisplayName", () => {
  it("should prefer email when present", () => {
    expect(navAccountDisplayName("x@y.com", "ignored", "Account")).toBe("x@y.com");
  });

  it("should map default_user_id placeholder to the translated account label", () => {
    expect(navAccountDisplayName(null, "default_user_id", "账号")).toBe("账号");
    expect(navAccountDisplayName(null, "DEFAULT_USER_ID", "Account")).toBe("Account");
  });

  it("should show a sensible token when user id is non-placeholder", () => {
    expect(navAccountDisplayName(null, "user-uuid-123", "Account")).toBe("user-uuid-123");
  });
});
