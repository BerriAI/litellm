import { describe, expect, it } from "vitest";
import { navAccountDisplayName } from "./navDisplayName";

describe("navAccountDisplayName", () => {
  it("should prefer email when present", () => {
    expect(navAccountDisplayName("dev@example.com", "ignored", "Acme Developer")).toBe("dev@example.com");
  });

  it("shows the saved alias instead of a UUID when SSO supplies no email", () => {
    expect(navAccountDisplayName(null, "abc301ea-49c0-47c9-aee2-53b86a550582", "Acme Developer")).toBe(
      "Acme Developer",
    );
  });

  it("should map default_user_id placeholder to Account", () => {
    expect(navAccountDisplayName(null, "default_user_id")).toBe("Account");
    expect(navAccountDisplayName(null, "DEFAULT_USER_ID")).toBe("Account");
  });

  it("should show a sensible token when user id is non-placeholder", () => {
    expect(navAccountDisplayName(null, "user-uuid-123")).toBe("user-uuid-123");
  });
});
