import { describe, expect, it } from "vitest";
import { navAccountDisplayName } from "./navDisplayName";
import { translate, type Translate } from "@/i18n/language";

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

  it("仅汉化本地账户占位文案，保留真实用户标识", () => {
    const t: Translate = (message, values) => translate("zh-CN", message, values);
    expect(navAccountDisplayName(null, null, t)).toBe("账户");
    expect(navAccountDisplayName(null, "default_user_id", t)).toBe("账户");
    expect(navAccountDisplayName(null, "Account", t)).toBe("Account");
    expect(navAccountDisplayName("Home@example.com", null, t)).toBe("Home@example.com");
  });
});
