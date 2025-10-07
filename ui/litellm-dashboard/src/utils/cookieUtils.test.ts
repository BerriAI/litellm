import { describe, it, expect, beforeEach, vi } from "vitest";
import { clearTokenCookies, getCookie } from "./cookieUtils";

describe("cookieUtils", () => {
  beforeEach(() => {
    document.cookie.split(";").forEach((c) => {
      document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/");
    });

    vi.spyOn(console, "log").mockImplementation(() => {});
  });

  describe("clearTokenCookies", () => {
    it("should clear token cookie from root path", () => {
      document.cookie = "token=test-token-value; path=/";
      expect(getCookie("token")).toBe("test-token-value");

      clearTokenCookies();
      expect(getCookie("token")).toBeNull();
    });

    it("should clear token cookie from /ui path", () => {
      document.cookie = "token=test-token-value; path=/ui";
      clearTokenCookies();
      expect(getCookie("token")).toBeNull();
    });

    it("should clear token cookies with different SameSite values", () => {
      document.cookie = "token=test-lax; path=/; SameSite=Lax";
      clearTokenCookies();
      expect(getCookie("token")).toBeNull();

      document.cookie = "token=test-strict; path=/; SameSite=Strict";
      clearTokenCookies();
      expect(getCookie("token")).toBeNull();
    });

    it("should handle multiple clearing attempts", () => {
      document.cookie = "token=test-value; path=/";

      clearTokenCookies();
      clearTokenCookies();
      clearTokenCookies();

      expect(getCookie("token")).toBeNull();
    });
  });

  describe("getCookie", () => {
    it("should return cookie value when it exists", () => {
      document.cookie = "token=my-test-token; path=/";
      expect(getCookie("token")).toBe("my-test-token");
    });

    it("should return null when cookie does not exist", () => {
      expect(getCookie("nonexistent")).toBeNull();
    });

    it("should handle JWT tokens with special characters", () => {
      const jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoidGVzdCJ9.signature";
      document.cookie = `token=${jwt}; path=/`;
      expect(getCookie("token")).toBe(jwt);
    });

    it("should return only the specified cookie", () => {
      document.cookie = "token=token-value; path=/";
      document.cookie = "other=other-value; path=/";

      expect(getCookie("token")).toBe("token-value");
      expect(getCookie("other")).toBe("other-value");
    });
  });
});
