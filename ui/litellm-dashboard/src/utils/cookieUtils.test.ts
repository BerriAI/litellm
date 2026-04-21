import { describe, it, expect, beforeEach, vi } from "vitest";
import { clearTokenCookies, getCookie, storeLoginToken } from "./cookieUtils";

describe("cookieUtils", () => {
  beforeEach(() => {
    document.cookie.split(";").forEach((c) => {
      document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/");
    });
    sessionStorage.clear();

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

    it("should return early when window is undefined (server-side rendering)", () => {
      const originalWindow = global.window;
      const originalDocument = global.document;

      // Mock server-side environment
      delete (global as any).window;
      delete (global as any).document;

      // This should not throw an error and should return early
      expect(() => clearTokenCookies()).not.toThrow();

      // Restore globals
      global.window = originalWindow;
      global.document = originalDocument;
    });

    it("should return early when document is undefined (server-side rendering)", () => {
      const originalDocument = global.document;

      // Mock server-side environment where document is undefined
      delete (global as any).document;

      // This should not throw an error and should return early
      expect(() => clearTokenCookies()).not.toThrow();

      // Restore globals
      global.document = originalDocument;
    });

    it("should add current path directory to paths array when different from root and /ui", () => {
      // Mock window.location.pathname using vi.stubGlobal
      const originalLocation = window.location;
      vi.stubGlobal('location', { ...originalLocation, pathname: '/custom/path/page.html' });

      // Spy on document.cookie to verify the paths being used
      const cookieSpy = vi.spyOn(document, 'cookie', 'set');

      clearTokenCookies();

      // Verify that cookies were cleared for /custom/path/ path
      expect(cookieSpy).toHaveBeenCalledWith(
        expect.stringContaining('path=/custom/path/')
      );

      vi.restoreAllMocks();
    });

    it("should not add current path directory when it's already in paths array", () => {
      // Mock window.location.pathname using vi.stubGlobal
      const originalLocation = window.location;
      vi.stubGlobal('location', { ...originalLocation, pathname: '/' });

      // Spy on document.cookie to count calls
      const cookieSpy = vi.spyOn(document, 'cookie', 'set');

      clearTokenCookies();

      // Count how many times each path was used
      const rootPathCalls = cookieSpy.mock.calls.filter(call =>
        call[0].includes('path=/;') || call[0].includes('path=/ ')
      );
      const uiPathCalls = cookieSpy.mock.calls.filter(call =>
        call[0].includes('path=/ui;') || call[0].includes('path=/ui ')
      );

      // Should have calls for root and /ui paths, but not duplicate root
      expect(rootPathCalls.length).toBeGreaterThan(0);
      expect(uiPathCalls.length).toBeGreaterThan(0);

      vi.restoreAllMocks();
    });

    it("should clear sessionStorage token", () => {
      sessionStorage.setItem("token", "stored-token");
      clearTokenCookies();
      expect(sessionStorage.getItem("token")).toBeNull();
    });
  });

  describe("storeLoginToken", () => {
    it("should store the token in sessionStorage", () => {
      storeLoginToken("my-jwt-token");
      expect(sessionStorage.getItem("token")).toBe("my-jwt-token");
    });

    it("should overwrite an existing token in sessionStorage", () => {
      storeLoginToken("old-token");
      expect(sessionStorage.getItem("token")).toBe("old-token");

      storeLoginToken("new-token");
      expect(sessionStorage.getItem("token")).toBe("new-token");
    });

    it("should not throw when window is undefined (server-side rendering)", () => {
      const originalWindow = global.window;
      delete (global as any).window;

      expect(() => storeLoginToken("token")).not.toThrow();

      global.window = originalWindow;
    });

    it("should not store empty string token", () => {
      storeLoginToken("");
      expect(sessionStorage.getItem("token")).toBeNull();
    });

    it("should not store whitespace-only token", () => {
      storeLoginToken("   ");
      expect(sessionStorage.getItem("token")).toBeNull();
    });

    it("should set a JS-accessible cookie at /ui path", () => {
      const cookieSpy = vi.spyOn(document, "cookie", "set");
      storeLoginToken("my-jwt-token");
      expect(cookieSpy).toHaveBeenCalledWith(
        expect.stringContaining("path=/ui")
      );
      vi.restoreAllMocks();
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

    it("should handle values containing '=' characters", () => {
      document.cookie = "token=abc=def=ghi; path=/";
      expect(getCookie("token")).toBe("abc=def=ghi");
    });

    it("should fall back to sessionStorage when cookie is not found", () => {
      sessionStorage.setItem("token", "session-stored-jwt");
      expect(getCookie("token")).toBe("session-stored-jwt");
    });

    it("should prefer cookie over sessionStorage", () => {
      document.cookie = "token=cookie-value; path=/";
      sessionStorage.setItem("token", "session-value");
      expect(getCookie("token")).toBe("cookie-value");
    });

    it("should not fall back to sessionStorage for non-token keys", () => {
      sessionStorage.setItem("other", "other-value");
      expect(getCookie("other")).toBeNull();
    });
  });
});
