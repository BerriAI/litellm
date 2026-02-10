import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { isJwtExpired, decodeToken, checkTokenValidity } from "./jwtUtils";
import { jwtDecode } from "jwt-decode";

vi.mock("jwt-decode");

describe("jwtUtils", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should return true if the token is expired", () => {
    const mockDateNow = 1716838401000;
    vi.spyOn(Date, "now").mockReturnValue(mockDateNow);
    vi.mocked(jwtDecode).mockReturnValue({
      exp: Math.floor(mockDateNow / 1000) - 1,
      user_id: "test",
    });

    expect(isJwtExpired("any-token")).toBe(true);
  });

  it("should return false if the token is not expired", () => {
    const mockDateNow = 1716838401000;
    vi.spyOn(Date, "now").mockReturnValue(mockDateNow);
    vi.mocked(jwtDecode).mockReturnValue({
      exp: Math.floor(mockDateNow / 1000) + 1000,
      user_id: "test",
    });

    expect(isJwtExpired("any-token")).toBe(false);
  });

  it("should return false if the token does not have an exp field", () => {
    vi.mocked(jwtDecode).mockReturnValue({
      user_id: "test",
    });

    expect(isJwtExpired("any-token")).toBe(false);
  });

  it("should return true if jwtDecode throws an error", () => {
    vi.mocked(jwtDecode).mockImplementation(() => {
      throw new Error("Invalid token");
    });

    expect(isJwtExpired("invalid-token")).toBe(true);
  });

  describe("decodeToken", () => {
    it("should return null if token is null", () => {
      expect(decodeToken(null)).toBeNull();
    });

    it("should return null if token is empty string", () => {
      expect(decodeToken("")).toBeNull();
    });

    it("should decode a valid token", () => {
      const mockPayload = {
        key: "api-key-123",
        user_id: "user-1",
        user_email: "user@example.com",
        user_role: "app_admin",
      };
      vi.mocked(jwtDecode).mockReturnValue(mockPayload);

      expect(decodeToken("valid-token")).toEqual(mockPayload);
      expect(jwtDecode).toHaveBeenCalledWith("valid-token");
    });

    it("should return null if jwtDecode throws an error", () => {
      vi.mocked(jwtDecode).mockImplementation(() => {
        throw new Error("Invalid token");
      });

      expect(decodeToken("invalid-token")).toBeNull();
    });
  });

  describe("checkTokenValidity", () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it("should return false if token is null", () => {
      expect(checkTokenValidity(null)).toBe(false);
    });

    it("should return false if token is empty string", () => {
      expect(checkTokenValidity("")).toBe(false);
    });

    it("should return true for a valid, non-expired token", () => {
      const mockDateNow = 1716838401000;
      vi.spyOn(Date, "now").mockReturnValue(mockDateNow);
      const mockPayload = {
        exp: Math.floor(mockDateNow / 1000) + 1000,
        user_id: "user-1",
      };
      vi.mocked(jwtDecode).mockReturnValue(mockPayload);

      expect(checkTokenValidity("valid-token")).toBe(true);
    });

    it("should return false for an expired token", () => {
      const mockDateNow = 1716838401000;
      vi.spyOn(Date, "now").mockReturnValue(mockDateNow);
      const mockPayload = {
        exp: Math.floor(mockDateNow / 1000) - 1,
        user_id: "user-1",
      };
      vi.mocked(jwtDecode).mockReturnValue(mockPayload);

      expect(checkTokenValidity("expired-token")).toBe(false);
    });

    it("should return false if token cannot be decoded", () => {
      vi.mocked(jwtDecode).mockImplementation(() => {
        throw new Error("Invalid token");
      });

      expect(checkTokenValidity("invalid-token")).toBe(false);
    });
  });
});
