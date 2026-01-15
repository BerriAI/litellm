import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { isJwtExpired } from "./jwtUtils";
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
});
