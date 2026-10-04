import { describe, expect, it } from "vitest";

import { ApiError } from "@/lib/http/client";

import { readFailureCode, restartsTraversal } from "./readFailure";

const failure = (code: string, status = 409) => new ApiError("read failed", status, { detail: { code, message: "x" } });

describe("readFailureCode", () => {
  it("reads the stable code from a proxy read failure", () => {
    expect(readFailureCode(failure("traversal_changed"))).toBe("traversal_changed");
    expect(readFailureCode(failure("busy", 503))).toBe("busy");
  });

  it.each([
    ["plain error", new Error("offline")],
    ["string detail", new ApiError("x", 400, { detail: "Invalid trace cursor" })],
    ["unknown code", failure("not_a_code")],
    ["no body", new ApiError("x", 500, null)],
  ])("returns null for %s", (_label, error) => {
    expect(readFailureCode(error)).toBeNull();
  });
});

describe("restartsTraversal", () => {
  it.each(["traversal_changed", "traversal_expired", "invalid_cursor"])("restarts on %s", (code) => {
    expect(restartsTraversal(failure(code))).toBe(true);
  });

  it.each(["busy", "unavailable", "view_not_ready", "resource_too_large", "budget_exceeded", "invalid_request"])(
    "keeps the current cursor on %s",
    (code) => {
      expect(restartsTraversal(failure(code, 503))).toBe(false);
    },
  );

  it("keeps the current cursor for errors without a code", () => {
    expect(restartsTraversal(new Error("offline"))).toBe(false);
  });
});
