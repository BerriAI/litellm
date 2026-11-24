import { describe, expect, it } from "vitest";
import { ApiError } from "@/lib/http/client";
import { grantedAppsFailed, readsEveryApp } from "./useGrantedApps";

describe("who reads every app", () => {
  it("should be a proxy admin and not an admin viewer, who needs a grant like any other user", () => {
    expect(readsEveryApp("Admin", false)).toBe(true);
    expect(readsEveryApp("Admin", true)).toBe(false);
    expect(readsEveryApp("Internal User", false)).toBe(false);
  });
});

describe("a failed read of granted apps", () => {
  it("should treat a 403 as having no grant and anything else as a failure to show", () => {
    expect(grantedAppsFailed(new ApiError("Forbidden.", 403, null))).toBe(false);
    expect(grantedAppsFailed(new ApiError("boom", 500, null))).toBe(true);
    expect(grantedAppsFailed(new TypeError("network down"))).toBe(true);
    expect(grantedAppsFailed(null)).toBe(false);
  });
});
