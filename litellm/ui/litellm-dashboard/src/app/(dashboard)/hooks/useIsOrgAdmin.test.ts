/* @vitest-environment jsdom */
import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Organization } from "@/components/networking";
import useIsOrgAdmin from "./useIsOrgAdmin";

const { useAuthorizedMock, useOrganizationsMock } = vi.hoisted(() => ({
  useAuthorizedMock: vi.fn(),
  useOrganizationsMock: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: useAuthorizedMock }));
vi.mock("./organizations/useOrganizations", () => ({ useOrganizations: useOrganizationsMock }));

const orgWithMembers = (members: { user_id: string; user_role: string }[]): Organization =>
  ({ organization_id: "org-1", members }) as unknown as Organization;

const renderAs = (userRole: string, organizations: Organization[] | undefined) => {
  useAuthorizedMock.mockReturnValue({ userId: "user-1", userRole });
  useOrganizationsMock.mockReturnValue({ data: organizations });
  return renderHook(() => useIsOrgAdmin()).result;
};

describe("useIsOrgAdmin", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("is true for the session a real org admin carries: internal_user plus an org_admin membership", () => {
    const result = renderAs("Internal User", [orgWithMembers([{ user_id: "user-1", user_role: "org_admin" }])]);
    expect(result.current).toBe(true);
  });

  it("is false for an internal user with no org_admin membership", () => {
    const result = renderAs("Internal User", [orgWithMembers([{ user_id: "user-1", user_role: "internal_user" }])]);
    expect(result.current).toBe(false);
  });

  it("is false while the organization list is still loading", () => {
    const result = renderAs("Internal User", undefined);
    expect(result.current).toBe(false);
  });

  it("is true for a session role of org_admin even with no membership rows", () => {
    expect(renderAs("org_admin", []).current).toBe(true);
    expect(renderAs("Org Admin", []).current).toBe(true);
  });

  it("is false for a proxy admin, who is covered by role-based gates instead", () => {
    const result = renderAs("Admin", []);
    expect(result.current).toBe(false);
  });
});
