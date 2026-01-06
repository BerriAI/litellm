import type { RoleMappings as RoleMappingsType } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { renderWithProviders } from "../../../../../tests/test-utils";
import RoleMappings from "./RoleMappings";

describe("RoleMappings", () => {
  it("should render successfully", () => {
    const roleMappings: RoleMappingsType = {
      provider: "generic",
      group_claim: "groups",
      default_role: "internal_user",
      roles: {
        proxy_admin: ["admin-group"],
        proxy_admin_viewer: [],
        internal_user: ["user-group"],
        internal_user_viewer: [],
      },
    };

    renderWithProviders(<RoleMappings roleMappings={roleMappings} />);

    expect(screen.getByText("Role Mappings")).toBeInTheDocument();
  });

  it("should return null when roleMappings is undefined", () => {
    const { container } = renderWithProviders(<RoleMappings roleMappings={undefined} />);

    expect(container.firstChild).toBeNull();
  });

  it("should display Group Claim and Default Role with correct values and display names", () => {
    const testCases: Array<{ role: RoleMappingsType["default_role"]; displayName: string; groupClaim: string }> = [
      { role: "internal_user_viewer", displayName: "Internal Viewer", groupClaim: "custom-groups-1" },
      { role: "internal_user", displayName: "Internal User", groupClaim: "custom-groups-2" },
      { role: "proxy_admin_viewer", displayName: "Proxy Admin Viewer", groupClaim: "custom-groups-3" },
      { role: "proxy_admin", displayName: "Proxy Admin", groupClaim: "custom-groups-4" },
    ];

    testCases.forEach(({ role, displayName, groupClaim }) => {
      const roleMappings: RoleMappingsType = {
        provider: "generic",
        group_claim: groupClaim,
        default_role: role,
        roles: {
          proxy_admin: [],
          proxy_admin_viewer: [],
          internal_user: [],
          internal_user_viewer: [],
        },
      };

      const { unmount } = renderWithProviders(<RoleMappings roleMappings={roleMappings} />);

      expect(screen.getByText("Group Claim")).toBeInTheDocument();
      expect(screen.getByText(groupClaim)).toBeInTheDocument();
      expect(screen.getByText("Default Role")).toBeInTheDocument();
      const displayNameElements = screen.getAllByText(displayName);
      expect(displayNameElements.length).toBeGreaterThan(0);
      unmount();
    });
  });

  it("should display table with roles, groups as Tags when mapped, and 'No groups mapped' when empty", () => {
    const roleMappings: RoleMappingsType = {
      provider: "generic",
      group_claim: "groups",
      default_role: "internal_user",
      roles: {
        proxy_admin: ["admin-group-1", "admin-group-2", "admin-group-3"],
        proxy_admin_viewer: ["viewer-group"],
        internal_user: ["user-group"],
        internal_user_viewer: [],
      },
    };

    renderWithProviders(<RoleMappings roleMappings={roleMappings} />);

    expect(screen.getByText("Role")).toBeInTheDocument();
    expect(screen.getByText("Mapped Groups")).toBeInTheDocument();
    expect(screen.getAllByText("Proxy Admin").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Proxy Admin Viewer").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Internal User").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Internal Viewer").length).toBeGreaterThan(0);
    expect(screen.getByText("admin-group-1")).toBeInTheDocument();
    expect(screen.getByText("admin-group-2")).toBeInTheDocument();
    expect(screen.getByText("admin-group-3")).toBeInTheDocument();
    expect(screen.getByText("viewer-group")).toBeInTheDocument();
    expect(screen.getByText("user-group")).toBeInTheDocument();
    expect(screen.getByText("No groups mapped")).toBeInTheDocument();
  });
});
