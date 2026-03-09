import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/leftnav", () => ({
  menuGroups: [
    {
      groupLabel: "AI GATEWAY",
      items: [
        { key: "api-keys", page: "api-keys", label: "Virtual Keys" },
        { key: "teams", page: "teams", label: "Teams" },
        { key: "agents", page: "agents", label: "Agents" },
        {
          key: "tools",
          page: "tools",
          label: "Tools",
          children: [{ key: "vector-stores", page: "vector-stores", label: "Vector Stores" }],
        },
      ],
    },
  ],
}));

import { isPageAccessibleForUser } from "./page_access";

describe("page access rules", () => {
  it("blocks internal user when page not in enabled list", () => {
    const canAccessTeams = isPageAccessibleForUser("teams", {
      userRole: "Internal User",
      enabledPagesInternalUsers: ["api-keys"],
    });

    expect(canAccessTeams).toBe(false);
  });

  it("allows admin even when internal enabled list excludes the page", () => {
    const canAccessTeams = isPageAccessibleForUser("teams", {
      userRole: "Admin",
      enabledPagesInternalUsers: ["api-keys"],
    });

    expect(canAccessTeams).toBe(true);
  });

  it("blocks agents page when disabled for internal users", () => {
    const canAccessAgents = isPageAccessibleForUser("agents", {
      userRole: "Internal User",
      enabledPagesInternalUsers: null,
      disableAgentsForInternalUsers: true,
      allowAgentsForTeamAdmins: false,
    });

    expect(canAccessAgents).toBe(false);
  });

  it("allows vector stores for team admins when override is enabled", () => {
    const canAccessVectorStores = isPageAccessibleForUser("vector-stores", {
      userRole: "Internal User",
      enabledPagesInternalUsers: null,
      disableVectorStoresForInternalUsers: true,
      allowVectorStoresForTeamAdmins: true,
      userId: "u1",
      teams: [
        {
          members_with_roles: [{ user_id: "u1", role: "admin" }],
        },
      ],
    });

    expect(canAccessVectorStores).toBe(true);
  });
});
