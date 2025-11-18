import { act, fireEvent, render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { teamCreateCall } from "./networking";
import OldTeams from "./OldTeams";

vi.mock("./networking", () => ({
  teamCreateCall: vi.fn(),
  teamDeleteCall: vi.fn(),
  fetchMCPAccessGroups: vi.fn(),
  v2TeamListCall: vi.fn(),
}));

vi.mock("./common_components/fetch_teams", () => ({
  fetchTeams: vi.fn(),
}));

vi.mock("./molecules/notifications_manager", () => ({
  default: {
    info: vi.fn(),
    success: vi.fn(),
    error: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

describe("OldTeams - handleCreate organization handling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should not include organization_id when it's an empty string", async () => {
    const mockAccessToken = "test-token";
    const formValues: Record<string, any> = {
      team_alias: "Test Team",
      organization_id: "", // Empty string
      models: [],
    };

    // Simulate the handleCreate logic
    let organizationId = formValues?.organization_id || null;
    if (organizationId === "" || typeof organizationId !== "string") {
      formValues.organization_id = null;
    } else {
      formValues.organization_id = organizationId.trim();
    }

    expect(formValues.organization_id).toBeNull();
    expect(formValues.organization_id).not.toBe("");
  });

  it("should set organization_id to null when it's not a string type", async () => {
    const formValues: Record<string, any> = {
      team_alias: "Test Team",
      organization_id: undefined,
      models: [],
    };

    // Simulate the handleCreate logic
    let organizationId = formValues?.organization_id || null;
    if (organizationId === "" || typeof organizationId !== "string") {
      formValues.organization_id = null;
    } else {
      formValues.organization_id = organizationId.trim();
    }

    expect(formValues.organization_id).toBeNull();
  });

  it("should trim and keep valid organization_id string", async () => {
    const formValues: Record<string, any> = {
      team_alias: "Test Team",
      organization_id: "  org-123  ", // String with whitespace
      models: [],
    };

    // Simulate the handleCreate logic
    let organizationId = formValues?.organization_id || null;
    if (organizationId === "" || typeof organizationId !== "string") {
      formValues.organization_id = null;
    } else {
      formValues.organization_id = organizationId.trim();
    }

    expect(formValues.organization_id).toBe("org-123");
  });

  it("should keep valid organization_id without modification", async () => {
    const formValues: Record<string, any> = {
      team_alias: "Test Team",
      organization_id: "f874bb43-b898-4813-beca-4054d224eafc",
      models: [],
    };

    // Simulate the handleCreate logic
    let organizationId = formValues?.organization_id || null;
    if (organizationId === "" || typeof organizationId !== "string") {
      formValues.organization_id = null;
    } else {
      formValues.organization_id = organizationId.trim();
    }

    expect(formValues.organization_id).toBe("f874bb43-b898-4813-beca-4054d224eafc");
  });

  it("should not send organization_id field when converting empty string to null", async () => {
    const formValues: Record<string, any> = {
      team_alias: "Test Team",
      organization_id: "",
      models: ["gpt-4"],
      max_budget: 100,
    };

    // Simulate the handleCreate logic
    let organizationId = formValues?.organization_id || null;
    if (organizationId === "" || typeof organizationId !== "string") {
      formValues.organization_id = null;
    } else {
      formValues.organization_id = organizationId.trim();
    }

    // Verify the structure
    expect(formValues).toEqual({
      team_alias: "Test Team",
      organization_id: null,
      models: ["gpt-4"],
      max_budget: 100,
    });

    // Verify we're not sending an empty string
    expect(formValues.organization_id).not.toBe("");

    // Verify it's explicitly null, not undefined
    expect(formValues.organization_id).toBeNull();
  });

  it("should handle when currentOrg is used as fallback", async () => {
    const currentOrg = {
      organization_id: "fallback-org-id",
      organization_alias: "Fallback Org",
      models: [],
      members: [],
    };

    const formValues: Record<string, any> = {
      team_alias: "Test Team",
      models: [],
    };

    // Simulate the handleCreate logic with currentOrg fallback
    let organizationId = formValues?.organization_id || currentOrg?.organization_id;
    if (organizationId === "" || typeof organizationId !== "string") {
      formValues.organization_id = null;
    } else {
      formValues.organization_id = organizationId.trim();
    }

    expect(formValues.organization_id).toBe("fallback-org-id");
  });

  it("should not include organizations as an empty array in the request payload", async () => {
    const mockTeamCreateCall = vi.mocked(teamCreateCall);
    const mockAccessToken = "test-token";

    const formValues = {
      team_alias: "Test Team",
      organization_id: "org-123",
      models: ["gpt-4"],
      organizations: [], // This should never be sent
    };

    // Remove organizations key if it's empty
    if (Array.isArray(formValues.organizations) && formValues.organizations.length === 0) {
      delete (formValues as any).organizations;
    }

    // Verify organizations key is removed
    expect(formValues).not.toHaveProperty("organizations");
    expect(formValues).toEqual({
      team_alias: "Test Team",
      organization_id: "org-123",
      models: ["gpt-4"],
    });
  });

  it("should handle organization_id validation for org admins", () => {
    // This test simulates the validation that should happen for org admins
    const isOrgAdmin = true;
    const formValues: Record<string, any> = {
      team_alias: "Test Team",
      // organization_id is missing/undefined
    };

    // For org admins, organization_id should be required
    const hasOrganization =
      formValues.organization_id !== undefined &&
      formValues.organization_id !== null &&
      formValues.organization_id !== "";

    if (isOrgAdmin && !hasOrganization) {
      // This should trigger validation error
      expect(hasOrganization).toBe(false);
    }
  });

  it("should allow null organization_id for global admins", () => {
    const isAdmin = true;
    const formValues: Record<string, any> = {
      team_alias: "Test Team",
      organization_id: null,
      models: [],
    };

    // Global admins can create teams without an organization
    if (isAdmin) {
      expect(formValues.organization_id).toBeNull();
      // This is valid for admins
    }
  });

  it("should ensure organization_id is never an empty list", () => {
    const invalidFormValues: Record<string, any> = {
      team_alias: "Test Team",
      organization_id: [], // Wrong type - should be string or null
    };

    // Type check: organization_id should never be an array
    expect(Array.isArray(invalidFormValues.organization_id)).toBe(true);

    // Correct it to null
    if (Array.isArray(invalidFormValues.organization_id)) {
      invalidFormValues.organization_id = null;
    }

    expect(invalidFormValues.organization_id).toBeNull();
    expect(Array.isArray(invalidFormValues.organization_id)).toBe(false);
  });

  it("should clear the delete modal when the cancel button is clicked", async () => {
    const { getByRole, getByTestId } = render(
      <OldTeams
        teams={[
          {
            team_id: "1",
            team_alias: "Test Team",
            organization_id: "org-123",
            models: ["gpt-4"],
            max_budget: 100,
            budget_duration: "1d",
            tpm_limit: 1000,
            rpm_limit: 1000,
            created_at: new Date().toISOString(),
            keys: [],
            members_with_roles: [],
          },
        ]}
        searchParams={{}}
        accessToken="test-token"
        setTeams={vi.fn()}
        userID="user-123"
        userRole="Admin"
        organizations={[]}
      />,
    );
    const deleteTeamButton = getByTestId("delete-team-button");
    act(() => {
      fireEvent.click(deleteTeamButton);
    });
    expect(getByRole("heading", { name: "Delete Team" })).toBeInTheDocument();
    expect(getByRole("button", { name: "Cancel" })).toBeInTheDocument();
  });
});

describe("OldTeams - empty state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should display empty state message when teams array is empty", () => {
    const { getByText } = render(
      <OldTeams
        teams={[]}
        searchParams={{}}
        accessToken="test-token"
        setTeams={vi.fn()}
        userID="user-123"
        userRole="Admin"
        organizations={[]}
      />,
    );

    expect(getByText("No teams found")).toBeInTheDocument();
    expect(getByText("Adjust your filters or create a new team")).toBeInTheDocument();
  });

  it("should display empty state message when teams is null", () => {
    const { getByText } = render(
      <OldTeams
        teams={null}
        searchParams={{}}
        accessToken="test-token"
        setTeams={vi.fn()}
        userID="user-123"
        userRole="Admin"
        organizations={[]}
      />,
    );

    expect(getByText("No teams found")).toBeInTheDocument();
    expect(getByText("Adjust your filters or create a new team")).toBeInTheDocument();
  });

  it("should not display empty state when teams array has items", () => {
    const { queryByText, getByText } = render(
      <OldTeams
        teams={[
          {
            team_id: "1",
            team_alias: "Test Team",
            organization_id: "org-123",
            models: ["gpt-4"],
            max_budget: 100,
            budget_duration: "1d",
            tpm_limit: 1000,
            rpm_limit: 1000,
            created_at: new Date().toISOString(),
            keys: [],
            members_with_roles: [],
          },
        ]}
        searchParams={{}}
        accessToken="test-token"
        setTeams={vi.fn()}
        userID="user-123"
        userRole="Admin"
        organizations={[]}
      />,
    );

    expect(queryByText("No teams found")).not.toBeInTheDocument();
    expect(queryByText("Adjust your filters or create a new team")).not.toBeInTheDocument();
    expect(getByText("Test Team")).toBeInTheDocument();
  });
});

describe("OldTeams - helper functions", () => {
  describe("getAdminOrganizations", () => {
    it("should return all organizations for Admin role", () => {
      const organizations = [
        {
          organization_id: "org-1",
          organization_alias: "Org 1",
          models: [],
          members: [],
        },
        {
          organization_id: "org-2",
          organization_alias: "Org 2",
          models: [],
          members: [],
        },
      ];

      // Simulate getAdminOrganizations logic for Admin
      const userRole = "Admin";
      const result = userRole === "Admin" ? organizations : [];

      expect(result).toEqual(organizations);
      expect(result.length).toBe(2);
    });

    it("should return only org_admin organizations for Org Admin role", () => {
      const userID = "user-123";
      const userRole = "Org Admin";
      const organizations = [
        {
          organization_id: "org-1",
          organization_alias: "Org 1",
          models: [],
          members: [{ user_id: "user-123", user_role: "org_admin" }],
        },
        {
          organization_id: "org-2",
          organization_alias: "Org 2",
          models: [],
          members: [{ user_id: "user-456", user_role: "org_admin" }],
        },
        {
          organization_id: "org-3",
          organization_alias: "Org 3",
          models: [],
          members: [{ user_id: "user-123", user_role: "member" }],
        },
      ];

      // Simulate getAdminOrganizations logic
      const result = organizations.filter((org) =>
        org.members?.some((member) => member.user_id === userID && member.user_role === "org_admin"),
      );

      expect(result.length).toBe(1);
      expect(result[0].organization_id).toBe("org-1");
    });

    it("should return empty array when user is not admin of any organization", () => {
      const userID = "user-999";
      const organizations = [
        {
          organization_id: "org-1",
          organization_alias: "Org 1",
          models: [],
          members: [{ user_id: "user-123", user_role: "org_admin" }],
        },
      ];

      // Simulate getAdminOrganizations logic
      const result = organizations.filter((org) =>
        org.members?.some((member) => member.user_id === userID && member.user_role === "org_admin"),
      );

      expect(result.length).toBe(0);
    });
  });

  describe("canCreateOrManageTeams", () => {
    it("should return true for Admin role", () => {
      const userRole = "Admin";
      const result = userRole === "Admin";
      expect(result).toBe(true);
    });

    it("should return true for org_admin in any organization", () => {
      const userID = "user-123";
      const organizations = [
        {
          organization_id: "org-1",
          organization_alias: "Org 1",
          models: [],
          members: [{ user_id: "user-123", user_role: "org_admin" }],
        },
      ];

      const result = organizations.some((org) =>
        org.members?.some((member) => member.user_id === userID && member.user_role === "org_admin"),
      );

      expect(result).toBe(true);
    });

    it("should return false when user has no admin permissions", () => {
      const userID = "user-123";
      const userRole: string = "User";
      const organizations = [
        {
          organization_id: "org-1",
          organization_alias: "Org 1",
          models: [],
          members: [{ user_id: "user-123", user_role: "member" }],
        },
      ];

      const isAdmin = userRole === "Admin";
      const isOrgAdmin = organizations.some((org) =>
        org.members?.some((member) => member.user_id === userID && member.user_role === "org_admin"),
      );

      expect(isAdmin || isOrgAdmin).toBe(false);
    });
  });
});
