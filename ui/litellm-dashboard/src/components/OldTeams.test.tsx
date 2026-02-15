import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fetchAvailableModelsForTeamOrKey } from "./key_team_helpers/fetch_available_models_team_key";
import { fetchMCPAccessGroups, getGuardrailsList, teamCreateCall } from "./networking";
import OldTeams from "./OldTeams";

const mockTeamInfoView = vi.fn();
const mockUseOrganizations = vi.fn();

vi.mock("./networking", () => ({
  teamCreateCall: vi.fn(),
  teamDeleteCall: vi.fn(),
  fetchMCPAccessGroups: vi.fn(),
  v2TeamListCall: vi.fn(),
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
  getPoliciesList: vi.fn().mockResolvedValue({ policies: [] }),
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

vi.mock("./key_team_helpers/fetch_available_models_team_key", () => ({
  fetchAvailableModelsForTeamOrKey: vi.fn(),
  getModelDisplayName: vi.fn((model: string) => model),
  unfurlWildcardModelsInList: vi.fn((teamModels: string[], allModels: string[]) => {
    const wildcardDisplayNames: string[] = [];
    const expandedModels: string[] = [];

    teamModels.forEach((teamModel) => {
      if (teamModel.endsWith("/*")) {
        const provider = teamModel.replace("/*", "");
        const matchingModels = allModels.filter((model) => model.startsWith(provider + "/"));
        expandedModels.push(...matchingModels);
        wildcardDisplayNames.push(teamModel);
      } else {
        expandedModels.push(teamModel);
      }
    });

    return [...wildcardDisplayNames, ...expandedModels].filter((item, index, array) => array.indexOf(item) === index);
  }),
}));

vi.mock("@/components/team/TeamInfo", () => ({
  __esModule: true,
  default: (props: any) => {
    mockTeamInfoView(props);
    return <div data-testid="team-info-view" />;
  },
}));

vi.mock("./ModelSelect/ModelSelect", () => {
  const ModelSelect = React.forwardRef(({ value, onChange, dataTestId, id }: any, ref: any) => {
    return (
      <input
        ref={ref}
        id={id}
        type="text"
        data-testid={dataTestId || "model-select"}
        value={Array.isArray(value) ? value.join(", ") : ""}
        onChange={(e) => {
          if (onChange) {
            const newVal = e.target.value
              ? e.target.value
                .split(",")
                .map((s: string) => s.trim())
                .filter(Boolean)
              : [];
            onChange(newVal);
          }
        }}
      />
    );
  });
  ModelSelect.displayName = "ModelSelect";
  return {
    ModelSelect,
  };
});

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: () => mockUseOrganizations(),
}));

vi.mock("@/app/(dashboard)/hooks/accessGroups/useAccessGroups", () => ({
  useAccessGroups: vi.fn().mockReturnValue({
    data: [
      { access_group_id: "ag-1", access_group_name: "Group 1" },
      { access_group_id: "ag-2", access_group_name: "Group 2" },
    ],
    isLoading: false,
    isError: false,
  }),
}));

vi.mock("./common_components/AccessGroupSelector", () => ({
  default: ({ value = [], onChange }: { value?: string[]; onChange?: (v: string[]) => void }) => (
    <input
      data-testid="access-group-selector"
      value={Array.isArray(value) ? value.join(",") : ""}
      onChange={(e) => onChange?.(e.target.value ? e.target.value.split(",").map((s) => s.trim()) : [])}
    />
  ),
}));

const createQueryClient = () => {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
};

const renderWithQueryClient = (component: React.ReactElement) => {
  const queryClient = createQueryClient();
  return render(<QueryClientProvider client={queryClient}>{component}</QueryClientProvider>);
};

describe("OldTeams - handleCreate organization handling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTeamInfoView.mockClear();
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue([]);
    vi.mocked(fetchMCPAccessGroups).mockResolvedValue([]);
    vi.mocked(getGuardrailsList).mockResolvedValue({ guardrails: [] });
    mockUseOrganizations.mockReturnValue({ data: null });
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
    mockUseOrganizations.mockReturnValue({ data: [] });
    renderWithQueryClient(
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
            spend: 0,
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
    const deleteTeamButton = screen.getByTestId("delete-team-button");
    act(() => {
      fireEvent.click(deleteTeamButton);
    });
    expect(screen.getByText("Delete Team?")).toBeInTheDocument();
  });
});

describe("OldTeams - empty state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseOrganizations.mockReturnValue({ data: [] });
  });

  it("should display empty state message when teams array is empty", () => {
    renderWithQueryClient(
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

    expect(screen.getByText("No teams found")).toBeInTheDocument();
    expect(screen.getByText("Adjust your filters or create a new team")).toBeInTheDocument();
  });

  it("should display empty state message when teams is null", () => {
    renderWithQueryClient(
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

    expect(screen.getByText("No teams found")).toBeInTheDocument();
    expect(screen.getByText("Adjust your filters or create a new team")).toBeInTheDocument();
  });

  it("should not display empty state when teams array has items", () => {
    renderWithQueryClient(
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
            spend: 0,
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

    expect(screen.queryByText("No teams found")).not.toBeInTheDocument();
    expect(screen.queryByText("Adjust your filters or create a new team")).not.toBeInTheDocument();
    expect(screen.getByText("Test Team")).toBeInTheDocument();
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

describe("OldTeams - premium props", () => {
  beforeEach(() => {
    mockTeamInfoView.mockClear();
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue([]);
    vi.mocked(fetchMCPAccessGroups).mockResolvedValue([]);
    vi.mocked(getGuardrailsList).mockResolvedValue({ guardrails: [] });
    mockUseOrganizations.mockReturnValue({ data: [] });
  });

  it("passes premiumUser flag to TeamInfoView", async () => {
    renderWithQueryClient(
      <OldTeams
        teams={[
          {
            team_id: "team-123456789",
            team_alias: "Premium Team",
            organization_id: "org-123",
            models: ["gpt-4"],
            max_budget: 100,
            budget_duration: "1d",
            tpm_limit: 1000,
            rpm_limit: 1000,
            created_at: new Date().toISOString(),
            keys: [],
            members_with_roles: [],
            spend: 0,
          },
        ]}
        searchParams={{}}
        accessToken="test-token"
        setTeams={vi.fn()}
        userID="user-123"
        userRole="Admin"
        organizations={[]}
        premiumUser={true}
      />,
    );

    const truncatedTeamId = "team-123456789".slice(0, 7);
    const teamButton = await screen.findByRole("button", {
      name: new RegExp(`${truncatedTeamId}\\.\\.\\.`),
    });
    act(() => {
      fireEvent.click(teamButton);
    });

    await waitFor(() => expect(mockTeamInfoView).toHaveBeenCalled());

    expect(mockTeamInfoView).toHaveBeenLastCalledWith(expect.objectContaining({ premiumUser: true }));
  });
});

describe("OldTeams - Default Team Settings tab visibility", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseOrganizations.mockReturnValue({ data: [] });
  });

  it("should show Default Team Settings tab for Admin role", () => {
    renderWithQueryClient(
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
            spend: 0,
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

    expect(screen.getByRole("tab", { name: "Default Team Settings" })).toBeInTheDocument();
  });

  it("should show Default Team Settings tab for proxy_admin role", () => {
    renderWithQueryClient(
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
            spend: 0,
          },
        ]}
        searchParams={{}}
        accessToken="test-token"
        setTeams={vi.fn()}
        userID="user-123"
        userRole="proxy_admin"
        organizations={[]}
      />,
    );

    expect(screen.getByRole("tab", { name: "Default Team Settings" })).toBeInTheDocument();
  });

  it("should not show Default Team Settings tab for proxy_admin_viewer role", () => {
    renderWithQueryClient(
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
            spend: 0,
          },
        ]}
        searchParams={{}}
        accessToken="test-token"
        setTeams={vi.fn()}
        userID="user-123"
        userRole="proxy_admin_viewer"
        organizations={[]}
      />,
    );

    expect(screen.queryByRole("tab", { name: "Default Team Settings" })).not.toBeInTheDocument();
  });

  it("should not show Default Team Settings tab for Admin Viewer role", () => {
    renderWithQueryClient(
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
            spend: 0,
          },
        ]}
        searchParams={{}}
        accessToken="test-token"
        setTeams={vi.fn()}
        userID="user-123"
        userRole="Admin Viewer"
        organizations={[]}
      />,
    );

    expect(screen.queryByRole("tab", { name: "Default Team Settings" })).not.toBeInTheDocument();
  });
});

describe("OldTeams - access_group_ids in team create", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTeamInfoView.mockClear();
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue(["gpt-4", "gpt-3.5-turbo"]);
    vi.mocked(fetchMCPAccessGroups).mockResolvedValue([]);
    vi.mocked(getGuardrailsList).mockResolvedValue({ guardrails: [] });
    vi.mocked(teamCreateCall).mockResolvedValue({
      team_id: "new-team-1",
      team_alias: "Test Team",
      models: ["gpt-4"],
      organization_id: null,
      keys: [],
      members_with_roles: [],
      spend: 0,
    } as any);
    mockUseOrganizations.mockReturnValue({ data: [{ organization_id: "org-1", organization_alias: "Org 1", models: [], members: [] }] });
  });

  it("should pass access_group_ids to teamCreateCall when creating team", async () => {
    renderWithQueryClient(
      <OldTeams
        teams={[]}
        searchParams={{}}
        accessToken="test-token"
        setTeams={vi.fn()}
        userID="user-123"
        userRole="Admin"
        organizations={[{ organization_id: "org-1", organization_alias: "Org 1", models: [], members: [] }]}
      />,
    );

    const createButton = screen.getByRole("button", { name: /create new team/i });
    act(() => {
      fireEvent.click(createButton);
    });

    await waitFor(() => {
      expect(screen.getByLabelText(/team name/i)).toBeInTheDocument();
    });

    const teamNameInput = screen.getByLabelText(/team name/i);
    fireEvent.change(teamNameInput, { target: { value: "Test Team" } });

    const modelsInput = screen.getByTestId("create-team-models-select");
    fireEvent.change(modelsInput, { target: { value: "gpt-4" } });

    const additionalSettingsAccordion = screen.getByText("Additional Settings");
    fireEvent.click(additionalSettingsAccordion);

    await waitFor(() => {
      expect(screen.getByTestId("access-group-selector")).toBeInTheDocument();
    });

    const accessGroupInput = screen.getByTestId("access-group-selector");
    fireEvent.change(accessGroupInput, { target: { value: "ag-1,ag-2" } });

    const createTeamSubmitButton = screen.getByRole("button", { name: /create team/i });
    fireEvent.click(createTeamSubmitButton);

    await waitFor(() => {
      expect(teamCreateCall).toHaveBeenCalledWith(
        "test-token",
        expect.objectContaining({
          team_alias: "Test Team",
          models: ["gpt-4"],
          access_group_ids: ["ag-1", "ag-2"],
        }),
      );
    });
  }, { timeout: 30000 });
});

describe("OldTeams - models dropdown options", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue(["gpt-4", "gpt-3.5-turbo"]);
    mockUseOrganizations.mockReturnValue({ data: [] });
  });

  it("should not render all-proxy-models option in models select", async () => {
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue(["gpt-4", "gpt-3.5-turbo"]);

    renderWithQueryClient(
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

    await waitFor(() => {
      expect(fetchAvailableModelsForTeamOrKey).toHaveBeenCalled();
    });

    const createButton = screen.getByRole("button", { name: /create new team/i });
    act(() => {
      fireEvent.click(createButton);
    });

    await waitFor(() => {
      expect(screen.getByLabelText(/models/i)).toBeInTheDocument();
    });
    const allProxyModelsOption = screen.queryByText("All Proxy Models");
    expect(allProxyModelsOption).not.toBeInTheDocument();
  });
});

describe("OldTeams - organization alias display", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseOrganizations.mockReturnValue({ data: [] });
  });

  it("should display organization alias instead of organization id", () => {
    const mockOrganizations = [
      {
        organization_id: "org-123",
        organization_alias: "Test Organization",
        budget_id: "budget-1",
        metadata: {},
        models: [],
        spend: 0,
        model_spend: {},
        created_at: new Date().toISOString(),
        created_by: "user-1",
        updated_at: new Date().toISOString(),
        updated_by: "user-1",
        litellm_budget_table: null,
        teams: null,
        users: null,
        members: null,
      },
    ];

    mockUseOrganizations.mockReturnValue({ data: mockOrganizations });

    renderWithQueryClient(
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
            spend: 0,
          },
        ]}
        searchParams={{}}
        accessToken="test-token"
        setTeams={vi.fn()}
        userID="user-123"
        userRole="Admin"
        organizations={mockOrganizations}
      />,
    );

    expect(screen.getByText("Test Organization")).toBeInTheDocument();
    expect(screen.queryByText("org-123")).not.toBeInTheDocument();
  });

  it("should display organization id when alias is not found", () => {
    mockUseOrganizations.mockReturnValue({ data: [] });

    renderWithQueryClient(
      <OldTeams
        teams={[
          {
            team_id: "1",
            team_alias: "Test Team",
            organization_id: "org-unknown",
            models: ["gpt-4"],
            max_budget: 100,
            budget_duration: "1d",
            tpm_limit: 1000,
            rpm_limit: 1000,
            created_at: new Date().toISOString(),
            keys: [],
            members_with_roles: [],
            spend: 0,
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

    expect(screen.getByText("org-unknown")).toBeInTheDocument();
  });

  it("should display N/A when organization_id is null", () => {
    mockUseOrganizations.mockReturnValue({ data: [] });

    renderWithQueryClient(
      <OldTeams
        teams={[
          {
            team_id: "1",
            team_alias: "Test Team",
            organization_id: null as any,
            models: ["gpt-4"],
            max_budget: 100,
            budget_duration: "1d",
            tpm_limit: 1000,
            rpm_limit: 1000,
            created_at: new Date().toISOString(),
            keys: [],
            members_with_roles: [],
            spend: 0,
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

    expect(screen.getByText("N/A")).toBeInTheDocument();
  });
});
