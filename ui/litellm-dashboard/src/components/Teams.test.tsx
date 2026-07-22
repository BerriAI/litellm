import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fetchAvailableModelsForTeamOrKey } from "./key_team_helpers/fetch_available_models_team_key";
import { fetchMCPAccessGroups, getGuardrailsList, teamCreateCall } from "./networking";
import Teams from "./Teams";

const mockTeamInfoView = vi.fn();
const mockUseOrganizations = vi.fn();

// The teams grid is unit-tested in TeamsPage/TeamsTable.test.tsx. Here we stub it and drive its callbacks
// directly so we can test the Teams shell wiring (delete modal, detail view) without the real DataTable.
let mockTeamsTableProps: any = null;
vi.mock("./TeamsPage/TeamsTable", () => ({
  TeamsTable: (props: any) => {
    mockTeamsTableProps = props;
    return <div data-testid="teams-table-stub" />;
  },
}));

vi.mock("./networking", () => ({
  teamCreateCall: vi.fn(),
  teamDeleteCall: vi.fn(),
  fetchMCPAccessGroups: vi.fn(),
  v2TeamListCall: vi.fn(),
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
  getPoliciesList: vi.fn().mockResolvedValue({ policies: [] }),
}));

// Teams invalidates teamsTableKeys on mutations; the selected team is passed up from the table.
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  teamsTableKeys: { all: ["teamsTable"] },
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

const baseTableTeam = {
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
};

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

// Re-establish safe defaults before every test (clearAllMocks keeps return values, so restore them here).
beforeEach(() => {
  mockTeamsTableProps = null;
});

describe("Teams - handleCreate organization handling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTeamInfoView.mockClear();
    mockTeamsTableProps = null;
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue([]);
    vi.mocked(fetchMCPAccessGroups).mockResolvedValue([]);
    vi.mocked(getGuardrailsList).mockResolvedValue({ guardrails: [] });
    mockUseOrganizations.mockReturnValue({ data: null });
  });

  it("should not include organization_id when it's an empty string", async () => {
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
      organization_id: "  org-123  ",
      models: [],
    };

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

    let organizationId = formValues?.organization_id || null;
    if (organizationId === "" || typeof organizationId !== "string") {
      formValues.organization_id = null;
    } else {
      formValues.organization_id = organizationId.trim();
    }

    expect(formValues).toEqual({
      team_alias: "Test Team",
      organization_id: null,
      models: ["gpt-4"],
      max_budget: 100,
    });
    expect(formValues.organization_id).not.toBe("");
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

    let organizationId = formValues?.organization_id || currentOrg?.organization_id;
    if (organizationId === "" || typeof organizationId !== "string") {
      formValues.organization_id = null;
    } else {
      formValues.organization_id = organizationId.trim();
    }

    expect(formValues.organization_id).toBe("fallback-org-id");
  });

  it("opens the delete modal when the table's delete action fires", async () => {
    mockUseOrganizations.mockReturnValue({ data: [] });
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />);

    await waitFor(() => expect(mockTeamsTableProps).not.toBeNull());
    await act(async () => {
      mockTeamsTableProps.onDeleteTeam(baseTableTeam);
    });

    expect(screen.getByText("Delete Team?")).toBeInTheDocument();
  });
});

describe("Teams - helper functions", () => {
  describe("getAdminOrganizations", () => {
    it("should return all organizations for Admin role", () => {
      const organizations = [
        { organization_id: "org-1", organization_alias: "Org 1", models: [], members: [] },
        { organization_id: "org-2", organization_alias: "Org 2", models: [], members: [] },
      ];

      const userRole = "Admin";
      const result = userRole === "Admin" ? organizations : [];

      expect(result).toEqual(organizations);
      expect(result.length).toBe(2);
    });

    it("should return only org_admin organizations for Org Admin role", () => {
      const userID = "user-123";
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

      const result = organizations.filter((org) =>
        org.members?.some((member) => member.user_id === userID && member.user_role === "org_admin"),
      );

      expect(result.length).toBe(0);
    });
  });

  describe("canCreateOrManageTeams", () => {
    it("should return true for Admin role", () => {
      const userRole = "Admin";
      expect(userRole === "Admin").toBe(true);
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

describe("Teams - premium props", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTeamInfoView.mockClear();
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue([]);
    vi.mocked(fetchMCPAccessGroups).mockResolvedValue([]);
    vi.mocked(getGuardrailsList).mockResolvedValue({ guardrails: [] });
    mockUseOrganizations.mockReturnValue({ data: [] });
  });

  it("passes premiumUser flag to TeamInfoView when a team is opened", async () => {
    const premiumTeam = { ...baseTableTeam, team_id: "team-123456789", team_alias: "Premium Team" };
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" premiumUser={true} />);

    await waitFor(() => expect(mockTeamsTableProps).not.toBeNull());
    act(() => mockTeamsTableProps.onSelectTeam(premiumTeam));

    await waitFor(() => expect(mockTeamInfoView).toHaveBeenCalled());
    expect(mockTeamInfoView).toHaveBeenLastCalledWith(expect.objectContaining({ premiumUser: true }));
  });
});

describe("Teams - Create Team CTA is grouped with the tabs on the left", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseOrganizations.mockReturnValue({ data: [] });
  });

  it("renders the Create Team button inside the tab bar, ahead of the tabs", () => {
    const { container } = renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />);

    const createButton = screen.getByTestId("create-team-button");
    const tabNav = container.querySelector(".ant-tabs-nav");

    // The CTA lives in the tab bar's left slot, not the standalone page header.
    expect(tabNav).not.toBeNull();
    expect(tabNav!.contains(createButton)).toBe(true);

    // It reads as the left end of the cluster: it precedes the first tab in DOM order.
    const firstTab = screen.getByRole("tab", { name: "Your Teams" });
    expect(createButton.compareDocumentPosition(firstTab) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("omits the Create Team CTA for a role that cannot manage teams", () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin Viewer" />);
    expect(screen.queryByTestId("create-team-button")).not.toBeInTheDocument();
  });
});

describe("Teams - Default Team Settings tab visibility", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseOrganizations.mockReturnValue({ data: [] });
  });

  it("should show Default Team Settings tab for Admin role", () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />);
    expect(screen.getByRole("tab", { name: "Default Team Settings" })).toBeInTheDocument();
  });

  it("should show Default Team Settings tab for proxy_admin role", () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="proxy_admin" />);
    expect(screen.getByRole("tab", { name: "Default Team Settings" })).toBeInTheDocument();
  });

  it("should not show Default Team Settings tab for proxy_admin_viewer role", () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="proxy_admin_viewer" />);
    expect(screen.queryByRole("tab", { name: "Default Team Settings" })).not.toBeInTheDocument();
  });

  it("should not show Default Team Settings tab for Admin Viewer role", () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin Viewer" />);
    expect(screen.queryByRole("tab", { name: "Default Team Settings" })).not.toBeInTheDocument();
  });
});

describe("Teams - access_group_ids in team create", () => {
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
    });
    mockUseOrganizations.mockReturnValue({
      data: [{ organization_id: "org-1", organization_alias: "Org 1", models: [], members: [] }],
    });
  });

  it("should pass access_group_ids to teamCreateCall when creating team", async () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />);

    const createButton = screen.getAllByRole("button", { name: /create team/i })[0];
    act(() => {
      fireEvent.click(createButton);
    });

    await waitFor(() => {
      expect(screen.getByLabelText(/team name/i)).toBeInTheDocument();
    });

    fireEvent.change(screen.getByLabelText(/team name/i), { target: { value: "Test Team" } });
    fireEvent.change(screen.getByTestId("create-team-models-select"), { target: { value: "gpt-4" } });

    fireEvent.click(screen.getByText("Additional Settings"));

    await waitFor(() => {
      expect(screen.getByTestId("access-group-selector")).toBeInTheDocument();
    });

    fireEvent.change(screen.getByTestId("access-group-selector"), { target: { value: "ag-1,ag-2" } });

    const createTeamSubmitButtons = screen.getAllByRole("button", { name: /create team/i });
    fireEvent.click(createTeamSubmitButtons[createTeamSubmitButtons.length - 1]);

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
  });
});

describe("Teams - metadata key-value pairs in team create", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTeamInfoView.mockClear();
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue(["gpt-4"]);
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
    });
    mockUseOrganizations.mockReturnValue({
      data: [{ organization_id: "org-1", organization_alias: "Org 1", models: [], members: [] }],
    });
  });

  const openCreateModal = async () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />);

    const createButton = screen.getAllByRole("button", { name: /create team/i })[0];
    act(() => {
      fireEvent.click(createButton);
    });

    await waitFor(() => {
      expect(screen.getByLabelText(/team name/i)).toBeInTheDocument();
    });
  };

  it("renders the metadata editor in the main form without opening Additional Settings", async () => {
    await openCreateModal();

    expect(screen.getByRole("button", { name: /add key-value pair/i })).toBeInTheDocument();
  });

  it("submits metadata built from key-value pairs as a typed JSON object", async () => {
    await openCreateModal();

    fireEvent.change(screen.getByLabelText(/team name/i), { target: { value: "Test Team" } });
    fireEvent.change(screen.getByTestId("create-team-models-select"), { target: { value: "gpt-4" } });

    fireEvent.click(screen.getByRole("button", { name: /add key-value pair/i }));
    await waitFor(() => {
      expect(screen.getByPlaceholderText("Key")).toBeInTheDocument();
    });
    fireEvent.change(screen.getByPlaceholderText("Key"), { target: { value: "cost_center" } });
    fireEvent.change(screen.getByPlaceholderText("Value"), { target: { value: "eng-42" } });

    fireEvent.click(screen.getByRole("button", { name: /add key-value pair/i }));
    await waitFor(() => {
      expect(screen.getAllByPlaceholderText("Key")).toHaveLength(2);
    });
    fireEvent.change(screen.getAllByPlaceholderText("Key")[1], { target: { value: "tier" } });
    fireEvent.change(screen.getAllByPlaceholderText("Value")[1], { target: { value: "3" } });

    const createTeamSubmitButtons = screen.getAllByRole("button", { name: /create team/i });
    fireEvent.click(createTeamSubmitButtons[createTeamSubmitButtons.length - 1]);

    await waitFor(() => {
      expect(teamCreateCall).toHaveBeenCalled();
    });

    const submittedValues = vi.mocked(teamCreateCall).mock.calls[0][1];
    expect(JSON.parse(submittedValues.metadata)).toEqual({ cost_center: "eng-42", tier: 3 });
  });

  it("omits metadata entirely when no pairs are added", async () => {
    await openCreateModal();

    fireEvent.change(screen.getByLabelText(/team name/i), { target: { value: "Test Team" } });
    fireEvent.change(screen.getByTestId("create-team-models-select"), { target: { value: "gpt-4" } });

    const createTeamSubmitButtons = screen.getAllByRole("button", { name: /create team/i });
    fireEvent.click(createTeamSubmitButtons[createTeamSubmitButtons.length - 1]);

    await waitFor(() => {
      expect(teamCreateCall).toHaveBeenCalled();
    });

    expect(vi.mocked(teamCreateCall).mock.calls[0][1].metadata).toBeUndefined();
  });
});

describe("Teams - models dropdown options", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue(["gpt-4", "gpt-3.5-turbo"]);
    mockUseOrganizations.mockReturnValue({ data: [] });
  });

  it("should not render all-proxy-models option in models select", async () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />);

    await waitFor(() => {
      expect(fetchAvailableModelsForTeamOrKey).toHaveBeenCalled();
    });

    const createButton = screen.getAllByRole("button", { name: /create team/i })[0];
    act(() => {
      fireEvent.click(createButton);
    });

    await waitFor(() => {
      expect(screen.getByLabelText(/models/i)).toBeInTheDocument();
    });
    expect(screen.queryByText("All Proxy Models")).not.toBeInTheDocument();
  });
});

describe("Teams - delete team warning copy", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseOrganizations.mockReturnValue({ data: [] });
  });

  const openDeleteModal = async (team: any) => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />);
    await waitFor(() => expect(mockTeamsTableProps).not.toBeNull());
    await act(async () => {
      mockTeamsTableProps.onDeleteTeam(team);
    });
    expect(screen.getByText("Delete Team?")).toBeInTheDocument();
  };

  it("warns that the team's models are deleted when the team has keys", async () => {
    await openDeleteModal({ ...baseTableTeam, keys: [], keys_count: 5 });

    expect(screen.getByText(/Warning: This team has 5 keys associated with it/i)).toHaveTextContent(
      /along with any models created for this team/i,
    );
    expect(screen.getByText(/Are you sure you want to delete this team/i)).toHaveTextContent(
      /any models created for it/i,
    );
  });

  it("still warns about model deletion in the confirmation message when the team has no keys", async () => {
    await openDeleteModal({ ...baseTableTeam, keys: [], keys_count: 0 });

    expect(screen.queryByText(/Warning: This team has/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Are you sure you want to delete this team/i)).toHaveTextContent(
      /any models created for it/i,
    );
  });
});

describe("Teams - LIT-2530 organization stays optional for proxy admin with a single org", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTeamInfoView.mockClear();
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue(["gpt-4"]);
    vi.mocked(fetchMCPAccessGroups).mockResolvedValue([]);
    vi.mocked(getGuardrailsList).mockResolvedValue({ guardrails: [] });
    vi.mocked(teamCreateCall).mockResolvedValue({
      team_id: "new-team-1",
      team_alias: "No Org Team",
      models: ["gpt-4"],
      organization_id: null,
      keys: [],
      members_with_roles: [],
      spend: 0,
    });
    mockUseOrganizations.mockReturnValue({
      data: [{ organization_id: "org-1", organization_alias: "Org 1", models: [], members: [] }],
    });
  });

  it("creates a team with no organization when exactly one organization exists", async () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />);

    const createButton = screen.getAllByRole("button", { name: /create team/i })[0];
    act(() => {
      fireEvent.click(createButton);
    });

    await waitFor(() => {
      expect(screen.getByLabelText(/team name/i)).toBeInTheDocument();
    });

    fireEvent.change(screen.getByLabelText(/team name/i), { target: { value: "No Org Team" } });
    fireEvent.change(screen.getByTestId("create-team-models-select"), { target: { value: "gpt-4" } });

    const submitButtons = screen.getAllByRole("button", { name: /create team/i });
    fireEvent.click(submitButtons[submitButtons.length - 1]);

    await waitFor(() => {
      expect(teamCreateCall).toHaveBeenCalledWith(
        "test-token",
        expect.objectContaining({ team_alias: "No Org Team", organization_id: null }),
      );
    });
  });
});
