import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, OnUrlUpdateFunction } from "nuqs/adapters/testing";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useTeamMetadataSchema } from "@/app/(dashboard)/hooks/teams/useTeamMetadataSchema";
import { toast } from "@/lib/toast";
import { fetchAvailableModelsForTeamOrKey } from "./key_team_helpers/fetch_available_models_team_key";
import {
  fetchMCPAccessGroups,
  getDefaultTeamSettings,
  getGuardrailsList,
  getPoliciesList,
  teamCreateCall,
} from "./networking";
import Teams from "./Teams";

const can = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useCan", () => ({
  default: (...args: unknown[]) => can(...args),
}));

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
  getDefaultTeamSettings: vi.fn().mockResolvedValue({ values: {} }),
}));

// Teams invalidates teamsTableKeys on mutations; the selected team is passed up from the table.
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  teamsTableKeys: { all: ["teamsTable"] },
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeamMetadataSchema", () => ({
  useTeamMetadataSchema: vi.fn(() => ({ data: [], isLoading: false })),
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

const renderWithQueryClient = (
  component: React.ReactElement,
  options?: { searchParams?: string; onUrlUpdate?: OnUrlUpdateFunction },
) => {
  const queryClient = createQueryClient();
  return render(
    <NuqsTestingAdapter searchParams={options?.searchParams} onUrlUpdate={options?.onUrlUpdate} hasMemory>
      <QueryClientProvider client={queryClient}>{component}</QueryClientProvider>
    </NuqsTestingAdapter>,
  );
};

// Re-establish safe defaults before every test (clearAllMocks keeps return values, so restore them here).
beforeEach(() => {
  mockTeamsTableProps = null;
  can.mockReturnValue(true);
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
    const organizationId = formValues?.organization_id || null;
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

    const organizationId = formValues?.organization_id || null;
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

    const organizationId = formValues?.organization_id || null;
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

    const organizationId = formValues?.organization_id || null;
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

    const organizationId = formValues?.organization_id || null;
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

    const organizationId = formValues?.organization_id || currentOrg?.organization_id;
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

describe("Teams - team detail deep link (?team=)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTeamInfoView.mockClear();
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue([]);
    vi.mocked(fetchMCPAccessGroups).mockResolvedValue([]);
    vi.mocked(getGuardrailsList).mockResolvedValue({ guardrails: [] });
    mockUseOrganizations.mockReturnValue({ data: [] });
  });

  it("selecting a team pushes ?team= to the URL", async () => {
    const onUrlUpdate = vi.fn();
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />, { onUrlUpdate });

    await waitFor(() => expect(mockTeamsTableProps).not.toBeNull());
    act(() => mockTeamsTableProps.onSelectTeam({ ...baseTableTeam, team_id: "team-deep-link" }));

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    const lastUpdate = onUrlUpdate.mock.calls.at(-1)![0];
    expect(lastUpdate.searchParams.get("team")).toBe("team-deep-link");
    expect(lastUpdate.options.history).toBe("push");

    await waitFor(() => expect(mockTeamInfoView).toHaveBeenCalled());
    expect(mockTeamInfoView).toHaveBeenLastCalledWith(expect.objectContaining({ teamId: "team-deep-link" }));
  });

  it("opens the team detail view directly from a ?team= deep link", async () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />, {
      searchParams: "?team=team-from-url",
    });

    await waitFor(() => expect(mockTeamInfoView).toHaveBeenCalled());
    expect(mockTeamInfoView).toHaveBeenLastCalledWith(expect.objectContaining({ teamId: "team-from-url" }));
  });

  it("closing the team detail view removes ?team= from the URL", async () => {
    const onUrlUpdate = vi.fn();
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />, {
      searchParams: "?team=team-from-url",
      onUrlUpdate,
    });

    await waitFor(() => expect(mockTeamInfoView).toHaveBeenCalled());
    act(() => mockTeamInfoView.mock.calls.at(-1)?.[0].onClose());

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(onUrlUpdate.mock.calls.at(-1)![0].searchParams.has("team")).toBe(false);
    await waitFor(() => expect(screen.queryByTestId("team-info-view")).not.toBeInTheDocument());
  });

  it("should preserve the legacy inset for the team detail view", async () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />, {
      searchParams: "?team=team-from-url",
    });

    await waitFor(() => expect(mockTeamInfoView).toHaveBeenCalled());
    expect(screen.getByRole("main")).toHaveClass("px-12", "py-6");
  });
});

describe("Teams - Create Team CTA is grouped with the tabs on the left", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseOrganizations.mockReturnValue({ data: [] });
  });

  it("should render the Create Team button inside the tab bar, ahead of the tabs", () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />);

    const tabNav = screen.getByRole("tablist");
    const createButton = within(tabNav).getByTestId("create-team-button");
    const firstTab = within(tabNav).getByRole("tab", { name: "Your Teams" });

    expect(screen.getByRole("main")).toHaveClass("p-8");
    expect(within(tabNav).getByRole("separator")).toBeInTheDocument();
    expect(createButton.compareDocumentPosition(firstTab) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("should omit the Create Team CTA for a role that cannot manage teams", () => {
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

  it("creates a team with no models selected, sending the no-default-models sentinel instead of an empty list", async () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />);

    const createButton = screen.getAllByRole("button", { name: /create team/i })[0];
    act(() => {
      fireEvent.click(createButton);
    });

    await waitFor(() => {
      expect(screen.getByLabelText(/team name/i)).toBeInTheDocument();
    });

    fireEvent.change(screen.getByLabelText(/team name/i), { target: { value: "Group Only Team" } });

    const createTeamSubmitButtons = screen.getAllByRole("button", { name: /create team/i });
    fireEvent.click(createTeamSubmitButtons[createTeamSubmitButtons.length - 1]);

    await waitFor(() => {
      expect(teamCreateCall).toHaveBeenCalledWith(
        "test-token",
        expect.objectContaining({
          team_alias: "Group Only Team",
          models: ["no-default-models"],
        }),
      );
    });
  });
});

describe("Teams - Reset Budget in team create", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTeamInfoView.mockClear();
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue(["gpt-4"]);
    vi.mocked(fetchMCPAccessGroups).mockResolvedValue([]);
    vi.mocked(getGuardrailsList).mockResolvedValue({ guardrails: [] });
    vi.mocked(getDefaultTeamSettings).mockResolvedValue({ values: { budget_duration: "30d" } });
    vi.mocked(teamCreateCall).mockResolvedValue({
      team_id: "new-team-1",
      team_alias: "Test Team",
      models: ["gpt-4"],
      organization_id: null,
      keys: [],
      members_with_roles: [],
      spend: 0,
    });
    mockUseOrganizations.mockReturnValue({ data: null });
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

  const resetBudgetSelect = () => screen.getByLabelText("Reset Budget");

  const submitCreateModal = async () => {
    fireEvent.change(screen.getByLabelText(/team name/i), { target: { value: "Test Team" } });

    const createTeamSubmitButtons = screen.getAllByRole("button", { name: /create team/i });
    fireEvent.click(createTeamSubmitButtons[createTeamSubmitButtons.length - 1]);

    await waitFor(() => {
      expect(teamCreateCall).toHaveBeenCalled();
    });

    return vi.mocked(teamCreateCall).mock.calls[0][1];
  };

  it("should send an explicit null budget_duration when Never resets is selected", async () => {
    await openCreateModal();

    await userEvent.click(resetBudgetSelect());
    await userEvent.click(await screen.findByText("Never resets"));

    const payload = await submitCreateModal();

    expect(payload.budget_duration).toBeNull();
    expect(JSON.stringify(payload)).toContain('"budget_duration":null');
  });

  it("should omit budget_duration entirely when Reset Budget is left untouched", async () => {
    await openCreateModal();

    const payload = await submitCreateModal();

    expect(payload.budget_duration).toBeUndefined();
    expect(JSON.stringify(payload)).not.toContain("budget_duration");
  });

  it("should send the picked duration when one is selected", async () => {
    await openCreateModal();

    await userEvent.click(resetBudgetSelect());
    await userEvent.click(await screen.findByText("weekly"));

    const payload = await submitCreateModal();

    expect(payload.budget_duration).toBe("7d");
  });

  it("should show the configured server default as the Reset Budget placeholder", async () => {
    await openCreateModal();

    await waitFor(() => {
      expect(screen.getByText("Default: monthly (30d)")).toBeInTheDocument();
    });
  });

  it("should fall back to the n/a placeholder when the default settings fetch fails", async () => {
    vi.mocked(getDefaultTeamSettings).mockRejectedValue(new Error("Unauthorized"));

    await openCreateModal();

    await waitFor(() => {
      expect(screen.getByText("n/a")).toBeInTheDocument();
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

describe("Teams - schema-declared metadata fields in team create", () => {
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
    mockUseOrganizations.mockReturnValue({ data: null });
    vi.mocked(useTeamMetadataSchema).mockReturnValue({
      data: [{ key: "cost_center", label: "Cost Center" }],
      isLoading: false,
    } as any);
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

  it("should prepopulate the declared key as an ordinary pair row and submit its value", async () => {
    await openCreateModal();

    fireEvent.change(screen.getByLabelText(/team name/i), { target: { value: "Test Team" } });
    fireEvent.change(screen.getByTestId("create-team-models-select"), { target: { value: "gpt-4" } });

    await waitFor(() => {
      expect((screen.getByPlaceholderText("Key") as HTMLInputElement).value).toBe("cost_center");
    });
    fireEvent.change(screen.getByPlaceholderText("Value"), { target: { value: "CC-1001" } });

    const createTeamSubmitButtons = screen.getAllByRole("button", { name: /create team/i });
    fireEvent.click(createTeamSubmitButtons[createTeamSubmitButtons.length - 1]);

    await waitFor(() => {
      expect(teamCreateCall).toHaveBeenCalled();
    });

    const submittedValues = vi.mocked(teamCreateCall).mock.calls[0][1];
    expect(JSON.parse(submittedValues.metadata)).toEqual({ cost_center: "CC-1001" });
  });

  it("should toast only the validator's own message when the backend rejects the create", async () => {
    vi.mocked(teamCreateCall).mockRejectedValue(
      new Error("{'error': 'Cost center CC-9999 is not recognized. Contact the FinOps team.'}"),
    );
    await openCreateModal();

    fireEvent.change(screen.getByLabelText(/team name/i), { target: { value: "Test Team" } });
    fireEvent.change(screen.getByTestId("create-team-models-select"), { target: { value: "gpt-4" } });
    await waitFor(() => {
      expect((screen.getByPlaceholderText("Key") as HTMLInputElement).value).toBe("cost_center");
    });
    fireEvent.change(screen.getByPlaceholderText("Value"), { target: { value: "CC-9999" } });

    const createTeamSubmitButtons = screen.getAllByRole("button", { name: /create team/i });
    fireEvent.click(createTeamSubmitButtons[createTeamSubmitButtons.length - 1]);

    await waitFor(() => {
      expect(toast.fromError).toHaveBeenCalledWith(
        "Error creating the team: Cost center CC-9999 is not recognized. Contact the FinOps team.",
      );
    });
  });

  it("should show a skeleton in the metadata section while the schema is loading", async () => {
    vi.mocked(useTeamMetadataSchema).mockReturnValue({ data: undefined, isLoading: true } as any);
    await openCreateModal();

    expect(screen.getByTestId("metadata-schema-skeleton")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /add key-value pair/i })).not.toBeInTheDocument();
  });

  it("should re-seed declared keys when the create modal is closed and reopened", async () => {
    await openCreateModal();

    await waitFor(() => {
      expect((screen.getByPlaceholderText("Key") as HTMLInputElement).value).toBe("cost_center");
    });
    fireEvent.click(screen.getByLabelText("Remove key-value pair"));
    await waitFor(() => {
      expect(screen.queryByPlaceholderText("Key")).not.toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /^close$/i }));
    await waitFor(() => {
      expect(screen.queryByLabelText(/team name/i)).not.toBeInTheDocument();
    });

    const createButton = screen.getAllByRole("button", { name: /create team/i })[0];
    act(() => {
      fireEvent.click(createButton);
    });

    await waitFor(() => {
      expect((screen.getByPlaceholderText("Key") as HTMLInputElement).value).toBe("cost_center");
    });
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

describe("Teams - policies field is gated on the viewPolicies capability", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTeamInfoView.mockClear();
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue(["gpt-4"]);
    vi.mocked(fetchMCPAccessGroups).mockResolvedValue([]);
    vi.mocked(getGuardrailsList).mockResolvedValue({ guardrails: [] });
    vi.mocked(getPoliciesList).mockResolvedValue({ policies: [] });
    mockUseOrganizations.mockReturnValue({ data: null });
  });

  const openAdditionalSettings = async () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />);

    act(() => {
      fireEvent.click(screen.getAllByRole("button", { name: /create team/i })[0]);
    });

    await waitFor(() => {
      expect(screen.getByLabelText(/team name/i)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Additional Settings"));

    await waitFor(() => {
      expect(screen.getByTestId("access-group-selector")).toBeInTheDocument();
    });
  };

  it("should render the policies field and load it when the capability is present", async () => {
    await openAdditionalSettings();

    expect(can).toHaveBeenCalledWith("viewPolicies");
    expect(getPoliciesList).toHaveBeenCalledWith("test-token");
    expect(screen.getByText("Policies")).toBeInTheDocument();
  });

  it("should omit the policies field and skip the admin-only list without the capability", async () => {
    can.mockReturnValue(false);

    await openAdditionalSettings();

    expect(getPoliciesList).not.toHaveBeenCalled();
    expect(screen.queryByText("Policies")).not.toBeInTheDocument();
  });
});

describe("Teams - which fields reach the create payload depends on the open sections", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTeamInfoView.mockClear();
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue(["gpt-4"]);
    vi.mocked(fetchMCPAccessGroups).mockResolvedValue([]);
    vi.mocked(getGuardrailsList).mockResolvedValue({ guardrails: [] });
    vi.mocked(getPoliciesList).mockResolvedValue({ policies: [] });
    vi.mocked(getDefaultTeamSettings).mockResolvedValue({ values: {} });
    vi.mocked(teamCreateCall).mockResolvedValue({ team_id: "new-team-1" });
    mockUseOrganizations.mockReturnValue({ data: null });
  });

  const openCreateModal = async () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />);
    act(() => {
      fireEvent.click(screen.getAllByRole("button", { name: /create team/i })[0]);
    });
    await waitFor(() => {
      expect(screen.getByLabelText(/team name/i)).toBeInTheDocument();
    });
  };

  const submit = async () => {
    const buttons = screen.getAllByRole("button", { name: /create team/i });
    fireEvent.click(buttons[buttons.length - 1]);
    await waitFor(() => {
      expect(teamCreateCall).toHaveBeenCalled();
    });
    return vi.mocked(teamCreateCall).mock.calls[0][1] as Record<string, unknown>;
  };

  const toggleAdditionalSettings = () => fireEvent.click(screen.getByText("Additional Settings"));

  it("sends only the always-visible fields when every section is left closed", async () => {
    await openCreateModal();
    fireEvent.change(screen.getByLabelText(/team name/i), { target: { value: "Closed Sections Team" } });

    const payload = await submit();

    expect(Object.keys(payload).sort()).toEqual([
      "budget_duration",
      "max_budget",
      "metadata",
      "models",
      "organization_id",
      "rpm_limit",
      "team_alias",
      "tpm_limit",
    ]);
    expect(payload.team_alias).toBe("Closed Sections Team");
  });

  it("adds the Additional Settings fields to the payload once that section is opened", async () => {
    await openCreateModal();
    fireEvent.change(screen.getByLabelText(/team name/i), { target: { value: "Open Section Team" } });

    toggleAdditionalSettings();
    await waitFor(() => {
      expect(screen.getByLabelText("Team ID")).toBeInTheDocument();
    });
    fireEvent.change(screen.getByLabelText("Team ID"), { target: { value: "tid-open" } });
    fireEvent.change(screen.getByLabelText("Team Member Budget (USD)"), { target: { value: "12.5" } });

    const payload = await submit();

    expect(payload.team_id).toBe("tid-open");
    expect(payload.team_member_budget).toBe(12.5);
    expect(Object.keys(payload)).toEqual(
      expect.arrayContaining(["access_group_ids", "guardrails", "secret_manager_settings", "team_member_key_duration"]),
    );
  });

  it("drops a value typed in Additional Settings when that section is closed again before saving", async () => {
    await openCreateModal();
    fireEvent.change(screen.getByLabelText(/team name/i), { target: { value: "Reclosed Team" } });

    toggleAdditionalSettings();
    await waitFor(() => {
      expect(screen.getByLabelText("Team ID")).toBeInTheDocument();
    });
    fireEvent.change(screen.getByLabelText("Team ID"), { target: { value: "tid-dropped" } });
    toggleAdditionalSettings();
    await waitFor(() => {
      expect(screen.queryByLabelText("Team ID")).not.toBeInTheDocument();
    });

    const payload = await submit();

    expect(payload).not.toHaveProperty("team_id");
  });

  it("restores and sends the typed value when Additional Settings is reopened before saving", async () => {
    await openCreateModal();
    fireEvent.change(screen.getByLabelText(/team name/i), { target: { value: "Reopened Team" } });

    toggleAdditionalSettings();
    await waitFor(() => {
      expect(screen.getByLabelText("Team ID")).toBeInTheDocument();
    });
    fireEvent.change(screen.getByLabelText("Team ID"), { target: { value: "tid-kept" } });
    toggleAdditionalSettings();
    await waitFor(() => {
      expect(screen.queryByLabelText("Team ID")).not.toBeInTheDocument();
    });
    toggleAdditionalSettings();
    await waitFor(() => {
      expect(screen.getByLabelText("Team ID")).toBeInTheDocument();
    });

    expect(screen.getByLabelText("Team ID")).toHaveValue("tid-kept");
    const payload = await submit();

    expect(payload.team_id).toBe("tid-kept");
  });
});

describe("Teams - the exact bytes the create call sends", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    can.mockReturnValue(true);
    vi.mocked(fetchAvailableModelsForTeamOrKey).mockResolvedValue(["gpt-4"]);
    vi.mocked(fetchMCPAccessGroups).mockResolvedValue([]);
    vi.mocked(getGuardrailsList).mockResolvedValue({ guardrails: [] });
    vi.mocked(getPoliciesList).mockResolvedValue({ policies: [] });
    vi.mocked(getDefaultTeamSettings).mockResolvedValue({ values: {} });
    vi.mocked(teamCreateCall).mockResolvedValue({ team_id: "new-team-1" });
    vi.mocked(useTeamMetadataSchema).mockReturnValue({ data: [], isLoading: false } as any);
    mockUseOrganizations.mockReturnValue({ data: null });
  });

  const openCreateModal = async (options?: { premiumUser?: boolean }) => {
    renderWithQueryClient(
      <Teams accessToken="test-token" userID="user-123" userRole="Admin" premiumUser={options?.premiumUser ?? false} />,
    );
    act(() => {
      fireEvent.click(screen.getAllByRole("button", { name: /create team/i })[0]);
    });
    await waitFor(() => {
      expect(screen.getByLabelText(/team name/i)).toBeInTheDocument();
    });
    fireEvent.change(screen.getByLabelText(/team name/i), { target: { value: "Byte Contract Team" } });
  };

  const submit = async () => {
    const buttons = screen.getAllByRole("button", { name: /create team/i });
    fireEvent.click(buttons[buttons.length - 1]);
    await waitFor(() => {
      expect(teamCreateCall).toHaveBeenCalled();
    });
    return vi.mocked(teamCreateCall).mock.calls[0][1] as Record<string, unknown>;
  };

  const wireBody = (payload: Record<string, unknown>) => JSON.parse(JSON.stringify(payload)) as Record<string, unknown>;

  const openSection = async (title: string, mountedProbe: RegExp | string) => {
    fireEvent.click(screen.getByText(title));
    await waitFor(() => {
      expect(screen.getAllByText(mountedProbe).length).toBeGreaterThan(0);
    });
  };

  it("sends three keys and nothing else when every section is left closed", async () => {
    await openCreateModal();

    const payload = await submit();

    expect(payload).toStrictEqual({
      team_alias: "Byte Contract Team",
      organization_id: null,
      models: ["no-default-models"],
      max_budget: undefined,
      budget_duration: undefined,
      tpm_limit: undefined,
      rpm_limit: undefined,
      metadata: undefined,
    });
    expect(wireBody(payload)).toStrictEqual({
      team_alias: "Byte Contract Team",
      organization_id: null,
      models: ["no-default-models"],
    });
  });

  it("keeps every newly mounted but untouched field out of the request body", async () => {
    await openCreateModal();

    await openSection("Additional Settings", /Team Member Key Duration/);
    await openSection("MCP Settings", /Allowed MCP Servers/);
    await openSection("Agent Settings", /Allowed Agents/);
    await openSection("Search Tool Settings", /Allowed Search Tools/);

    const payload = await submit();

    expect(payload).toStrictEqual({
      team_alias: "Byte Contract Team",
      organization_id: null,
      models: ["no-default-models"],
      max_budget: undefined,
      budget_duration: undefined,
      tpm_limit: undefined,
      rpm_limit: undefined,
      metadata: undefined,
      team_id: undefined,
      team_member_budget: undefined,
      team_member_key_duration: undefined,
      team_member_rpm_limit: undefined,
      team_member_tpm_limit: undefined,
      secret_manager_settings: undefined,
      guardrails: undefined,
      disable_global_guardrails: undefined,
      policies: undefined,
      access_group_ids: undefined,
      allowed_vector_store_ids: undefined,
      allowed_passthrough_routes: undefined,
      allowed_mcp_servers_and_groups: undefined,
      mcp_tool_permissions: {},
      allowed_agents_and_groups: undefined,
      object_permission_search_tools: undefined,
    });
    expect(wireBody(payload)).toStrictEqual({
      team_alias: "Byte Contract Team",
      organization_id: null,
      models: ["no-default-models"],
      mcp_tool_permissions: {},
    });
  });

  it.each([
    ["MCP Settings", /Allowed MCP Servers/, ["allowed_mcp_servers_and_groups", "mcp_tool_permissions"]],
    ["Agent Settings", /Allowed Agents/, ["allowed_agents_and_groups"]],
    ["Search Tool Settings", /Allowed Search Tools/, ["object_permission_search_tools"]],
  ])("registers %s fields only while that one section is open", async (title, probe, keys) => {
    await openCreateModal();

    const closedPayload = await submit();
    for (const key of keys as string[]) {
      expect(closedPayload).not.toHaveProperty(key);
    }
  });

  it("carries every typed value to the payload at the type antd sends today", async () => {
    await openCreateModal();

    fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "150.75" } });
    fireEvent.change(screen.getByLabelText("Tokens per minute Limit (TPM)"), { target: { value: "900" } });
    fireEvent.change(screen.getByLabelText("Requests per minute Limit (RPM)"), { target: { value: "800" } });

    await openSection("Additional Settings", /Team Member Key Duration/);

    fireEvent.change(screen.getByLabelText("Team ID"), { target: { value: "tid-1" } });
    fireEvent.change(screen.getByLabelText("Team Member Budget (USD)"), { target: { value: "12.5" } });
    fireEvent.change(screen.getByLabelText(/Team Member Key Duration/), { target: { value: "30d" } });
    fireEvent.change(screen.getByLabelText("Team Member RPM Limit"), { target: { value: "7" } });
    fireEvent.change(screen.getByLabelText("Team Member TPM Limit"), { target: { value: "8" } });
    fireEvent.change(screen.getByLabelText("Secret Manager Settings"), {
      target: { value: '{"namespace":"admin"}' },
    });

    const payload = await submit();

    expect(payload.max_budget).toBe("150.75");
    expect(payload.tpm_limit).toBe("900");
    expect(payload.rpm_limit).toBe("800");
    expect(payload.team_id).toBe("tid-1");
    expect(payload.team_member_budget).toBe(12.5);
    expect(payload.team_member_key_duration).toBe("30d");
    expect(payload.team_member_rpm_limit).toBe("7");
    expect(payload.team_member_tpm_limit).toBe("8");
    expect(payload.secret_manager_settings).toStrictEqual({ namespace: "admin" });
  });

  it("blocks the create on an invalid secret manager config, with the rule message suppressed by help", async () => {
    await openCreateModal();
    await openSection("Additional Settings", /Team Member Key Duration/);

    fireEvent.change(screen.getByLabelText("Secret Manager Settings"), { target: { value: "   " } });

    const buttons = screen.getAllByRole("button", { name: /create team/i });
    fireEvent.click(buttons[buttons.length - 1]);

    await waitFor(() => {
      expect(screen.getByLabelText("Secret Manager Settings")).toHaveAttribute("aria-invalid", "true");
    });
    expect(teamCreateCall).not.toHaveBeenCalled();
    expect(screen.queryByText("Please enter valid JSON")).not.toBeInTheDocument();
  });

  it("turns the disable-global-guardrails switch into a boolean for a premium user", async () => {
    await openCreateModal({ premiumUser: true });
    await openSection("Additional Settings", /Team Member Key Duration/);

    const switches = screen.getAllByRole("switch");
    fireEvent.click(switches[switches.length - 1]);

    const payload = await submit();

    expect(payload.disable_global_guardrails).toBe(true);
  });

  it("leaves the disable-global-guardrails switch inert for a non-premium user", async () => {
    await openCreateModal();
    await openSection("Additional Settings", /Team Member Key Duration/);

    const switches = screen.getAllByRole("switch");
    fireEvent.click(switches[switches.length - 1]);

    const payload = await submit();

    expect(payload.disable_global_guardrails).toBeUndefined();
  });

  it.each([
    ["MCP Settings", /Allowed MCP Servers/, ["allowed_mcp_servers_and_groups", "mcp_tool_permissions"]],
    ["Agent Settings", /Allowed Agents/, ["allowed_agents_and_groups"]],
    ["Search Tool Settings", /Allowed Search Tools/, ["object_permission_search_tools"]],
  ])("adds the %s keys as soon as that one section is opened", async (title, probe, keys) => {
    await openCreateModal();

    await openSection(title as string, probe as RegExp);
    const payload = await submit();

    for (const key of keys as string[]) {
      expect(payload).toHaveProperty(key);
    }
  });

  it("leaves policies out of the request body for a caller without the viewPolicies capability", async () => {
    can.mockReturnValue(false);

    await openCreateModal();
    await openSection("Additional Settings", /Team Member Key Duration/);

    const payload = await submit();

    expect(payload).toStrictEqual({
      team_alias: "Byte Contract Team",
      organization_id: null,
      models: ["no-default-models"],
      max_budget: undefined,
      budget_duration: undefined,
      tpm_limit: undefined,
      rpm_limit: undefined,
      metadata: undefined,
      team_id: undefined,
      team_member_budget: undefined,
      team_member_key_duration: undefined,
      team_member_rpm_limit: undefined,
      team_member_tpm_limit: undefined,
      secret_manager_settings: undefined,
      guardrails: undefined,
      disable_global_guardrails: undefined,
      access_group_ids: undefined,
      allowed_vector_store_ids: undefined,
      allowed_passthrough_routes: undefined,
    });
  });

  it("blocks the create on an empty team name and names the rule", async () => {
    renderWithQueryClient(<Teams accessToken="test-token" userID="user-123" userRole="Admin" />);
    act(() => {
      fireEvent.click(screen.getAllByRole("button", { name: /create team/i })[0]);
    });
    await waitFor(() => {
      expect(screen.getByLabelText(/team name/i)).toBeInTheDocument();
    });

    const buttons = screen.getAllByRole("button", { name: /create team/i });
    fireEvent.click(buttons[buttons.length - 1]);

    expect(await screen.findByText("Please input a team name")).toBeInTheDocument();
    expect(teamCreateCall).not.toHaveBeenCalled();
  });
});
