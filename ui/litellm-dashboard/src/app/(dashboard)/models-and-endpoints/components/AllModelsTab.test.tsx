import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, beforeAll, vi, beforeEach } from "vitest";
import AllModelsTab from "./AllModelsTab";
import * as useTeamsModule from "@/app/(dashboard)/hooks/useTeams";
import * as useAuthorizedModule from "@/app/(dashboard)/hooks/useAuthorized";

// Mock window.matchMedia for Ant Design components
beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
});

describe("AllModelsTab", () => {
  const mockSetSelectedModelGroup = vi.fn();
  const mockSetSelectedModelId = vi.fn();
  const mockSetSelectedTeamId = vi.fn();
  const mockSetEditModel = vi.fn();

  const defaultProps = {
    selectedModelGroup: "all",
    setSelectedModelGroup: mockSetSelectedModelGroup,
    availableModelGroups: ["gpt-4", "gpt-3.5-turbo"],
    availableModelAccessGroups: ["sales-team", "engineering-team"],
    setSelectedModelId: mockSetSelectedModelId,
    setSelectedTeamId: mockSetSelectedTeamId,
    setEditModel: mockSetEditModel,
    modelData: {
      data: [],
    },
  };

  const mockUseAuthorized = {
    token: "mock-token",
    accessToken: "mock-access-token",
    userId: "user-123",
    userEmail: "test@example.com",
    userRole: "Admin",
    premiumUser: true,
    disabledPersonalKeyCreation: false,
    showSSOBanner: false,
  };

  beforeAll(() => {
    // Mock useAuthorized hook
    vi.spyOn(useAuthorizedModule, "default").mockReturnValue(mockUseAuthorized);
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render with empty data", () => {
    // Mock useTeams hook
    vi.spyOn(useTeamsModule, "default").mockReturnValue({
      teams: [],
      setTeams: vi.fn(),
    });

    const { container } = render(<AllModelsTab {...defaultProps} />);
    expect(container).toBeTruthy();
    expect(screen.getByText("Current Team:")).toBeInTheDocument();
  });

  it("should filter models by direct team access when current team is selected", async () => {
    const user = userEvent.setup();
    const mockTeams = [
      {
        team_id: "team-456",
        team_alias: "Engineering Team",
        models: ["gpt-4"],
        max_budget: null,
        budget_duration: null,
        tpm_limit: null,
        rpm_limit: null,
        organization_id: "org-123",
        created_at: "2024-01-01",
        keys: [],
        members_with_roles: [],
      },
    ];

    // Mock useTeams hook with team data
    vi.spyOn(useTeamsModule, "default").mockReturnValue({
      teams: mockTeams,
      setTeams: vi.fn(),
    });

    const modelData = {
      data: [
        {
          model_name: "gpt-4-accessible",
          model_info: {
            id: "model-1",
            access_via_team_ids: ["team-456"], // Direct team access
            access_groups: [],
          },
        },
        {
          model_name: "gpt-3.5-turbo-blocked",
          model_info: {
            id: "model-2",
            access_via_team_ids: ["team-789"], // Different team
            access_groups: [],
          },
        },
      ],
    };

    render(<AllModelsTab {...defaultProps} modelData={modelData} />);

    // Initially on "personal" team, should show 0 results (no models have direct_access)
    expect(screen.getByText("Showing 0 results")).toBeInTheDocument();

    // Click on the team selector to change to Engineering Team
    const teamSelector = screen.getAllByRole("button").find((btn) => btn.textContent?.includes("Personal"));
    expect(teamSelector).toBeInTheDocument();

    await user.click(teamSelector!);

    // Click on Engineering Team option
    await waitFor(async () => {
      const engineeringOption = await screen.findByText(/Engineering Team/);
      await user.click(engineeringOption);
    });

    // After selecting Engineering Team, should show 1 result (gpt-4-accessible has direct team access)
    await waitFor(() => {
      expect(screen.getByText("Showing 1 - 1 of 1 results")).toBeInTheDocument();
    });
  });

  it("should filter models by access group matching when team models match model access groups", async () => {
    const user = userEvent.setup();
    const mockTeams = [
      {
        team_id: "team-sales",
        team_alias: "Sales Team",
        models: ["sales-model-group"], // Team has this model group
        max_budget: null,
        budget_duration: null,
        tpm_limit: null,
        rpm_limit: null,
        organization_id: "org-123",
        created_at: "2024-01-01",
        keys: [],
        members_with_roles: [],
      },
    ];

    // Mock useTeams hook
    vi.spyOn(useTeamsModule, "default").mockReturnValue({
      teams: mockTeams,
      setTeams: vi.fn(),
    });

    const modelData = {
      data: [
        {
          model_name: "gpt-4-sales",
          model_info: {
            id: "model-sales-1",
            access_via_team_ids: [], // No direct team access
            access_groups: ["sales-model-group"], // But has access group that matches team's models
          },
        },
        {
          model_name: "gpt-4-engineering",
          model_info: {
            id: "model-eng-1",
            access_via_team_ids: [],
            access_groups: ["engineering-model-group"], // Different access group
          },
        },
      ],
    };

    render(<AllModelsTab {...defaultProps} modelData={modelData} />);

    // Initially on "personal" team, should show 0 results
    expect(screen.getByText("Showing 0 results")).toBeInTheDocument();

    // Click on the team selector
    const teamSelector = screen.getAllByRole("button").find((btn) => btn.textContent?.includes("Personal"));
    expect(teamSelector).toBeInTheDocument();

    await user.click(teamSelector!);

    // Click on Sales Team option
    await waitFor(async () => {
      const salesOption = await screen.findByText(/Sales Team/);
      await user.click(salesOption);
    });

    // After selecting Sales Team, should show 1 result (gpt-4-sales has matching access group)
    await waitFor(() => {
      expect(screen.getByText("Showing 1 - 1 of 1 results")).toBeInTheDocument();
    });
  });

  it("should filter models by direct_access for personal team", async () => {
    // Mock useTeams hook
    vi.spyOn(useTeamsModule, "default").mockReturnValue({
      teams: [],
      setTeams: vi.fn(),
    });

    const modelData = {
      data: [
        {
          model_name: "gpt-4-personal",
          model_info: {
            id: "model-personal-1",
            direct_access: true, // Available for personal use
            access_via_team_ids: [],
            access_groups: [],
          },
        },
        {
          model_name: "gpt-4-team-only",
          model_info: {
            id: "model-team-1",
            direct_access: false, // Not available for personal use
            access_via_team_ids: ["team-123"],
            access_groups: [],
          },
        },
      ],
    };

    render(<AllModelsTab {...defaultProps} modelData={modelData} />);

    // When currentTeam is "personal" (default), it should filter by direct_access === true
    // This tests the personal access logic in lines 72-73
    // Should show 1 result (only gpt-4-personal with direct_access=true)
    await waitFor(() => {
      expect(screen.getByText("Showing 1 - 1 of 1 results")).toBeInTheDocument();
    });
  });
});
