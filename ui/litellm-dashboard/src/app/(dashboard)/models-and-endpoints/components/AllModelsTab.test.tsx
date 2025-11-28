import * as useAuthorizedModule from "@/app/(dashboard)/hooks/useAuthorized";
import * as useTeamsModule from "@/app/(dashboard)/hooks/useTeams";
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AllModelsTab from "./AllModelsTab";

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

  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(useAuthorizedModule, "default").mockReturnValue(mockUseAuthorized);
  });

  it("should render with empty data", () => {
    vi.spyOn(useTeamsModule, "default").mockReturnValue({
      teams: [],
      setTeams: vi.fn(),
    });

    render(<AllModelsTab {...defaultProps} />);
    expect(screen.getByText("Current Team:")).toBeInTheDocument();
  });

  it("should filter models by direct team access when current team is selected", async () => {
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
            access_via_team_ids: ["team-456"],
            access_groups: [],
          },
        },
        {
          model_name: "gpt-3.5-turbo-blocked",
          model_info: {
            id: "model-2",
            access_via_team_ids: ["team-789"],
            access_groups: [],
          },
        },
      ],
    };

    render(<AllModelsTab {...defaultProps} modelData={modelData} />);

    await waitFor(() => {
      expect(screen.getByText("Showing 0 results")).toBeInTheDocument();
    });
  });

  it("should filter models by access group matching when team models match model access groups", async () => {
    const mockTeams = [
      {
        team_id: "team-sales",
        team_alias: "Sales Team",
        models: ["sales-model-group"],
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
            access_via_team_ids: [],
            access_groups: ["sales-model-group"],
          },
        },
        {
          model_name: "gpt-4-engineering",
          model_info: {
            id: "model-eng-1",
            access_via_team_ids: [],
            access_groups: ["engineering-model-group"],
          },
        },
      ],
    };

    render(<AllModelsTab {...defaultProps} modelData={modelData} />);

    await waitFor(() => {
      expect(screen.getByText("Showing 0 results")).toBeInTheDocument();
    });
  });

  it("should filter models by direct_access for personal team", async () => {
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
            direct_access: true,
            access_via_team_ids: [],
            access_groups: [],
          },
        },
        {
          model_name: "gpt-4-team-only",
          model_info: {
            id: "model-team-1",
            direct_access: false,
            access_via_team_ids: ["team-123"],
            access_groups: [],
          },
        },
      ],
    };

    render(<AllModelsTab {...defaultProps} modelData={modelData} />);

    await waitFor(() => {
      expect(screen.getByText("Showing 1 - 1 of 1 results")).toBeInTheDocument();
    });
  });

  it("should show config model status for models defined in configs", async () => {
    vi.spyOn(useTeamsModule, "default").mockReturnValue({
      teams: [],
      setTeams: vi.fn(),
    });

    const modelData = {
      data: [
        {
          model_name: "gpt-4-config",
          litellm_model_name: "gpt-4-config",
          provider: "openai",
          model_info: {
            id: "model-config-1",
            db_model: false,
            direct_access: true,
            access_via_team_ids: [],
            access_groups: [],
            created_by: "user-123",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
          },
        },
        {
          model_name: "gpt-4-db",
          litellm_model_name: "gpt-4-db",
          provider: "openai",
          model_info: {
            id: "model-db-1",
            db_model: true,
            direct_access: true,
            access_via_team_ids: [],
            access_groups: [],
            created_by: "user-123",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
          },
        },
      ],
    };

    render(<AllModelsTab {...defaultProps} modelData={modelData} />);

    await waitFor(() => {
      expect(screen.getByText("Config Model")).toBeInTheDocument();
      expect(screen.getByText("DB Model")).toBeInTheDocument();
    });
  });

  it("should show 'Defined in config' for models defined in configs", async () => {
    vi.spyOn(useTeamsModule, "default").mockReturnValue({
      teams: [],
      setTeams: vi.fn(),
    });

    const modelData = {
      data: [
        {
          model_name: "gpt-4-config-model",
          litellm_model_name: "gpt-4-config-model",
          provider: "openai",
          model_info: {
            id: "model-config-defined",
            db_model: false,
            direct_access: true,
            access_via_team_ids: [],
            access_groups: [],
            created_by: "user-123",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
          },
        },
      ],
    };

    render(<AllModelsTab {...defaultProps} modelData={modelData} />);

    expect(screen.getByText("Defined in config")).toBeInTheDocument();
  });
});
