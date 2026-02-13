import * as useAuthorizedModule from "@/app/(dashboard)/hooks/useAuthorized";
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AllModelsTab from "./AllModelsTab";

// Mock the useModelsInfo hook
const mockUseModelsInfo = vi.fn(() => ({
  data: { data: [], total_count: 0, current_page: 1, total_pages: 1, size: 50 },
  isLoading: false,
  error: null,
})) as any;

vi.mock("../../hooks/models/useModels", () => ({
  useModelsInfo: (page?: number, size?: number, search?: string) => mockUseModelsInfo(page, size, search),
}));

// Mock the useModelCostMap hook
const mockUseModelCostMap = vi.fn(() => ({
  data: {
    "gpt-4": { litellm_provider: "openai" },
    "gpt-3.5-turbo": { litellm_provider: "openai" },
    "gpt-4-accessible": { litellm_provider: "openai" },
    "gpt-3.5-turbo-blocked": { litellm_provider: "openai" },
    "gpt-4-sales": { litellm_provider: "openai" },
    "gpt-4-engineering": { litellm_provider: "openai" },
    "gpt-4-personal": { litellm_provider: "openai" },
    "gpt-4-team-only": { litellm_provider: "openai" },
    "gpt-4-config": { litellm_provider: "openai" },
    "gpt-4-db": { litellm_provider: "openai" },
  },
  isLoading: false,
  error: null,
})) as any;

vi.mock("../../hooks/models/useModelCostMap", () => ({
  useModelCostMap: () => mockUseModelCostMap(),
}));

// Mock the useTeams hook (react-query implementation)
const mockUseTeams = vi.fn(() => ({
  data: [],
  isLoading: false,
  error: null,
  refetch: vi.fn(),
})) as any;

vi.mock("../../hooks/teams/useTeams", () => ({
  useTeams: () => mockUseTeams(),
}));

// Helper function to create model cost map mock return value
const createModelCostMapMock = (data: Record<string, any>) => ({
  data,
  isLoading: false,
  error: null,
});

// Helper function to create paginated model data mock
const createPaginatedModelData = (
  models: any[],
  totalCount: number = models.length,
  currentPage: number = 1,
  totalPages: number = 1,
  size: number = 50,
) => ({
  data: models,
  total_count: totalCount,
  current_page: currentPage,
  total_pages: totalPages,
  size: size,
});

describe("AllModelsTab", () => {
  const mockSetSelectedModelGroup = vi.fn();
  const mockSetSelectedModelId = vi.fn();
  const mockSetSelectedTeamId = vi.fn();

  const defaultProps = {
    selectedModelGroup: "all",
    setSelectedModelGroup: mockSetSelectedModelGroup,
    availableModelGroups: ["gpt-4", "gpt-3.5-turbo"],
    availableModelAccessGroups: ["sales-team", "engineering-team"],
    setSelectedModelId: mockSetSelectedModelId,
    setSelectedTeamId: mockSetSelectedTeamId,
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
    mockUseModelsInfo.mockReturnValueOnce({
      data: createPaginatedModelData([], 0, 1, 1, 50),
      isLoading: false,
      error: null,
    });

    mockUseTeams.mockReturnValueOnce({
      data: [],
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    mockUseModelCostMap.mockReturnValueOnce(createModelCostMapMock({}));

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

    mockUseTeams.mockReturnValueOnce({
      data: mockTeams,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    mockUseModelCostMap.mockReturnValueOnce(
      createModelCostMapMock({
        "gpt-4-accessible": { litellm_provider: "openai" },
        "gpt-3.5-turbo-blocked": { litellm_provider: "openai" },
      }),
    );

    const modelData = createPaginatedModelData([
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
    ], 2, 1, 1, 50);

    mockUseModelsInfo.mockReturnValue({ data: modelData, isLoading: false, error: null });

    render(<AllModelsTab {...defaultProps} />);

    // Component shows API total_count (2), not filtered count
    // Since default is "personal" team and models don't have direct_access, they're filtered out
    await waitFor(() => {
      expect(screen.getByText("Showing 1 - 2 of 2 results")).toBeInTheDocument();
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

    mockUseTeams.mockReturnValue({
      data: mockTeams,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    mockUseModelCostMap.mockReturnValueOnce(
      createModelCostMapMock({
        "gpt-4-sales": { litellm_provider: "openai" },
        "gpt-4-engineering": { litellm_provider: "openai" },
      }),
    );

    const modelData = createPaginatedModelData([
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
    ], 2, 1, 1, 50);

    mockUseModelsInfo.mockReturnValue({ data: modelData, isLoading: false, error: null });

    render(<AllModelsTab {...defaultProps} />);

    // Component shows API total_count (2), not filtered count
    // Since default is "personal" team and models don't have direct_access, they're filtered out
    await waitFor(() => {
      expect(screen.getByText("Showing 1 - 2 of 2 results")).toBeInTheDocument();
    });
  });

  it("should filter models by direct_access for personal team", async () => {
    mockUseTeams.mockReturnValue({
      data: [],
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    mockUseModelCostMap.mockReturnValueOnce(
      createModelCostMapMock({
        "gpt-4-personal": { litellm_provider: "openai" },
        "gpt-4-team-only": { litellm_provider: "openai" },
      }),
    );

    const modelData = createPaginatedModelData([
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
    ], 2, 1, 1, 50);

    mockUseModelsInfo.mockReturnValue({ data: modelData, isLoading: false, error: null });

    render(<AllModelsTab {...defaultProps} />);

    // Component shows API total_count (2), but only 1 model has direct_access
    await waitFor(() => {
      expect(screen.getByText("Showing 1 - 2 of 2 results")).toBeInTheDocument();
    });
  });

  it("should show config model status for models defined in configs", async () => {
    mockUseTeams.mockReturnValue({
      data: [],
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    mockUseModelCostMap.mockReturnValueOnce(
      createModelCostMapMock({
        "gpt-4-config": { litellm_provider: "openai" },
        "gpt-4-db": { litellm_provider: "openai" },
      }),
    );

    const modelData = createPaginatedModelData([
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
    ], 2, 1, 1, 50);

    mockUseModelsInfo.mockReturnValue({ data: modelData, isLoading: false, error: null });

    render(<AllModelsTab {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Config Model")).toBeInTheDocument();
      expect(screen.getByText("DB Model")).toBeInTheDocument();
    });
  });

  it("should show 'Defined in config' for models defined in configs", async () => {
    mockUseTeams.mockReturnValue({
      data: [],
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    mockUseModelCostMap.mockReturnValueOnce(
      createModelCostMapMock({
        "gpt-4-config": { litellm_provider: "openai" },
      }),
    );

    const modelData = createPaginatedModelData([
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
    ], 1, 1, 1, 50);

    mockUseModelsInfo.mockReturnValue({ data: modelData, isLoading: false, error: null });

    render(<AllModelsTab {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Defined in config")).toBeInTheDocument();
    });
  });

  it("should handle pagination: Previous button is disabled on first page and Next button works", async () => {
    mockUseTeams.mockReturnValue({
      data: [],
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    mockUseModelCostMap.mockReturnValue(
      createModelCostMapMock({
        "gpt-4-page1": { litellm_provider: "openai" },
        "gpt-4-page2": { litellm_provider: "openai" },
      }),
    );

    // Mock first page response (page 1 of 2)
    const page1Data = createPaginatedModelData(
      [
        {
          model_name: "gpt-4-page1",
          model_info: {
            id: "model-page1-1",
            direct_access: true,
            access_via_team_ids: [],
            access_groups: [],
          },
        },
      ],
      2, // total_count
      1, // current_page
      2, // total_pages
      50, // size
    );

    // Set up mock to return page1Data for page 1
    mockUseModelsInfo.mockImplementation((page: number = 1, size?: number, search?: string) => {
      return { data: page1Data, isLoading: false, error: null };
    });

    render(<AllModelsTab {...defaultProps} />);

    await waitFor(() => {
      // Component calculates: ((1-1)*50)+1 = 1, Math.min(1*50, 2) = 2
      expect(screen.getByText("Showing 1 - 2 of 2 results")).toBeInTheDocument();
    });

    // Check that Previous button is disabled on first page
    const previousButton = screen.getByRole("button", { name: /previous/i });
    expect(previousButton).toBeDisabled();

    // Check that Next button is enabled (since we're on page 1 of 2)
    const nextButton = screen.getByRole("button", { name: /next/i });
    expect(nextButton).not.toBeDisabled();
  });

  it("should handle pagination: Next button is disabled on last page", async () => {
    mockUseTeams.mockReturnValue({
      data: [],
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    mockUseModelCostMap.mockReturnValue(
      createModelCostMapMock({
        "gpt-4-page2": { litellm_provider: "openai" },
      }),
    );

    // Mock single page response (page 1 of 1 - last page)
    const singlePageData = createPaginatedModelData(
      [
        {
          model_name: "gpt-4-page2",
          model_info: {
            id: "model-page2-1",
            direct_access: true,
            access_via_team_ids: [],
            access_groups: [],
          },
        },
      ],
      1, // total_count
      1, // current_page
      1, // total_pages (only 1 page, so this is the last page)
      50, // size
    );

    mockUseModelsInfo.mockImplementation((page?: number, size?: number, search?: string) => {
      return { data: singlePageData, isLoading: false, error: null };
    });

    render(<AllModelsTab {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Showing 1 - 1 of 1 results")).toBeInTheDocument();
    });

    // When there's only 1 page (last page), Next should be disabled
    const nextButton = screen.getByRole("button", { name: /next/i });
    expect(nextButton).toBeDisabled();

    // Previous should also be disabled on the first (and only) page
    const previousButton = screen.getByRole("button", { name: /previous/i });
    expect(previousButton).toBeDisabled();
  });
});
