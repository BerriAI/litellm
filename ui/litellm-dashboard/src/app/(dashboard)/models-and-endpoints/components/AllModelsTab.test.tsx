import * as useAuthorizedModule from "@/app/(dashboard)/hooks/useAuthorized";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import AllModelsTab from "./AllModelsTab";
import { STATUS_COLUMN_ID, toServerSortField } from "./ModelsTableColumns";

const mockModelDeleteCall = vi.fn().mockResolvedValue({});
const mockModelPatchUpdateCall = vi.fn().mockResolvedValue({});
vi.mock("@/components/networking", () => ({
  modelDeleteCall: (...args: unknown[]) => mockModelDeleteCall(...args),
  modelPatchUpdateCall: (...args: unknown[]) => mockModelPatchUpdateCall(...args),
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: { success: vi.fn(), fromBackend: vi.fn() },
}));

vi.mock("@/components/model_dashboard/ModelSettingsModal/ModelSettingsModal", () => ({
  default: function ModelSettingsModalMock({ isVisible }: { isVisible: boolean }) {
    return isVisible ? <div data-testid="model-settings-modal" /> : null;
  },
}));

const mockInvalidateQueries = vi.fn();
vi.mock("@tanstack/react-query", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tanstack/react-query")>();
  return { ...actual, useQueryClient: () => ({ invalidateQueries: mockInvalidateQueries }) };
});

interface ModelsInfoArgs {
  page?: number;
  size?: number;
  search?: string;
  teamId?: string;
  sortBy?: string;
  sortOrder?: string;
}

const modelsInfoCalls: ModelsInfoArgs[] = [];
const mockRefetch = vi.fn();
let modelsInfoResult: Record<string, unknown> = {};

type UseModelsInfoArgs = [
  page?: number,
  size?: number,
  search?: string,
  modelId?: string,
  teamId?: string,
  sortBy?: string,
  sortOrder?: string,
];

vi.mock("../../hooks/models/useModels", () => ({
  useModelsInfo: (...args: UseModelsInfoArgs) => {
    const [page, size, search, , teamId, sortBy, sortOrder] = args;
    const call: ModelsInfoArgs = { page, size, search, teamId, sortBy, sortOrder };
    modelsInfoCalls.push(call);
    return { ...modelsInfoResult, refetch: mockRefetch };
  },
}));

vi.mock("../../hooks/models/useModelCostMap", () => ({
  useModelCostMap: () => ({ data: { "gpt-4": { litellm_provider: "openai" } }, isLoading: false, error: null }),
}));

const mockTeams = [{ team_id: "team-1", team_alias: "Engineering" }];
vi.mock("../../hooks/teams/useTeams", () => ({
  useTeams: () => ({ data: mockTeams, isLoading: false, error: null, refetch: vi.fn() }),
}));

const BASE_MODEL_INFO = {
  id: "model-1",
  db_model: true,
  created_by: "user-123",
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-02T00:00:00Z",
  team_id: "team-1",
  access_groups: [],
};

const makeRow = (overrides: Record<string, unknown> = {}) => ({
  model_name: "gpt-4",
  litellm_params: { model: "openai/gpt-4", custom_llm_provider: "openai" },
  model_info: { ...BASE_MODEL_INFO, ...((overrides.model_info as Record<string, unknown>) ?? {}) },
});

const setModelsInfo = (rows: Record<string, unknown>[], totalCount = rows.length, isLoading = false) => {
  modelsInfoResult = {
    data: { data: rows, total_count: totalCount, current_page: 1, total_pages: 1, size: 50 },
    isLoading,
    isFetching: false,
    error: null,
  };
};

const lastModelsInfoCall = (): ModelsInfoArgs => modelsInfoCalls[modelsInfoCalls.length - 1];

const MOCK_AUTHORIZED = {
  isLoading: false,
  isAuthorized: true,
  token: "mock-token",
  accessToken: "mock-access-token",
  userId: "user-123",
  userEmail: "test@example.com",
  userRole: "Admin",
  premiumUser: true,
  disabledPersonalKeyCreation: false,
  showSSOBanner: false,
};

const mockSetSelectedModelGroup = vi.fn();
const mockSetSelectedModelId = vi.fn();
const mockSetSelectedTeamId = vi.fn();

const defaultProps = {
  selectedModelGroup: "all",
  setSelectedModelGroup: mockSetSelectedModelGroup,
  availableModelGroups: ["gpt-4", "gpt-3.5-turbo"],
  availableModelAccessGroups: ["sales-team"],
  setSelectedModelId: mockSetSelectedModelId,
  setSelectedTeamId: mockSetSelectedTeamId,
};

describe("AllModelsTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    modelsInfoCalls.length = 0;
    setModelsInfo([makeRow()]);
    vi.spyOn(useAuthorizedModule, "default").mockReturnValue(MOCK_AUTHORIZED);
  });

  it("renders the fetched models and the server row count", async () => {
    setModelsInfo([makeRow()], 137);
    render(<AllModelsTab {...defaultProps} />);

    expect(await screen.findByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-50 of 137");
  });

  it("shows the empty state when the proxy returns no models", () => {
    setModelsInfo([], 0);
    render(<AllModelsTab {...defaultProps} />);

    expect(screen.getByText("No models found")).toBeInTheDocument();
  });

  it("shows the loading skeleton while the first page is in flight", () => {
    setModelsInfo([], 0, true);
    render(<AllModelsTab {...defaultProps} />);

    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("No models found")).not.toBeInTheDocument();
  });

  describe("server sort contract", () => {
    const sortHeader = (columnId: string): HTMLElement => screen.getByTestId(`sort-header-${columnId}`);

    const expectIndicator = async (columnId: string, state: "asc" | "desc" | "none") => {
      await waitFor(() => {
        expect(sortHeader(columnId).querySelector(`[data-sort-indicator="${state}"]`)).not.toBeNull();
      });
    };

    const cases: [string, string, string, "asc" | "desc"][] = [
      ["Model Information", "model_name", "model_name", "asc"],
      ["Created By", "model_info_created_by", "created_at", "asc"],
      ["Updated At", "model_info_updated_at", "updated_at", "asc"],
      ["Costs", "input_cost", "costs", "desc"],
    ];

    it.each(cases)("sorts %s using the server field %s", async (_label, columnId, serverField, firstDirection) => {
      const user = userEvent.setup();
      render(<AllModelsTab {...defaultProps} />);

      await user.click(sortHeader(columnId));
      await expectIndicator(columnId, firstDirection);

      expect(lastModelsInfoCall().sortBy).toBe(serverField);
      expect(lastModelsInfoCall().sortOrder).toBe(firstDirection);
    });

    it("maps the hidden Status column to the server field status", () => {
      expect(toServerSortField(STATUS_COLUMN_ID)).toBe("status");
    });

    it("cycles a sorted column back to unsorted", async () => {
      const user = userEvent.setup();
      render(<AllModelsTab {...defaultProps} />);

      await user.click(sortHeader("model_info_updated_at"));
      await expectIndicator("model_info_updated_at", "asc");
      expect(lastModelsInfoCall().sortOrder).toBe("asc");

      await user.click(sortHeader("model_info_updated_at"));
      await expectIndicator("model_info_updated_at", "desc");
      expect(lastModelsInfoCall().sortOrder).toBe("desc");

      await user.click(sortHeader("model_info_updated_at"));
      await expectIndicator("model_info_updated_at", "none");
      expect(lastModelsInfoCall().sortBy).toBeUndefined();
    });
  });

  it("queries the selected team and resets to the first page", async () => {
    const user = userEvent.setup();
    render(<AllModelsTab {...defaultProps} />);

    expect(lastModelsInfoCall().teamId).toBeUndefined();

    await user.click(screen.getByTestId("models-team-select"));
    await user.click(await screen.findByRole("option", { name: "Engineering" }));

    await waitFor(() => {
      expect(lastModelsInfoCall().teamId).toBe("team-1");
    });
    expect(lastModelsInfoCall().page).toBe(1);
  });

  it("debounces the model name search into the server query", async () => {
    const user = userEvent.setup();
    render(<AllModelsTab {...defaultProps} />);

    await user.type(screen.getByTestId("datatable-search"), "claude");

    await waitFor(() => {
      expect(lastModelsInfoCall().search).toBe("claude");
    });
  });

  it("applies a public model name filter through the drawer", async () => {
    const user = userEvent.setup();
    render(<AllModelsTab {...defaultProps} />);

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await user.click(await screen.findByPlaceholderText("Filter by Public Model Name"));
    await user.click(await screen.findByRole("option", { name: "gpt-3.5-turbo" }));
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => {
      expect(mockSetSelectedModelGroup).toHaveBeenCalledWith("gpt-3.5-turbo");
    });
  });

  it("filters the fetched page down to the selected model group", () => {
    setModelsInfo([makeRow(), { ...makeRow(), model_name: "claude-opus" }], 2);
    render(<AllModelsTab {...defaultProps} selectedModelGroup="claude-opus" />);

    const table = screen.getByRole("table");
    expect(within(table).getByText("claude-opus")).toBeInTheDocument();
    expect(within(table).queryByText("gpt-4")).not.toBeInTheDocument();
  });

  it("resets search, filters, team and sorting from the drawer reset button", async () => {
    const user = userEvent.setup();
    render(<AllModelsTab {...defaultProps} selectedModelGroup="gpt-4" />);

    await user.click(screen.getByTestId("models-team-select"));
    await user.click(await screen.findByRole("option", { name: "Engineering" }));
    await waitFor(() => expect(lastModelsInfoCall().teamId).toBe("team-1"));

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await user.click(await screen.findByTestId("filter-drawer-reset"));

    expect(mockSetSelectedModelGroup).toHaveBeenCalledWith("all");
    await waitFor(() => {
      expect(lastModelsInfoCall().teamId).toBeUndefined();
    });
  });

  it("opens the delete modal from the row and deletes the model", async () => {
    const user = userEvent.setup();
    render(<AllModelsTab {...defaultProps} />);

    await user.click(await screen.findByTestId("model-delete-model-1"));
    expect(await screen.findByText("Delete Model")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^delete$/i }));

    await waitFor(() => {
      expect(mockModelDeleteCall).toHaveBeenCalledWith("mock-access-token", "model-1");
    });
  });

  it("pauses a model through the row toggle", async () => {
    const user = userEvent.setup();
    render(<AllModelsTab {...defaultProps} />);

    await user.click(await screen.findByTestId("model-pause-toggle-model-1"));

    await waitFor(() => {
      expect(mockModelPatchUpdateCall).toHaveBeenCalledWith("mock-access-token", { blocked: true }, "model-1");
    });
  });

  it("opens the model settings modal from the toolbar", async () => {
    const user = userEvent.setup();
    render(<AllModelsTab {...defaultProps} />);

    expect(screen.queryByTestId("model-settings-modal")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("models-settings-trigger"));
    expect(screen.getByTestId("model-settings-modal")).toBeInTheDocument();
  });

  it("opens the model detail view from the model ID cell", async () => {
    const user = userEvent.setup();
    render(<AllModelsTab {...defaultProps} />);

    await user.click(await screen.findByTestId("model-id-model-1"));

    expect(mockSetSelectedModelId).toHaveBeenCalledWith("model-1");
  });

  it("opens the team detail view from the team ID cell", async () => {
    const user = userEvent.setup();
    render(<AllModelsTab {...defaultProps} />);

    await user.click(await screen.findByTestId("model-team-id-model-1"));

    expect(mockSetSelectedTeamId).toHaveBeenCalledWith("team-1");
  });

  describe("virtual key hint", () => {
    it("explains personal key creation while viewing current team models", () => {
      render(<AllModelsTab {...defaultProps} />);

      expect(screen.getByText(/create a Virtual Key without selecting a team/i)).toBeInTheDocument();
    });

    it("names the selected team in the hint", async () => {
      const user = userEvent.setup();
      render(<AllModelsTab {...defaultProps} />);

      await user.click(screen.getByTestId("models-team-select"));
      await user.click(await screen.findByRole("option", { name: "Engineering" }));

      expect(await screen.findByText(/select Team as "Engineering"/i)).toBeInTheDocument();
    });

    it("hides the hint when viewing all available models", async () => {
      const user = userEvent.setup();
      render(<AllModelsTab {...defaultProps} />);

      await user.click(screen.getByTestId("models-view-select"));
      await user.click(await screen.findByRole("option", { name: "All Available Models" }));

      await waitFor(() => {
        expect(screen.queryByText(/create a Virtual Key/i)).not.toBeInTheDocument();
      });
    });
  });
});
