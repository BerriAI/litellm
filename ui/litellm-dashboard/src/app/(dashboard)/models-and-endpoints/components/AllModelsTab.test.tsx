import * as useAuthorizedModule from "@/app/(dashboard)/hooks/useAuthorized";
import {
  fireEvent,
  render,
  renderWithProviders,
  screen,
  testQueryClient,
  waitFor,
  within,
} from "@/../tests/test-utils";
import { QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { PropsWithChildren } from "react";
import { beforeEach, describe, expect, it, type Mock, vi } from "vitest";

import AllModelsTab from "./AllModelsTab";
import { STATUS_COLUMN_ID, toServerSortField } from "./ModelsTableColumns";

const mockModelDeleteCall = vi.fn().mockResolvedValue({});
const mockModelPatchUpdateCall = vi.fn().mockResolvedValue({});
vi.mock("@/components/networking", () => ({
  serverRootPath: "/",
  modelDeleteCall: (...args: unknown[]) => mockModelDeleteCall(...args),
  modelPatchUpdateCall: (...args: unknown[]) => mockModelPatchUpdateCall(...args),
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
  modelName?: string;
  accessGroup?: string;
  wildcardOnly?: boolean;
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
  excludeAutoRouters?: boolean,
  modelName?: string,
  accessGroup?: string,
  wildcardOnly?: boolean,
];

vi.mock("../../hooks/models/useModels", () => ({
  useModelsInfo: (...args: UseModelsInfoArgs) => {
    const [page, size, search, , teamId, sortBy, sortOrder, , modelName, accessGroup, wildcardOnly] = args;
    const call: ModelsInfoArgs = {
      page,
      size,
      search,
      teamId,
      sortBy,
      sortOrder,
      modelName,
      accessGroup,
      wildcardOnly,
    };
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
    isError: false,
    error: null,
  };
};

const setModelsInfoError = () => {
  modelsInfoResult = { data: undefined, isLoading: false, isFetching: false, isError: true, error: new Error("boom") };
};

const lastModelsInfoCall = (): ModelsInfoArgs => modelsInfoCalls[modelsInfoCalls.length - 1];

const SEARCH_SETTLE_MS = 400;

const MOCK_AUTHORIZED = {
  isLoading: false,
  isAuthorized: true,
  token: "mock-token",
  accessToken: "mock-access-token",
  userId: "user-123",
  userEmail: "test@example.com",
  userRole: "Admin",
  userRoleLabel: "Admin",
  isViewOnly: false,
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

type TabProps = Partial<typeof defaultProps>;

interface RenderTabOptions {
  // NuqsTestingAdapter resets the update queue on mount, which swallows URL writes made by
  // mount effects (the DataTable server page clamp). Opt out to observe those writes.
  keepMountUpdates?: boolean;
}

const renderTab = (
  props: TabProps = {},
  searchParams?: string,
  { keepMountUpdates = false }: RenderTabOptions = {},
) => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  const ui = <AllModelsTab {...defaultProps} {...props} />;
  if (!keepMountUpdates) {
    const view = renderWithProviders(ui, { searchParams, onUrlUpdate });
    return { ...view, onUrlUpdate };
  }
  const MountPreservingProviders = ({ children }: PropsWithChildren) => (
    <NuqsTestingAdapter
      searchParams={searchParams}
      onUrlUpdate={onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={false}
    >
      <QueryClientProvider client={testQueryClient}>{children}</QueryClientProvider>
    </NuqsTestingAdapter>
  );
  const view = render(ui, { wrapper: MountPreservingProviders });
  return { ...view, onUrlUpdate };
};

const lastUrl = (onUrlUpdate: Mock<OnUrlUpdateFunction>): URLSearchParams => {
  const event = onUrlUpdate.mock.calls.at(-1)?.[0];
  if (!event) throw new Error("no URL update was emitted");
  return event.searchParams;
};

const sortHeader = (columnId: string): HTMLElement => screen.getByTestId(`sort-header-${columnId}`);

const expectIndicator = async (columnId: string, state: "asc" | "desc" | "none") => {
  await waitFor(() => {
    expect(sortHeader(columnId).querySelector(`[data-sort-indicator="${state}"]`)).not.toBeNull();
  });
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
    renderTab();

    expect(await screen.findByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-50 of 137");
  });

  it("does not re-query after the mount-time debounced search settles unchanged", async () => {
    renderTab();
    const callsAfterMount = modelsInfoCalls.length;

    await new Promise((resolve) => setTimeout(resolve, SEARCH_SETTLE_MS));

    expect(modelsInfoCalls.length).toBe(callsAfterMount);
  });

  it("shows the empty state when the proxy returns no models", () => {
    setModelsInfo([], 0);
    renderTab();

    expect(screen.getByText("No models found")).toBeInTheDocument();
  });

  it("shows the loading skeleton while the first page is in flight", () => {
    setModelsInfo([], 0, true);
    renderTab();

    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("No models found")).not.toBeInTheDocument();
  });

  describe("server sort contract", () => {
    const cases: [string, string, string, "asc" | "desc"][] = [
      ["Model Information", "model_name", "model_name", "asc"],
      ["Created By", "model_info_created_by", "created_at", "asc"],
      ["Updated At", "model_info_updated_at", "updated_at", "asc"],
      ["Costs", "input_cost", "costs", "desc"],
    ];

    it.each(cases)("sorts %s using the server field %s", async (_label, columnId, serverField, firstDirection) => {
      const user = userEvent.setup();
      renderTab();

      await user.click(sortHeader(columnId));
      await expectIndicator(columnId, firstDirection);

      expect(lastModelsInfoCall().sortBy).toBe(serverField);
      expect(lastModelsInfoCall().sortOrder).toBe(firstDirection);
    });

    it("maps the hidden Source column to the server field status", () => {
      expect(toServerSortField(STATUS_COLUMN_ID)).toBe("status");
    });

    it("cycles a sorted column back to unsorted", async () => {
      const user = userEvent.setup();
      renderTab();

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

  describe("URL state", () => {
    it("reads search, page, page size, sort, team, access group and view from the URL", async () => {
      setModelsInfo([makeRow()], 100);
      renderTab(
        {},
        "?model_search=claude&page=2&page_size=25&sort_by=model_name&sort_order=desc&filter_team=team-1&filter_access_group=sales-team&view=all",
      );

      const expectedQuery: ModelsInfoArgs = {
        page: 2,
        size: 25,
        search: "claude",
        sortBy: "model_name",
        sortOrder: "desc",
        teamId: "team-1",
        accessGroup: "sales-team",
      };
      expect(lastModelsInfoCall()).toMatchObject(expectedQuery);
      expect(screen.getByTestId("datatable-search")).toHaveValue("claude");
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 26-50 of 100");
      expect(screen.getByTestId("models-team-select")).toHaveTextContent("Engineering");
      expect(screen.getByTestId("models-view-select")).toHaveTextContent("All Available Models");
      expect(screen.getByTestId("filter-chip-model_info_access_groups")).toHaveTextContent("sales-team");
      await expectIndicator("model_name", "desc");
      expect(screen.queryByText(/create a Virtual Key/i)).not.toBeInTheDocument();
    });

    it("translates a sortable column id in ?sort_by= into its server sort field", () => {
      renderTab({}, "?sort_by=model_info_created_by");

      expect(lastModelsInfoCall().sortBy).toBe("created_at");
      expect(lastModelsInfoCall().sortOrder).toBe("asc");
    });

    it("ignores a ?sort_by= that names a column which cannot be sorted", async () => {
      renderTab({}, "?sort_by=litellm_credential_name&sort_order=desc");

      expect(lastModelsInfoCall().sortBy).toBeUndefined();
      expect(lastModelsInfoCall().sortOrder).toBeUndefined();
      await expectIndicator("model_name", "none");
    });

    it("clamps ?page_size= to the largest page size the table offers", () => {
      renderTab({}, "?page_size=500");

      expect(lastModelsInfoCall().size).toBe(50);
    });

    it("keeps ?page= when the models request failed instead of snapping back to page 1", async () => {
      setModelsInfoError();
      const { onUrlUpdate } = renderTab({}, "?page=3", { keepMountUpdates: true });

      await new Promise((resolve) => setTimeout(resolve, SEARCH_SETTLE_MS));

      expect(onUrlUpdate).not.toHaveBeenCalled();
      expect(lastModelsInfoCall().page).toBe(3);
    });

    it("clamps ?page= to the last page once the server reports fewer rows", async () => {
      setModelsInfo([makeRow()], 1);
      const { onUrlUpdate } = renderTab({}, "?page=3", { keepMountUpdates: true });

      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      expect(lastUrl(onUrlUpdate).has("page")).toBe(false);
      await waitFor(() => expect(lastModelsInfoCall().page).toBe(1));
    });

    it("writes the typed search to ?model_search= and resets the page", async () => {
      setModelsInfo([makeRow()], 100);
      const { onUrlUpdate } = renderTab({}, "?page=2");

      fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "claude" } });

      await waitFor(() => expect(lastUrl(onUrlUpdate).get("model_search")).toBe("claude"));
      expect(lastUrl(onUrlUpdate).has("page")).toBe(false);
      await waitFor(() => expect(lastModelsInfoCall().search).toBe("claude"));
      expect(lastModelsInfoCall().page).toBe(1);
    });

    it("writes the selected team to ?filter_team=, resets the page, and drops the key for Personal", async () => {
      const user = userEvent.setup();
      setModelsInfo([makeRow()], 100);
      const { onUrlUpdate } = renderTab({}, "?page=2");

      await user.click(screen.getByTestId("models-team-select"));
      await user.click(await screen.findByRole("option", { name: "Engineering" }));

      await waitFor(() => expect(lastUrl(onUrlUpdate).get("filter_team")).toBe("team-1"));
      expect(lastUrl(onUrlUpdate).has("page")).toBe(false);
      await waitFor(() => expect(lastModelsInfoCall().teamId).toBe("team-1"));

      await user.click(screen.getByTestId("models-team-select"));
      await user.click(await screen.findByRole("option", { name: "Personal" }));

      await waitFor(() => expect(lastUrl(onUrlUpdate).has("filter_team")).toBe(false));
    });

    it("writes the view mode to ?view= and drops the key for the default view", async () => {
      const user = userEvent.setup();
      const { onUrlUpdate } = renderTab();

      await user.click(screen.getByTestId("models-view-select"));
      await user.click(await screen.findByRole("option", { name: "All Available Models" }));

      await waitFor(() => expect(lastUrl(onUrlUpdate).get("view")).toBe("all"));
      expect(screen.getByTestId("models-view-select")).toHaveTextContent("All Available Models");

      await user.click(screen.getByTestId("models-view-select"));
      await user.click(await screen.findByRole("option", { name: "Current Team Models" }));

      await waitFor(() => expect(lastUrl(onUrlUpdate).has("view")).toBe(false));
    });

    it("writes header sorting to ?sort_by= and ?sort_order= and clears both when unsorted", async () => {
      const user = userEvent.setup();
      const { onUrlUpdate } = renderTab();

      await user.click(sortHeader("model_info_updated_at"));
      await waitFor(() => expect(lastUrl(onUrlUpdate).get("sort_by")).toBe("model_info_updated_at"));
      expect(lastUrl(onUrlUpdate).has("sort_order")).toBe(false);

      await user.click(sortHeader("model_info_updated_at"));
      await waitFor(() => expect(lastUrl(onUrlUpdate).get("sort_order")).toBe("desc"));

      await user.click(sortHeader("model_info_updated_at"));
      await waitFor(() => expect(lastUrl(onUrlUpdate).has("sort_by")).toBe(false));
      expect(lastUrl(onUrlUpdate).has("sort_order")).toBe(false);
    });

    it("writes the access group filter to ?filter_access_group= and drops it when the chip is removed", async () => {
      const user = userEvent.setup();
      const { onUrlUpdate } = renderTab();

      await user.click(screen.getByTestId("datatable-filters-trigger"));
      await user.click(await screen.findByPlaceholderText("Filter by Model Access Group"));
      await user.click(await screen.findByRole("option", { name: "sales-team" }));
      await user.click(screen.getByTestId("filter-drawer-apply"));

      await waitFor(() => expect(lastUrl(onUrlUpdate).get("filter_access_group")).toBe("sales-team"));

      await user.click(await screen.findByTestId("filter-chip-remove-model_info_access_groups"));

      await waitFor(() => expect(lastUrl(onUrlUpdate).has("filter_access_group")).toBe(false));
      expect(lastModelsInfoCall().accessGroup).toBeUndefined();
    });

    it("writes page and page size changes to ?page= and ?page_size=", async () => {
      const user = userEvent.setup();
      setModelsInfo([makeRow()], 200);
      const { onUrlUpdate } = renderTab();

      await user.click(screen.getByTestId("pagination-next"));
      await waitFor(() => expect(lastUrl(onUrlUpdate).get("page")).toBe("2"));
      await waitFor(() => expect(lastModelsInfoCall().page).toBe(2));

      await user.click(screen.getByTestId("pagination-page-size"));
      await user.click(await screen.findByRole("option", { name: "25" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate).get("page_size")).toBe("25"));
      await waitFor(() => expect(lastModelsInfoCall().size).toBe(25));
    });

    it("clears every table key and the model group in a single URL update from the drawer reset", async () => {
      const user = userEvent.setup();
      setModelsInfo([makeRow()], 100);
      const { onUrlUpdate } = renderTab(
        { selectedModelGroup: "gpt-4" },
        "?model_search=claude&page=2&page_size=25&sort_by=model_name&sort_order=desc&filter_team=team-1&filter_access_group=sales-team&view=all",
      );
      const callsBefore = onUrlUpdate.mock.calls.length;

      await user.click(screen.getByTestId("datatable-filters-trigger"));
      await user.click(await screen.findByTestId("filter-drawer-reset"));

      await waitFor(() => expect(onUrlUpdate.mock.calls.length).toBe(callsBefore + 1));
      const url = lastUrl(onUrlUpdate);
      for (const key of [
        "model_search",
        "page",
        "page_size",
        "sort_by",
        "sort_order",
        "filter_team",
        "filter_access_group",
        "view",
      ]) {
        expect(url.has(key)).toBe(false);
      }
      expect(mockSetSelectedModelGroup).toHaveBeenCalledWith("all");
      await waitFor(() => expect(lastModelsInfoCall()).toMatchObject({ page: 1, size: 50 }));
      expect(lastModelsInfoCall().teamId).toBeUndefined();
      expect(lastModelsInfoCall().sortBy).toBeUndefined();
      expect(lastModelsInfoCall().accessGroup).toBeUndefined();
    });
  });

  it("queries the selected team and resets to the first page", async () => {
    const user = userEvent.setup();
    renderTab();

    expect(lastModelsInfoCall().teamId).toBeUndefined();

    await user.click(screen.getByTestId("models-team-select"));
    await user.click(await screen.findByRole("option", { name: "Engineering" }));

    await waitFor(() => {
      expect(lastModelsInfoCall().teamId).toBe("team-1");
    });
    expect(lastModelsInfoCall().page).toBe(1);
  });

  it("debounces the model name search into the server query", async () => {
    renderTab();

    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "claude" } });

    await waitFor(() => {
      expect(lastModelsInfoCall().search).toBe("claude");
    });
  });

  it("applies a public model name filter through the drawer", async () => {
    const user = userEvent.setup();
    renderTab();

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await user.click(await screen.findByPlaceholderText("Filter by Public Model Name"));
    await user.click(await screen.findByRole("option", { name: "gpt-3.5-turbo" }));
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => {
      expect(mockSetSelectedModelGroup).toHaveBeenCalledWith("gpt-3.5-turbo");
    });
  });

  it("renders every row the server returned for the selected model group so rows match the footer total", () => {
    setModelsInfo([makeRow(), { ...makeRow({ model_info: { id: "model-2" } }), model_name: "claude-opus" }], 2);
    renderTab({ selectedModelGroup: "claude-opus" });

    const table = screen.getByRole("table");
    expect(within(table).getByText("claude-opus")).toBeInTheDocument();
    expect(within(table).getByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-2 of 2");
  });

  it("asks the server for wildcard deployments instead of hiding rows client-side", () => {
    setModelsInfo([makeRow(), { ...makeRow({ model_info: { id: "model-2" } }), model_name: "openai/*" }], 2);
    renderTab({ selectedModelGroup: "wildcard" });

    expect(lastModelsInfoCall().wildcardOnly).toBe(true);
    expect(within(screen.getByRole("table")).getByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-2 of 2");
  });

  it("asks the server for the selected access group instead of hiding rows client-side", async () => {
    const user = userEvent.setup();
    renderTab();
    expect(lastModelsInfoCall().wildcardOnly).toBe(false);

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await user.click(await screen.findByPlaceholderText("Filter by Model Access Group"));
    await user.click(await screen.findByRole("option", { name: "sales-team" }));
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(lastModelsInfoCall().accessGroup).toBe("sales-team"));
    expect(within(screen.getByRole("table")).getByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-1 of 1");
  });

  it("asks the server for the exact selected model group so deployments beyond the first page are found", () => {
    renderTab({ selectedModelGroup: "claude-opus" });

    expect(lastModelsInfoCall().modelName).toBe("claude-opus");
    expect(lastModelsInfoCall().search).toBeUndefined();
  });

  it.each(["all", "wildcard"])("sends no exact model name for the %s pseudo group", (group) => {
    renderTab({ selectedModelGroup: group });

    expect(lastModelsInfoCall().modelName).toBeUndefined();
  });

  it("shows no filter chip for the all pseudo group", () => {
    renderTab({ selectedModelGroup: "all" });

    expect(screen.queryByTestId("filter-chip-model_name")).not.toBeInTheDocument();
  });

  it("keeps the exact model group alongside a typed search", async () => {
    renderTab({ selectedModelGroup: "claude-opus" });

    fireEvent.change(screen.getByPlaceholderText("Search model names…"), { target: { value: "opus" } });

    await waitFor(() => expect(lastModelsInfoCall().search).toBe("opus"));
    expect(lastModelsInfoCall().modelName).toBe("claude-opus");
  });

  it("resets search, filters, team and sorting from the drawer reset button", async () => {
    const user = userEvent.setup();
    renderTab({ selectedModelGroup: "gpt-4" });

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
    renderTab();

    await user.click(await screen.findByTestId("model-delete-model-1"));
    expect(await screen.findByText("Delete Model")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^delete$/i }));

    await waitFor(() => {
      expect(mockModelDeleteCall).toHaveBeenCalledWith("mock-access-token", "model-1");
    });
  });

  it("pauses a model through the row toggle", async () => {
    const user = userEvent.setup();
    renderTab();

    await user.click(await screen.findByTestId("model-pause-toggle-model-1"));

    await waitFor(() => {
      expect(mockModelPatchUpdateCall).toHaveBeenCalledWith("mock-access-token", { blocked: true }, "model-1");
    });
  });

  it("opens the model settings modal from the toolbar", async () => {
    const user = userEvent.setup();
    renderTab();

    expect(screen.queryByTestId("model-settings-modal")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("models-settings-trigger"));
    expect(screen.getByTestId("model-settings-modal")).toBeInTheDocument();
  });

  it("opens the model detail view from the model ID cell", async () => {
    const user = userEvent.setup();
    renderTab();

    await user.click(await screen.findByTestId("model-id-model-1"));

    expect(mockSetSelectedModelId).toHaveBeenCalledWith("model-1");
  });

  it("opens the team detail view from the team ID cell", async () => {
    const user = userEvent.setup();
    renderTab();

    await user.click(await screen.findByTestId("model-team-id-model-1"));

    expect(mockSetSelectedTeamId).toHaveBeenCalledWith("team-1");
  });

  describe("virtual key hint", () => {
    it("explains personal key creation while viewing current team models", () => {
      renderTab();

      expect(screen.getByText(/create a Virtual Key without selecting a team/i)).toBeInTheDocument();
    });

    it("links the Virtual Keys page through the migrated /ui route", () => {
      renderTab();

      expect(screen.getByRole("link", { name: "Virtual Keys page" })).toHaveAttribute("href", "/ui/api-keys");
    });

    it("links the team hint's Virtual Keys page through the migrated /ui route", async () => {
      const user = userEvent.setup();
      renderTab();

      await user.click(screen.getByTestId("models-team-select"));
      await user.click(await screen.findByRole("option", { name: "Engineering" }));

      await screen.findByText(/select Team as "Engineering"/i);
      expect(screen.getByRole("link", { name: "Virtual Keys page" })).toHaveAttribute("href", "/ui/api-keys");
    });

    it("names the selected team in the hint", async () => {
      const user = userEvent.setup();
      renderTab();

      await user.click(screen.getByTestId("models-team-select"));
      await user.click(await screen.findByRole("option", { name: "Engineering" }));

      expect(await screen.findByText(/select Team as "Engineering"/i)).toBeInTheDocument();
    });

    it("hides the hint when viewing all available models", async () => {
      const user = userEvent.setup();
      renderTab();

      await user.click(screen.getByTestId("models-view-select"));
      await user.click(await screen.findByRole("option", { name: "All Available Models" }));

      await waitFor(() => {
        expect(screen.queryByText(/create a Virtual Key/i)).not.toBeInTheDocument();
      });
    });
  });
});
