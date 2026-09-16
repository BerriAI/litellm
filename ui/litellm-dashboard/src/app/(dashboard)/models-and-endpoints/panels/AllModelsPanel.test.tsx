/* @vitest-environment jsdom */
import { renderWithProviders, screen, waitFor } from "@/../tests/test-utils";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, type Mock, vi } from "vitest";

import AllModelsPanel from "./AllModelsPanel";

vi.mock("@/components/networking", () => ({
  serverRootPath: "/",
  modelDeleteCall: vi.fn(),
  modelPatchUpdateCall: vi.fn(),
}));

vi.mock("@/components/model_dashboard/ModelSettingsModal/ModelSettingsModal", () => ({ default: () => null }));

interface ModelsInfoQuery {
  page?: number;
  modelName?: string;
  wildcardOnly?: boolean;
}

const modelsInfoCalls: ModelsInfoQuery[] = [];

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

const MODEL_ROW = {
  model_name: "gpt-4",
  litellm_params: { model: "openai/gpt-4", custom_llm_provider: "openai" },
  model_info: {
    id: "model-1",
    db_model: true,
    created_by: "user-123",
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-02T00:00:00Z",
    team_id: "team-1",
    access_groups: [],
  },
};

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useModelsInfo: (...args: UseModelsInfoArgs) => {
    const [page, , , , , , , , modelName, , wildcardOnly] = args;
    modelsInfoCalls.push({ page, modelName, wildcardOnly });
    return {
      data: { data: [MODEL_ROW], total_count: 100, current_page: page, total_pages: 2, size: 50 },
      isLoading: false,
      isFetching: false,
      isError: false,
      refetch: vi.fn(),
    };
  },
}));

vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({
  useModelCostMap: () => ({ data: {}, isLoading: false }),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: () => ({ data: [{ team_id: "team-1", team_alias: "Engineering" }], isLoading: false }),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    accessToken: "at",
    userId: "user-123",
    userRole: "Admin",
    isViewOnly: false,
  }),
}));

vi.mock("@/app/(dashboard)/models-and-endpoints/useModelDashboardData", () => ({
  useModelDashboardData: () => ({
    availableModelGroups: ["gpt-4", "gpt-3.5-turbo"],
    availableModelAccessGroups: ["sales-team"],
  }),
}));

const renderPanel = (searchParams?: string) => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  renderWithProviders(<AllModelsPanel />, { searchParams, onUrlUpdate });
  return { onUrlUpdate };
};

const lastUpdate = (onUrlUpdate: Mock<OnUrlUpdateFunction>) => {
  const event = onUrlUpdate.mock.calls.at(-1)?.[0];
  if (!event) throw new Error("no URL update was emitted");
  return event;
};

const lastModelsInfoCall = (): ModelsInfoQuery => modelsInfoCalls[modelsInfoCalls.length - 1];

const URL_SETTLE_MS = 100;

describe("AllModelsPanel", () => {
  beforeEach(() => {
    modelsInfoCalls.length = 0;
  });

  it("clears model_group together with every table key in one URL update from Reset Filters", async () => {
    const user = userEvent.setup();
    const { onUrlUpdate } = renderPanel(
      "?model_group=gpt-4&model_search=claude&page=2&page_size=25&sort_by=model_name&sort_order=desc&filter_team=team-1&access_group=sales-team&view_mode=all",
    );
    expect(lastModelsInfoCall().modelName).toBe("gpt-4");

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await user.click(await screen.findByTestId("filter-drawer-reset"));

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalledTimes(1));
    await new Promise((resolve) => setTimeout(resolve, URL_SETTLE_MS));
    expect(onUrlUpdate).toHaveBeenCalledTimes(1);
    expect(lastUpdate(onUrlUpdate).searchParams.toString()).toBe("");
    await waitFor(() => expect(lastModelsInfoCall()).toMatchObject({ page: 1, modelName: undefined }));
  });

  it("queries the exact model group named in ?model_group= and shows it as a chip", () => {
    renderPanel("?model_group=gpt-4");

    expect(lastModelsInfoCall()).toMatchObject({ modelName: "gpt-4", wildcardOnly: false });
    expect(screen.getByTestId("filter-chip-model_name")).toHaveTextContent("gpt-4");
  });

  it("treats ?model_group=wildcard as the wildcard-only sentinel", () => {
    renderPanel("?model_group=wildcard");

    expect(lastModelsInfoCall()).toMatchObject({ modelName: undefined, wildcardOnly: true });
    expect(screen.getByTestId("filter-chip-model_name")).toHaveTextContent("Wildcard Models (*)");
  });

  it("treats ?model_group=all as no model group filter", () => {
    renderPanel("?model_group=all");

    expect(lastModelsInfoCall()).toMatchObject({ modelName: undefined, wildcardOnly: false });
    expect(screen.queryByTestId("filter-chip-model_name")).not.toBeInTheDocument();
  });

  it("writes a drawer Public Model Name to ?model_group= and resets the page in the same update", async () => {
    const user = userEvent.setup();
    const { onUrlUpdate } = renderPanel("?page=2");

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await user.click(await screen.findByPlaceholderText("Filter by Public Model Name"));
    await user.click(await screen.findByRole("option", { name: "gpt-3.5-turbo" }));
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(lastUpdate(onUrlUpdate).searchParams.get("model_group")).toBe("gpt-3.5-turbo"));
    expect(lastUpdate(onUrlUpdate).searchParams.has("page")).toBe(false);
    await waitFor(() => expect(lastModelsInfoCall()).toMatchObject({ page: 1, modelName: "gpt-3.5-turbo" }));
  });

  it("drops ?model_group= when its filter chip is removed", async () => {
    const user = userEvent.setup();
    const { onUrlUpdate } = renderPanel("?model_group=gpt-4&model_search=claude");

    await user.click(screen.getByTestId("filter-chip-remove-model_name"));

    await waitFor(() => expect(lastUpdate(onUrlUpdate).searchParams.has("model_group")).toBe(false));
    expect(lastUpdate(onUrlUpdate).searchParams.get("model_search")).toBe("claude");
  });

  it("pushes ?model= onto history for the drill-in and keeps the table state for Back", async () => {
    const user = userEvent.setup();
    const { onUrlUpdate } = renderPanel("?model_search=claude&page=2&filter_team=team-1");

    await user.click(await screen.findByTestId("model-id-model-1"));

    await waitFor(() => expect(lastUpdate(onUrlUpdate).searchParams.get("model")).toBe("model-1"));
    const { searchParams, options } = lastUpdate(onUrlUpdate);
    expect(options.history).toBe("push");
    expect(searchParams.get("model_search")).toBe("claude");
    expect(searchParams.get("page")).toBe("2");
    expect(searchParams.get("filter_team")).toBe("team-1");
  });
});
