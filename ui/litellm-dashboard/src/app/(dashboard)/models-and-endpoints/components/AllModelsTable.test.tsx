import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ModelData } from "@/components/model_dashboard/types";

import { AllModelsTable } from "./AllModelsTable";

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: { success: vi.fn(), fromBackend: vi.fn() },
}));

const makeModel = (overrides: Partial<ModelData> = {}): ModelData =>
  ({
    model_name: "gpt-4-public",
    litellm_model_name: "openai/gpt-4",
    provider: "openai",
    input_cost: 30 as unknown as number,
    output_cost: 60 as unknown as number,
    max_tokens: 8192,
    max_input_tokens: 8192,
    litellm_params: { model: "openai/gpt-4" },
    cleanedLitellmParams: {},
    ...overrides,
    model_info: {
      id: "model-1",
      created_at: "2024-01-02T00:00:00Z",
      updated_at: "2024-03-04T00:00:00Z",
      created_by: "alice",
      team_id: "team-1",
      db_model: true,
      access_groups: null,
      ...(overrides.model_info ?? {}),
    },
  }) as ModelData;

const baseProps = {
  data: [makeModel()],
  rowCount: 1,
  isLoading: false,
  isRefreshing: false,
  onRefresh: vi.fn(),
  sorting: [],
  onSortingChange: vi.fn(),
  pagination: { pageIndex: 0, pageSize: 50 },
  onPaginationChange: vi.fn(),
  columnFilters: [],
  onColumnFiltersChange: vi.fn(),
  onResetFilters: vi.fn(),
  searchValue: "",
  onSearchChange: vi.fn(),
  teamOptions: [
    { value: "personal", label: "Personal" },
    { value: "team-1", label: "Engineering" },
  ],
  selectedTeamValue: "personal",
  onTeamChange: vi.fn(),
  isLoadingTeams: false,
  viewMode: "current_team" as const,
  onViewModeChange: vi.fn(),
  onOpenModelSettings: vi.fn(),
  availableModelGroups: ["gpt-4", "gpt-3.5-turbo"],
  availableModelAccessGroups: ["sales-team"],
  userRole: "Admin",
  userID: "alice",
  onModelIdClick: vi.fn(),
  onTeamIdClick: vi.fn(),
  onDeleteClick: vi.fn(),
  onTogglePauseClick: vi.fn(),
  pausingModelId: null,
};

const row = (modelId: string): HTMLElement => {
  const element = document.querySelector(`[data-row-id="${modelId}"]`);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`row ${modelId} not rendered`);
  }
  return element;
};

describe("AllModelsTable", () => {
  it("renders the nine design columns and hides Status behind the Columns menu", async () => {
    const user = userEvent.setup();
    render(<AllModelsTable {...baseProps} />);

    for (const header of [
      "Model ID",
      "Model Information",
      "Credentials",
      "Created By",
      "Updated At",
      "Costs",
      "Team ID",
      "Model Access Group",
      "Actions",
    ]) {
      expect(screen.getByRole("columnheader", { name: new RegExp(header, "i") })).toBeInTheDocument();
    }

    expect(screen.queryByRole("columnheader", { name: /^status$/i })).not.toBeInTheDocument();
    expect(screen.queryByText("DB Model")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /columns/i }));
    await user.click(await screen.findByRole("menuitemcheckbox", { name: /status/i }));

    expect(await screen.findByText("DB Model")).toBeInTheDocument();
  });

  it("opens the model detail from the model ID cell", async () => {
    const user = userEvent.setup();
    const onModelIdClick = vi.fn();
    render(<AllModelsTable {...baseProps} onModelIdClick={onModelIdClick} />);

    await user.click(screen.getByTestId("model-id-model-1"));

    expect(onModelIdClick).toHaveBeenCalledWith("model-1");
  });

  it("opens the team detail from the team ID cell", async () => {
    const user = userEvent.setup();
    const onTeamIdClick = vi.fn();
    render(<AllModelsTable {...baseProps} onTeamIdClick={onTeamIdClick} />);

    await user.click(screen.getByTestId("model-team-id-model-1"));

    expect(onTeamIdClick).toHaveBeenCalledWith("team-1");
  });

  it("shows a dash when the model has no team", () => {
    render(
      <AllModelsTable {...baseProps} data={[makeModel({ model_info: { team_id: "" } as ModelData["model_info"] })]} />,
    );

    expect(within(row("model-1")).getAllByText("-").length).toBeGreaterThan(0);
    expect(screen.queryByTestId("model-team-id-model-1")).not.toBeInTheDocument();
  });

  it("renders the model name over the litellm model name", () => {
    render(<AllModelsTable {...baseProps} />);

    const cell = screen.getByTestId("model-information-model-1");
    expect(within(cell).getByText("gpt-4-public")).toBeInTheDocument();
    expect(within(cell).getByText("openai/gpt-4")).toBeInTheDocument();
  });

  it("renders a reusable credential by name and falls back to Manual", () => {
    const { rerender } = render(
      <AllModelsTable
        {...baseProps}
        data={[makeModel({ litellm_params: { model: "openai/gpt-4", litellm_credential_name: "openai-prod" } })]}
      />,
    );
    expect(screen.getByText("openai-prod")).toBeInTheDocument();
    expect(screen.queryByText("Manual")).not.toBeInTheDocument();

    rerender(<AllModelsTable {...baseProps} data={[makeModel()]} />);
    expect(screen.getByText("Manual")).toBeInTheDocument();
  });

  it("shows 'Defined in config' for a config model and the creator for a DB model", () => {
    const { rerender } = render(<AllModelsTable {...baseProps} />);
    expect(screen.getByText("alice")).toBeInTheDocument();

    rerender(
      <AllModelsTable
        {...baseProps}
        data={[makeModel({ model_info: { db_model: false } as ModelData["model_info"] })]}
      />,
    );
    expect(screen.getByText("Defined in config")).toBeInTheDocument();
  });

  it("renders input and output costs and a dash when both are missing", () => {
    const { rerender } = render(<AllModelsTable {...baseProps} />);
    expect(screen.getByText("$30")).toBeInTheDocument();
    expect(screen.getByText("$60")).toBeInTheDocument();

    rerender(
      <AllModelsTable
        {...baseProps}
        data={[makeModel({ input_cost: null as unknown as number, output_cost: null as unknown as number })]}
      />,
    );
    expect(screen.queryByText(/^\$/)).not.toBeInTheDocument();
  });

  it("collapses extra access groups behind a +N more badge", () => {
    render(
      <AllModelsTable
        {...baseProps}
        data={[
          makeModel({
            model_info: { access_groups: ["sales-team", "eng-team", "growth"] } as ModelData["model_info"],
          }),
        ]}
      />,
    );

    expect(screen.getByText("sales-team")).toBeInTheDocument();
    expect(screen.getByText("+2 more")).toBeInTheDocument();
  });

  describe("pause / resume", () => {
    it("renders the toggle on for an active DB model and off for a blocked one", () => {
      const { rerender } = render(<AllModelsTable {...baseProps} />);
      expect(screen.getByTestId("model-pause-toggle-model-1")).toBeChecked();

      rerender(
        <AllModelsTable
          {...baseProps}
          data={[makeModel({ model_info: { blocked: true } as ModelData["model_info"] })]}
        />,
      );
      expect(screen.getByTestId("model-pause-toggle-model-1")).not.toBeChecked();
    });

    it("pauses an active model and resumes a blocked one", async () => {
      const user = userEvent.setup();
      const onTogglePauseClick = vi.fn();
      const { rerender } = render(<AllModelsTable {...baseProps} onTogglePauseClick={onTogglePauseClick} />);

      await user.click(screen.getByTestId("model-pause-toggle-model-1"));
      expect(onTogglePauseClick).toHaveBeenCalledWith("model-1", true);

      onTogglePauseClick.mockClear();
      rerender(
        <AllModelsTable
          {...baseProps}
          onTogglePauseClick={onTogglePauseClick}
          data={[makeModel({ model_info: { blocked: true } as ModelData["model_info"] })]}
        />,
      );

      await user.click(screen.getByTestId("model-pause-toggle-model-1"));
      expect(onTogglePauseClick).toHaveBeenCalledWith("model-1", false);
    });

    it("does not let a non-admin toggle a model", async () => {
      const user = userEvent.setup();
      const onTogglePauseClick = vi.fn();
      render(<AllModelsTable {...baseProps} userRole="Internal User" onTogglePauseClick={onTogglePauseClick} />);

      const toggle = screen.getByTestId("model-pause-toggle-model-1");
      expect(toggle).toHaveAttribute("data-disabled");
      await user.click(toggle);
      expect(onTogglePauseClick).not.toHaveBeenCalled();
    });

    it("does not let anyone toggle a config model", async () => {
      const user = userEvent.setup();
      const onTogglePauseClick = vi.fn();
      render(
        <AllModelsTable
          {...baseProps}
          onTogglePauseClick={onTogglePauseClick}
          data={[makeModel({ model_info: { db_model: false } as ModelData["model_info"] })]}
        />,
      );

      const toggle = screen.getByTestId("model-pause-toggle-model-1");
      expect(toggle).toHaveAttribute("data-disabled");
      await user.click(toggle);
      expect(onTogglePauseClick).not.toHaveBeenCalled();
    });

    it("replaces the toggle with a pending indicator while a PATCH is in flight", () => {
      render(<AllModelsTable {...baseProps} pausingModelId="model-1" />);

      expect(screen.getByTestId("model-pause-pending-model-1")).toBeInTheDocument();
      expect(screen.queryByTestId("model-pause-toggle-model-1")).not.toBeInTheDocument();
    });
  });

  describe("delete", () => {
    it("lets an admin delete a DB model", async () => {
      const user = userEvent.setup();
      const onDeleteClick = vi.fn();
      render(<AllModelsTable {...baseProps} userID="someone-else" onDeleteClick={onDeleteClick} />);

      await user.click(screen.getByTestId("model-delete-model-1"));
      expect(onDeleteClick).toHaveBeenCalledWith("model-1");
    });

    it("lets the creator delete their own DB model", async () => {
      const user = userEvent.setup();
      const onDeleteClick = vi.fn();
      render(<AllModelsTable {...baseProps} userRole="Internal User" userID="alice" onDeleteClick={onDeleteClick} />);

      await user.click(screen.getByTestId("model-delete-model-1"));
      expect(onDeleteClick).toHaveBeenCalledWith("model-1");
    });

    it("blocks deleting a model the user did not create", async () => {
      const user = userEvent.setup();
      const onDeleteClick = vi.fn();
      render(<AllModelsTable {...baseProps} userRole="Internal User" userID="bob" onDeleteClick={onDeleteClick} />);

      const deleteButton = screen.getByTestId("model-delete-model-1");
      expect(deleteButton).toBeDisabled();
      await user.click(deleteButton);
      expect(onDeleteClick).not.toHaveBeenCalled();
    });

    it("blocks deleting a config model", async () => {
      const user = userEvent.setup();
      const onDeleteClick = vi.fn();
      render(
        <AllModelsTable
          {...baseProps}
          onDeleteClick={onDeleteClick}
          data={[makeModel({ model_info: { db_model: false } as ModelData["model_info"] })]}
        />,
      );

      const deleteButton = screen.getByTestId("model-delete-model-1");
      expect(deleteButton).toBeDisabled();
      await user.click(deleteButton);
      expect(onDeleteClick).not.toHaveBeenCalled();
    });
  });

  describe("toolbar", () => {
    it("wires search, refresh, team, view and model settings", async () => {
      const user = userEvent.setup();
      const onSearchChange = vi.fn();
      const onRefresh = vi.fn();
      const onOpenModelSettings = vi.fn();
      render(
        <AllModelsTable
          {...baseProps}
          onSearchChange={onSearchChange}
          onRefresh={onRefresh}
          onOpenModelSettings={onOpenModelSettings}
        />,
      );

      await user.type(screen.getByTestId("datatable-search"), "gpt");
      expect(onSearchChange).toHaveBeenCalled();

      await user.click(screen.getByTestId("datatable-refresh"));
      expect(onRefresh).toHaveBeenCalled();

      await user.click(screen.getByTestId("models-settings-trigger"));
      expect(onOpenModelSettings).toHaveBeenCalled();

      expect(screen.getByTestId("models-team-select")).toHaveTextContent("Personal");
      expect(screen.getByTestId("models-view-select")).toHaveTextContent("Current Team Models");
    });

    it("switches the current team", async () => {
      const user = userEvent.setup();
      const onTeamChange = vi.fn();
      render(<AllModelsTable {...baseProps} onTeamChange={onTeamChange} />);

      await user.click(screen.getByTestId("models-team-select"));
      await user.click(await screen.findByRole("option", { name: "Engineering" }));

      expect(onTeamChange).toHaveBeenCalledWith("team-1");
    });

    it("runs the full reset from the filter drawer", async () => {
      const user = userEvent.setup();
      const onResetFilters = vi.fn();
      render(<AllModelsTable {...baseProps} onResetFilters={onResetFilters} />);

      await user.click(screen.getByTestId("datatable-filters-trigger"));
      await user.click(await screen.findByTestId("filter-drawer-reset"));

      expect(onResetFilters).toHaveBeenCalled();
    });

    it("renders active filters as removable chips", async () => {
      const user = userEvent.setup();
      const onColumnFiltersChange = vi.fn();
      render(
        <AllModelsTable
          {...baseProps}
          columnFilters={[{ id: "model_name", value: "wildcard" }]}
          onColumnFiltersChange={onColumnFiltersChange}
        />,
      );

      const chip = screen.getByTestId("filter-chip-model_name");
      expect(chip).toHaveTextContent("Public Model Name");
      expect(chip).toHaveTextContent("Wildcard Models (*)");

      await user.click(screen.getByTestId("filter-chip-remove-model_name"));
      expect(onColumnFiltersChange).toHaveBeenCalled();
    });
  });

  it("shows the server row count in the pagination footer", () => {
    render(<AllModelsTable {...baseProps} rowCount={137} />);

    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-50 of 137");
  });

  it("shows the empty state when there are no models", () => {
    render(<AllModelsTable {...baseProps} data={[]} rowCount={0} />);

    expect(screen.getByText("No models found")).toBeInTheDocument();
  });
});
