import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useReactTable, getCoreRowModel, flexRender } from "@tanstack/react-table";
import { columns } from "./columns";
import { ModelData } from "../../model_dashboard/types";
import { Table, TableHead, TableHeaderCell, TableBody, TableRow, TableCell } from "@tremor/react";
import * as providerInfoHelpers from "../../provider_info_helpers";

vi.mock("../../provider_info_helpers");

vi.mock("@tremor/react", async (importOriginal) => {
  const React = await import("react");
  const actual = await importOriginal<typeof import("@tremor/react")>();
  const IconComponent = React.forwardRef<HTMLButtonElement, any>(({ icon: IconComp, onClick, className, ...props }, ref) => {
    const ariaLabel = className?.includes("cursor-not-allowed")
      ? "Config model cannot be deleted on the dashboard. Please delete it from the config file."
      : "Delete model";
    return React.createElement(
      "button",
      { ...props, onClick, className, ref, "aria-label": ariaLabel },
      IconComp && React.createElement(IconComp, { className: "w-4 h-4" }),
    );
  });
  IconComponent.displayName = "Icon";
  return {
    ...actual,
    Icon: IconComponent,
  };
});

const createMockModel = (overrides: Partial<ModelData> = {}): ModelData => ({
  model_info: {
    id: "test-model-id",
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-02T00:00:00Z",
    created_by: "test-user",
    team_id: "test-team-id",
    db_model: true,
    access_groups: ["group1"],
  },
  model_name: "test-model",
  provider: "openai",
  litellm_model_name: "gpt-4",
  input_cost: 0.01,
  output_cost: 0.03,
  max_tokens: 4096,
  max_input_tokens: 8192,
  litellm_params: {
    model: "gpt-4",
    litellm_credential_name: "test-credential",
  },
  cleanedLitellmParams: {},
  ...overrides,
});

const TestTable = ({
  data,
  columns: cols,
}: {
  data: ModelData[];
  columns: ReturnType<typeof columns>;
}) => {
  const table = useReactTable({
    data,
    columns: cols,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <Table>
      <TableHead>
        {table.getHeaderGroups().map((headerGroup) => (
          <TableRow key={headerGroup.id}>
            {headerGroup.headers.map((header) => (
              <TableHeaderCell key={header.id}>
                {header.isPlaceholder
                  ? null
                  : flexRender(header.column.columnDef.header, header.getContext())}
              </TableHeaderCell>
            ))}
          </TableRow>
        ))}
      </TableHead>
      <TableBody>
        {table.getRowModel().rows.map((row) => (
          <TableRow key={row.id}>
            {row.getVisibleCells().map((cell) => (
              <TableCell key={cell.id}>
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
};

describe("columns", () => {
  beforeEach(() => {
    vi.mocked(providerInfoHelpers.getProviderLogoAndName).mockImplementation((provider: string) => {
      const providerMap: Record<string, { displayName: string; logo: string }> = {
        openai: { displayName: "OpenAI", logo: "/openai-logo.png" },
        anthropic: { displayName: "Anthropic", logo: "/anthropic-logo.png" },
      };
      return providerMap[provider] || { displayName: provider || "Unknown provider", logo: "" };
    });
  });

  const defaultProps = {
    userRole: "Admin",
    userID: "test-user",
    premiumUser: false,
    setSelectedModelId: vi.fn(),
    setSelectedTeamId: vi.fn(),
    getDisplayModelName: vi.fn((model: ModelData) => model.model_name || "-"),
    handleEditClick: vi.fn(),
    handleRefreshClick: vi.fn(),
    expandedRows: new Set<string>(),
    setExpandedRows: vi.fn(),
  };

  it("should render columns with table structure", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel();
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("Model ID")).toBeInTheDocument();
    expect(screen.getByText("Model Information")).toBeInTheDocument();
    expect(screen.getByText("Credentials")).toBeInTheDocument();
    expect(screen.getByText("Created By")).toBeInTheDocument();
    expect(screen.getByText("Updated At")).toBeInTheDocument();
    expect(screen.getByText("Costs")).toBeInTheDocument();
    expect(screen.getByText("Team ID")).toBeInTheDocument();
    expect(screen.getByText("Model Access Group")).toBeInTheDocument();
    expect(screen.getByText("Status")).toBeInTheDocument();
    expect(screen.getByText("Actions")).toBeInTheDocument();
  });

  it("should display model information with provider logo", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_name: "GPT-4",
      provider: "openai",
      litellm_model_name: "gpt-4",
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("GPT-4")).toBeInTheDocument();
    expect(screen.getByText("gpt-4")).toBeInTheDocument();
  });

  it("should display credential name when available", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      litellm_params: {
        model: "gpt-4",
        litellm_credential_name: "my-credential",
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("my-credential")).toBeInTheDocument();
  });

  it("should display 'Manual' when credential name is missing", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      litellm_params: {
        model: "gpt-4",
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("Manual")).toBeInTheDocument();
  });

  describe("credentials column", () => {
    it("should display Credentials header with info icon", () => {
      const cols = columns(
        defaultProps.userRole,
        defaultProps.userID,
        defaultProps.premiumUser,
        defaultProps.setSelectedModelId,
        defaultProps.setSelectedTeamId,
        defaultProps.getDisplayModelName,
        defaultProps.handleEditClick,
        defaultProps.handleRefreshClick,
        defaultProps.expandedRows,
        defaultProps.setExpandedRows,
      );

      const model = createMockModel();
      render(<TestTable data={[model]} columns={cols} />);

      expect(screen.getByText("Credentials")).toBeInTheDocument();
      // Info icon is in a flex container with Credentials - ant icons render as span with role="img"
      const credentialsHeader = screen.getByText("Credentials").closest("span");
      expect(credentialsHeader?.parentElement?.querySelector('[role="img"]')).toBeInTheDocument();
    });

    it("should display reusable credential with SyncOutlined icon and credential name", () => {
      const cols = columns(
        defaultProps.userRole,
        defaultProps.userID,
        defaultProps.premiumUser,
        defaultProps.setSelectedModelId,
        defaultProps.setSelectedTeamId,
        defaultProps.getDisplayModelName,
        defaultProps.handleEditClick,
        defaultProps.handleRefreshClick,
        defaultProps.expandedRows,
        defaultProps.setExpandedRows,
      );

      const model = createMockModel({
        litellm_params: {
          model: "gpt-4",
          litellm_credential_name: "my-reusable-credential",
        },
      });
      render(<TestTable data={[model]} columns={cols} />);

      expect(screen.getByText("my-reusable-credential")).toBeInTheDocument();
      const credentialCell = screen.getByText("my-reusable-credential").closest("div");
      expect(credentialCell).toHaveClass("flex");
      expect(screen.getByText("my-reusable-credential")).toHaveClass("text-blue-600");
    });

    it("should display Manual with EditOutlined when no credential name", () => {
      const cols = columns(
        defaultProps.userRole,
        defaultProps.userID,
        defaultProps.premiumUser,
        defaultProps.setSelectedModelId,
        defaultProps.setSelectedTeamId,
        defaultProps.getDisplayModelName,
        defaultProps.handleEditClick,
        defaultProps.handleRefreshClick,
        defaultProps.expandedRows,
        defaultProps.setExpandedRows,
      );

      const model = createMockModel({
        litellm_params: {
          model: "gpt-4",
        },
      });
      render(<TestTable data={[model]} columns={cols} />);

      expect(screen.getByText("Manual")).toBeInTheDocument();
      expect(screen.getByText("Manual")).toHaveClass("text-gray-500");
    });

    it("should display Manual when litellm_params is undefined", () => {
      const cols = columns(
        defaultProps.userRole,
        defaultProps.userID,
        defaultProps.premiumUser,
        defaultProps.setSelectedModelId,
        defaultProps.setSelectedTeamId,
        defaultProps.getDisplayModelName,
        defaultProps.handleEditClick,
        defaultProps.handleRefreshClick,
        defaultProps.expandedRows,
        defaultProps.setExpandedRows,
      );

      const model = createMockModel({
        litellm_params: undefined as any,
      });
      render(<TestTable data={[model]} columns={cols} />);

      expect(screen.getByText("Manual")).toBeInTheDocument();
    });

    it("should display Manual when litellm_credential_name is empty string", () => {
      const cols = columns(
        defaultProps.userRole,
        defaultProps.userID,
        defaultProps.premiumUser,
        defaultProps.setSelectedModelId,
        defaultProps.setSelectedTeamId,
        defaultProps.getDisplayModelName,
        defaultProps.handleEditClick,
        defaultProps.handleRefreshClick,
        defaultProps.expandedRows,
        defaultProps.setExpandedRows,
      );

      const model = createMockModel({
        litellm_params: {
          model: "gpt-4",
          litellm_credential_name: "",
        },
      });
      render(<TestTable data={[model]} columns={cols} />);

      expect(screen.getByText("Manual")).toBeInTheDocument();
    });
  });

  it("should display created by information for DB models", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        db_model: true,
        created_by: "admin-user",
        created_at: "2024-01-15T10:30:00Z",
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("admin-user")).toBeInTheDocument();
  });

  it("should display 'Defined in config' for config models", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        db_model: false,
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("Defined in config")).toBeInTheDocument();
  });

  it("should display costs when available", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      input_cost: 0.01,
      output_cost: 0.03,
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("In: $0.01")).toBeInTheDocument();
    expect(screen.getByText("Out: $0.03")).toBeInTheDocument();
  });

  it("should display '-' when costs are missing", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      input_cost: undefined as any,
      output_cost: undefined as any,
    });
    render(<TestTable data={[model]} columns={cols} />);

    const costCells = screen.getAllByText("-");
    expect(costCells.length).toBeGreaterThan(0);
  });

  it("should display '-' when team ID is missing", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        team_id: "",
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    const teamIdCells = screen.getAllByText("-");
    expect(teamIdCells.length).toBeGreaterThan(0);
  });

  it("should display access groups", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        access_groups: ["group1", "group2"],
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("group1")).toBeInTheDocument();
    expect(screen.getByText("+1")).toBeInTheDocument();
  });

  it("should expand access groups when expand button is clicked", async () => {
    const user = userEvent.setup();
    const setExpandedRows = vi.fn();
    const expandedRows = new Set<string>();
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      expandedRows,
      setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        id: "model-with-groups",
        access_groups: ["group1", "group2", "group3"],
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    const expandButton = screen.getByText("+2");
    expect(expandButton).toBeInTheDocument();

    await user.click(expandButton);
    expect(setExpandedRows).toHaveBeenCalled();
  });

  it("should display '-' when access groups are empty", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        access_groups: null,
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    const emptyCells = screen.getAllByText("-");
    expect(emptyCells.length).toBeGreaterThan(0);
  });

  it("should display 'DB Model' status for DB models", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        db_model: true,
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("DB Model")).toBeInTheDocument();
  });

  it("should display 'Config Model' status for config models", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        db_model: false,
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("Config Model")).toBeInTheDocument();
  });

  it("should allow Admin to delete DB models", async () => {
    const user = userEvent.setup();
    const setSelectedModelId = vi.fn();
    const cols = columns(
      "Admin",
      "admin-user",
      defaultProps.premiumUser,
      setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        db_model: true,
        id: "deletable-model",
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    const deleteButton = screen.getByRole("button", { name: "Delete model" });
    expect(deleteButton).toBeInTheDocument();

    await user.click(deleteButton);
    expect(setSelectedModelId).toHaveBeenCalledWith("deletable-model");
  });

  it("should allow model creator to delete their own DB models", async () => {
    const user = userEvent.setup();
    const setSelectedModelId = vi.fn();
    const cols = columns(
      "User",
      "model-creator",
      defaultProps.premiumUser,
      setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        db_model: true,
        created_by: "model-creator",
        id: "user-model",
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    const deleteButton = screen.getByRole("button", { name: "Delete model" });
    expect(deleteButton).toBeInTheDocument();

    await user.click(deleteButton);
    expect(setSelectedModelId).toHaveBeenCalledWith("user-model");
  });


  it("should disable delete for config models", () => {
    const cols = columns(
      "Admin",
      "admin-user",
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        db_model: false,
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    const deleteButton = screen.getByRole("button", { name: /config model cannot be deleted/i });
    expect(deleteButton).toBeInTheDocument();
    expect(deleteButton).toHaveClass("cursor-not-allowed");
  });

  it("should display collapsed access groups with expand button", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      new Set<string>(),
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        access_groups: ["group1", "group2", "group3"],
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("group1")).toBeInTheDocument();
    expect(screen.getByText("+2")).toBeInTheDocument();
    expect(screen.queryByText("group2")).not.toBeInTheDocument();
    expect(screen.queryByText("group3")).not.toBeInTheDocument();
  });

  it("should display expanded access groups when expanded", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      new Set<string>(["test-model-id"]),
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        id: "test-model-id",
        access_groups: ["group1", "group2", "group3"],
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("group1")).toBeInTheDocument();
    expect(screen.getByText("group2")).toBeInTheDocument();
    expect(screen.getByText("group3")).toBeInTheDocument();
    expect(screen.getByText("−")).toBeInTheDocument();
  });

  it("should display single access group without expand button", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        access_groups: ["group1"],
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("group1")).toBeInTheDocument();
    expect(screen.queryByText(/\+/)).not.toBeInTheDocument();
  });


  it("should handle missing display name gracefully", () => {
    const getDisplayModelName = vi.fn(() => "");
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel();
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("-")).toBeInTheDocument();
  });

  it("should handle missing created_at date", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        created_at: "",
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("Unknown date")).toBeInTheDocument();
  });

  it("should handle missing updated_at date", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        updated_at: "",
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    const updatedAtCells = screen.getAllByText("-");
    expect(updatedAtCells.length).toBeGreaterThan(0);
  });

  it("should handle missing created_by for DB models", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      model_info: {
        ...createMockModel().model_info,
        db_model: true,
        created_by: "",
      },
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("Unknown")).toBeInTheDocument();
  });

  it("should display only input cost when output cost is missing", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      input_cost: 0.01,
      output_cost: undefined as any,
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("In: $0.01")).toBeInTheDocument();
    expect(screen.queryByText(/Out:/)).not.toBeInTheDocument();
  });

  it("should display only output cost when input cost is missing", () => {
    const cols = columns(
      defaultProps.userRole,
      defaultProps.userID,
      defaultProps.premiumUser,
      defaultProps.setSelectedModelId,
      defaultProps.setSelectedTeamId,
      defaultProps.getDisplayModelName,
      defaultProps.handleEditClick,
      defaultProps.handleRefreshClick,
      defaultProps.expandedRows,
      defaultProps.setExpandedRows,
    );

    const model = createMockModel({
      input_cost: undefined as any,
      output_cost: 0.03,
    });
    render(<TestTable data={[model]} columns={cols} />);

    expect(screen.getByText("Out: $0.03")).toBeInTheDocument();
    expect(screen.queryByText(/In:/)).not.toBeInTheDocument();
  });
});
