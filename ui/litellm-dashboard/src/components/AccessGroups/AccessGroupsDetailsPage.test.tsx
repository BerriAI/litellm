import { useAccessGroupDetails } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroupDetails";
import { AccessGroupResponse } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { AccessGroupDetail } from "./AccessGroupsDetailsPage";

vi.mock("@/app/(dashboard)/hooks/accessGroups/useAccessGroupDetails");
vi.mock("./AccessGroupsModal/AccessGroupEditModal", () => ({
  AccessGroupEditModal: ({
    visible,
    onCancel,
  }: {
    visible: boolean;
    onCancel: () => void;
  }) =>
    visible ? (
      <div role="dialog" aria-label="Edit Access Group">
        <button onClick={onCancel}>Close Modal</button>
      </div>
    ) : null,
}));

const mockUseAccessGroupDetails = vi.mocked(useAccessGroupDetails);

const baseMockReturnValue = {
  data: undefined,
  isLoading: false,
  isError: false,
  error: null,
  isFetching: false,
  isPending: false,
  isSuccess: true,
  status: "success" as const,
  dataUpdatedAt: 0,
  errorUpdatedAt: 0,
  failureCount: 0,
  failureReason: null,
  errorUpdateCount: 0,
  isFetched: true,
  isFetchedAfterMount: true,
  isRefetching: false,
  isLoadingError: false,
  isPaused: false,
  isPlaceholderData: false,
  isRefetchError: false,
  isStale: false,
  fetchStatus: "idle" as const,
  refetch: vi.fn(),
} as unknown as ReturnType<typeof useAccessGroupDetails>;

const createMockAccessGroup = (
  overrides: Partial<AccessGroupResponse> = {}
): AccessGroupResponse => ({
  access_group_id: "ag-1",
  access_group_name: "Test Group",
  description: "A test access group",
  access_model_names: ["model-1", "model-2"],
  access_mcp_server_ids: ["mcp-1"],
  access_agent_ids: ["agent-1"],
  assigned_team_ids: ["team-1"],
  assigned_key_ids: ["key-1", "key-2"],
  created_at: "2025-01-01T00:00:00Z",
  created_by: null,
  updated_at: "2025-01-02T00:00:00Z",
  updated_by: null,
  ...overrides,
});

describe("AccessGroupDetail", () => {
  const mockOnBack = vi.fn();
  const accessGroupId = "ag-1";

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: createMockAccessGroup(),
    } as ReturnType<typeof useAccessGroupDetails>);
  });

  it("should render the component", () => {
    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );
    expect(screen.getByRole("heading", { name: "Test Group" })).toBeInTheDocument();
  });

  it("should not show access group content when loading", () => {
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useAccessGroupDetails>);

    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.queryByRole("heading", { name: "Test Group" })).not.toBeInTheDocument();
  });

  it("should show empty state when access group is not found", () => {
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: undefined,
      isLoading: false,
    } as ReturnType<typeof useAccessGroupDetails>);

    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.getByText("Access group not found")).toBeInTheDocument();
    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("should call onBack when back button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    const buttons = screen.getAllByRole("button");
    const backButton = buttons.find((btn) => !btn.textContent?.includes("Edit"));
    await user.click(backButton!);

    expect(mockOnBack).toHaveBeenCalledTimes(1);
  });

  it("should display access group name and ID", () => {
    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.getByRole("heading", { name: "Test Group" })).toBeInTheDocument();
    expect(screen.getByText(/ID:/)).toBeInTheDocument();
  });

  it("should display description in Group Details", () => {
    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.getByText("Group Details")).toBeInTheDocument();
    expect(screen.getByText("A test access group")).toBeInTheDocument();
  });

  it("should display em dash when description is empty", () => {
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: createMockAccessGroup({ description: null }),
    } as ReturnType<typeof useAccessGroupDetails>);

    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("should open edit modal when Edit Access Group button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.queryByRole("dialog", { name: "Edit Access Group" })).not.toBeInTheDocument();

    const editButton = screen.getByRole("button", { name: /Edit Access Group/i });
    await user.click(editButton);

    expect(screen.getByRole("dialog", { name: "Edit Access Group" })).toBeInTheDocument();
  });

  it("should close edit modal when Close Modal is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    await user.click(screen.getByRole("button", { name: /Edit Access Group/i }));
    expect(screen.getByRole("dialog", { name: "Edit Access Group" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Close Modal" }));
    expect(screen.queryByRole("dialog", { name: "Edit Access Group" })).not.toBeInTheDocument();
  });

  it("should display attached keys", () => {
    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.getByText("Attached Keys")).toBeInTheDocument();
    expect(screen.getByText("key-1")).toBeInTheDocument();
    expect(screen.getByText("key-2")).toBeInTheDocument();
  });

  it("should display attached teams", () => {
    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.getByText("Attached Teams")).toBeInTheDocument();
    expect(screen.getByText("team-1")).toBeInTheDocument();
  });

  it("should show View All button for keys when more than 5", () => {
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: createMockAccessGroup({
        assigned_key_ids: ["k1", "k2", "k3", "k4", "k5", "k6"],
      }),
    } as ReturnType<typeof useAccessGroupDetails>);

    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.getByRole("button", { name: "View All (6)" })).toBeInTheDocument();
  });

  it("should toggle between View All and Show Less for keys", async () => {
    const user = userEvent.setup();
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: createMockAccessGroup({
        assigned_key_ids: ["k1", "k2", "k3", "k4", "k5", "k6"],
      }),
    } as ReturnType<typeof useAccessGroupDetails>);

    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    await user.click(screen.getByRole("button", { name: "View All (6)" }));
    expect(screen.getByRole("button", { name: "Show Less" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Show Less" }));
    expect(screen.getByRole("button", { name: "View All (6)" })).toBeInTheDocument();
  });

  it("should show View All button for teams when more than 5", () => {
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: createMockAccessGroup({
        assigned_team_ids: ["t1", "t2", "t3", "t4", "t5", "t6"],
      }),
    } as ReturnType<typeof useAccessGroupDetails>);

    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.getByRole("button", { name: "View All (6)" })).toBeInTheDocument();
  });

  it("should show empty state when no keys attached", () => {
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: createMockAccessGroup({ assigned_key_ids: [] }),
    } as ReturnType<typeof useAccessGroupDetails>);

    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.getByText("No keys attached")).toBeInTheDocument();
  });

  it("should show empty state when no teams attached", () => {
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: createMockAccessGroup({ assigned_team_ids: [] }),
    } as ReturnType<typeof useAccessGroupDetails>);

    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.getByText("No teams attached")).toBeInTheDocument();
  });

  it("should display Models tab with model IDs", () => {
    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.getByRole("tab", { name: /Models/i })).toBeInTheDocument();
    expect(screen.getByText("model-1")).toBeInTheDocument();
    expect(screen.getByText("model-2")).toBeInTheDocument();
  });

  it("should display MCP Servers tab with server IDs", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    const mcpTab = screen.getByRole("tab", { name: /MCP Servers/i });
    expect(mcpTab).toBeInTheDocument();
    await user.click(mcpTab);
    expect(screen.getByText("mcp-1")).toBeInTheDocument();
  });

  it("should display Agents tab with agent IDs", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    const agentsTab = screen.getByRole("tab", { name: /Agents/i });
    expect(agentsTab).toBeInTheDocument();
    await user.click(agentsTab);
    expect(screen.getByText("agent-1")).toBeInTheDocument();
  });

  it("should show empty state in Models tab when no models assigned", () => {
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: createMockAccessGroup({ access_model_names: [] }),
    } as ReturnType<typeof useAccessGroupDetails>);

    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.getByText("No models assigned to this group")).toBeInTheDocument();
  });

  it("should show empty state in MCP Servers tab when none assigned", async () => {
    const user = userEvent.setup();
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: createMockAccessGroup({ access_mcp_server_ids: [] }),
    } as ReturnType<typeof useAccessGroupDetails>);

    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    await user.click(screen.getByRole("tab", { name: /MCP Servers/i }));
    expect(screen.getByText("No MCP servers assigned to this group")).toBeInTheDocument();
  });

  it("should show empty state in Agents tab when none assigned", async () => {
    const user = userEvent.setup();
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: createMockAccessGroup({ access_agent_ids: [] }),
    } as ReturnType<typeof useAccessGroupDetails>);

    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    await user.click(screen.getByRole("tab", { name: /Agents/i }));
    expect(screen.getByText("No agents assigned to this group")).toBeInTheDocument();
  });

  it("should truncate long key IDs with ellipsis", () => {
    const longKeyId = "a".repeat(25);
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: createMockAccessGroup({ assigned_key_ids: [longKeyId] }),
    } as ReturnType<typeof useAccessGroupDetails>);

    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.getByText(/a{10}\.\.\.a{6}/)).toBeInTheDocument();
  });

  it("should display created and last updated timestamps", () => {
    renderWithProviders(
      <AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />
    );

    expect(screen.getByText("Created")).toBeInTheDocument();
    expect(screen.getByText("Last Updated")).toBeInTheDocument();
  });
});
