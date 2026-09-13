import { useAccessGroupDetails } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroupDetails";
import { AccessGroupResponse } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../../tests/test-utils";
import { AccessGroupDetail } from "./AccessGroupsDetailsPage";

vi.mock("@/app/(dashboard)/hooks/accessGroups/useAccessGroupDetails");
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("./AccessGroupsModal/AccessGroupEditModal", () => ({
  AccessGroupEditModal: ({ visible, onCancel }: { visible: boolean; onCancel: () => void }) =>
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

const unnamed = (ids: readonly string[]) => ids.map((id) => ({ id, name: null }));

const createMockAccessGroup = (overrides: Partial<AccessGroupResponse> = {}): AccessGroupResponse => ({
  access_group_id: "ag-1",
  access_group_name: "Test Group",
  description: "A test access group",
  access_model_names: ["model-1", "model-2"],
  access_mcp_server_ids: ["mcp-1"],
  access_agent_ids: ["agent-1"],
  assigned_team_ids: ["team-1"],
  assigned_key_ids: ["key-1", "key-2"],
  access_mcp_servers: [{ id: "mcp-1", name: "GitHub MCP" }],
  access_agents: [{ id: "agent-1", name: "Support Agent" }],
  assigned_teams: [{ id: "team-1", name: "Platform Team" }],
  assigned_keys: [
    { id: "key-1", name: "ci-key" },
    { id: "key-2", name: null },
  ],
  created_at: "2025-01-01T00:00:00Z",
  created_by: null,
  updated_at: "2025-01-02T00:00:00Z",
  updated_by: null,
  ...overrides,
});

const renderWith = (overrides: Partial<AccessGroupResponse> = {}) => {
  mockUseAccessGroupDetails.mockReturnValue({
    ...baseMockReturnValue,
    data: createMockAccessGroup(overrides),
  } as ReturnType<typeof useAccessGroupDetails>);
  return renderWithProviders(<AccessGroupDetail accessGroupId="ag-1" onBack={vi.fn()} />);
};

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
    renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);
    expect(screen.getByRole("heading", { name: "Test Group" })).toBeInTheDocument();
  });

  it("should not show access group content when loading", () => {
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useAccessGroupDetails>);

    renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

    expect(screen.queryByRole("heading", { name: "Test Group" })).not.toBeInTheDocument();
  });

  it("should show empty state when access group is not found", () => {
    mockUseAccessGroupDetails.mockReturnValue({
      ...baseMockReturnValue,
      data: undefined,
      isLoading: false,
    } as ReturnType<typeof useAccessGroupDetails>);

    renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

    expect(screen.getByText("Access group not found")).toBeInTheDocument();
    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("should call onBack when back button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

    await user.click(screen.getByRole("button", { name: "Back" }));

    expect(mockOnBack).toHaveBeenCalledTimes(1);
  });

  it("should display access group name and ID", () => {
    renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

    expect(screen.getByRole("heading", { name: "Test Group" })).toBeInTheDocument();
    expect(screen.getByText(/ID:/)).toBeInTheDocument();
  });

  it("should display description in Group Details", () => {
    renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

    expect(screen.getByText("Group Details")).toBeInTheDocument();
    expect(screen.getByText("A test access group")).toBeInTheDocument();
  });

  it("should display em dash when description is empty", () => {
    renderWith({ description: null });

    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("should open edit modal when Edit Access Group button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

    expect(screen.queryByRole("dialog", { name: "Edit Access Group" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Edit Access Group/i }));

    expect(screen.getByRole("dialog", { name: "Edit Access Group" })).toBeInTheDocument();
  });

  it("should close edit modal when Close Modal is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

    await user.click(screen.getByRole("button", { name: /Edit Access Group/i }));
    expect(screen.getByRole("dialog", { name: "Edit Access Group" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Close Modal" }));
    expect(screen.queryByRole("dialog", { name: "Edit Access Group" })).not.toBeInTheDocument();
  });

  describe("attached keys", () => {
    it("should show the key alias and hide the token when the key has an alias", () => {
      renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

      expect(screen.getByText("Attached Keys")).toBeInTheDocument();
      expect(screen.getByText("ci-key")).toBeInTheDocument();
      expect(screen.queryByText("key-1")).not.toBeInTheDocument();
    });

    it("should fall back to the token when the key has no alias", () => {
      renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

      expect(screen.getByText("key-2")).toBeInTheDocument();
    });

    it("should link each key to its detail page", () => {
      renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

      expect(screen.getByRole("link", { name: "ci-key" })).toHaveAttribute(
        "href",
        expect.stringContaining("key=key-1"),
      );
      expect(screen.getByRole("link", { name: "key-2" })).toHaveAttribute("href", expect.stringContaining("key=key-2"));
    });

    it("should reveal the token in a tooltip when hovering an aliased key", async () => {
      const user = userEvent.setup();
      renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

      await user.hover(screen.getByText("ci-key"));

      expect(await screen.findByText("key-1")).toBeInTheDocument();
    });

    it("should show View All button for keys when more than 5", () => {
      renderWith({ assigned_keys: unnamed(["k1", "k2", "k3", "k4", "k5", "k6"]) });

      expect(screen.getByRole("button", { name: "View All (6)" })).toBeInTheDocument();
      expect(screen.queryByText("k6")).not.toBeInTheDocument();
    });

    it("should toggle between View All and Show Less for keys", async () => {
      const user = userEvent.setup();
      renderWith({ assigned_keys: unnamed(["k1", "k2", "k3", "k4", "k5", "k6"]) });

      await user.click(screen.getByRole("button", { name: "View All (6)" }));
      expect(screen.getByRole("button", { name: "Show Less" })).toBeInTheDocument();
      expect(screen.getByText("k6")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Show Less" }));
      expect(screen.getByRole("button", { name: "View All (6)" })).toBeInTheDocument();
    });

    it("should show empty state when no keys attached", () => {
      renderWith({ assigned_keys: [] });

      expect(screen.getByText("No keys attached")).toBeInTheDocument();
    });

    it("should truncate long unaliased tokens with ellipsis", () => {
      renderWith({ assigned_keys: unnamed(["a".repeat(25)]) });

      expect(screen.getByText(/^a{10}\.\.\.a{6}$/)).toBeInTheDocument();
    });

    it("should not truncate a long alias", () => {
      const alias = "b".repeat(25);
      renderWith({ assigned_keys: [{ id: "a".repeat(25), name: alias }] });

      expect(screen.getByText(alias)).toBeInTheDocument();
    });
  });

  describe("attached teams", () => {
    it("should show the team alias and hide the id when the team has an alias", () => {
      renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

      expect(screen.getByText("Attached Teams")).toBeInTheDocument();
      expect(screen.getByText("Platform Team")).toBeInTheDocument();
      expect(screen.queryByText("team-1")).not.toBeInTheDocument();
    });

    it("should link each team to its detail page", () => {
      renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

      expect(screen.getByRole("link", { name: "Platform Team" })).toHaveAttribute(
        "href",
        expect.stringContaining("team=team-1"),
      );
    });

    it("should reveal the team id in a tooltip when hovering an aliased team", async () => {
      const user = userEvent.setup();
      renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

      await user.hover(screen.getByText("Platform Team"));

      expect(await screen.findByText("team-1")).toBeInTheDocument();
    });

    it("should fall back to the team id when the team has no alias", () => {
      renderWith({ assigned_teams: unnamed(["team-ghost"]) });

      expect(screen.getByText("team-ghost")).toBeInTheDocument();
    });

    it("should show View All button for teams when more than 5", () => {
      renderWith({ assigned_teams: unnamed(["t1", "t2", "t3", "t4", "t5", "t6"]) });

      expect(screen.getByRole("button", { name: "View All (6)" })).toBeInTheDocument();
    });

    it("should show empty state when no teams attached", () => {
      renderWith({ assigned_teams: [] });

      expect(screen.getByText("No teams attached")).toBeInTheDocument();
    });
  });

  it("should display Models tab with model names", () => {
    renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

    expect(screen.getByRole("tab", { name: /Models/i })).toBeInTheDocument();
    expect(screen.getByText("model-1")).toBeInTheDocument();
    expect(screen.getByText("model-2")).toBeInTheDocument();
  });

  describe("MCP Servers tab", () => {
    it("should show server names instead of ids", async () => {
      const user = userEvent.setup();
      renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

      await user.click(screen.getByRole("tab", { name: /MCP Servers/i }));

      expect(screen.getByText("GitHub MCP")).toBeInTheDocument();
      expect(screen.queryByText("mcp-1")).not.toBeInTheDocument();
    });

    it("should reveal the server id in a tooltip when hovering the name", async () => {
      const user = userEvent.setup();
      renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

      await user.click(screen.getByRole("tab", { name: /MCP Servers/i }));
      await user.hover(screen.getByText("GitHub MCP"));

      expect(await screen.findByText("mcp-1")).toBeInTheDocument();
    });

    it("should fall back to the id when the server has no name", async () => {
      const user = userEvent.setup();
      renderWith({ access_mcp_servers: unnamed(["mcp-deleted"]) });

      await user.click(screen.getByRole("tab", { name: /MCP Servers/i }));

      expect(screen.getByText("mcp-deleted")).toBeInTheDocument();
    });

    it("should show empty state when none assigned", async () => {
      const user = userEvent.setup();
      renderWith({ access_mcp_servers: [] });

      await user.click(screen.getByRole("tab", { name: /MCP Servers/i }));

      expect(screen.getByText("No MCP servers assigned to this group")).toBeInTheDocument();
    });
  });

  describe("Agents tab", () => {
    it("should show agent names instead of ids", async () => {
      const user = userEvent.setup();
      renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

      await user.click(screen.getByRole("tab", { name: /Agents/i }));

      expect(screen.getByText("Support Agent")).toBeInTheDocument();
      expect(screen.queryByText("agent-1")).not.toBeInTheDocument();
    });

    it("should fall back to the id when the agent has no name", async () => {
      const user = userEvent.setup();
      renderWith({ access_agents: unnamed(["agent-deleted"]) });

      await user.click(screen.getByRole("tab", { name: /Agents/i }));

      expect(screen.getByText("agent-deleted")).toBeInTheDocument();
    });

    it("should show empty state when none assigned", async () => {
      const user = userEvent.setup();
      renderWith({ access_agents: [] });

      await user.click(screen.getByRole("tab", { name: /Agents/i }));

      expect(screen.getByText("No agents assigned to this group")).toBeInTheDocument();
    });
  });

  it("should show empty state in Models tab when no models assigned", () => {
    renderWith({ access_model_names: [] });

    expect(screen.getByText("No models assigned to this group")).toBeInTheDocument();
  });

  it("should count resources from the resolved lists in the tab badges", () => {
    renderWith({
      access_mcp_servers: unnamed(["m1", "m2", "m3"]),
      access_agents: unnamed(["a1", "a2"]),
    });

    expect(screen.getByRole("tab", { name: /MCP Servers/i })).toHaveTextContent("3");
    expect(screen.getByRole("tab", { name: /Agents/i })).toHaveTextContent("2");
  });

  it("should display created and last updated timestamps", () => {
    renderWithProviders(<AccessGroupDetail accessGroupId={accessGroupId} onBack={mockOnBack} />);

    expect(screen.getByText("Created")).toBeInTheDocument();
    expect(screen.getByText("Last Updated")).toBeInTheDocument();
  });
});
