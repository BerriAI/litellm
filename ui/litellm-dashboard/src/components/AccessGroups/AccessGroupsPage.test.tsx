import { renderWithProviders, screen, within } from "@/../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AccessGroupsPage } from "./AccessGroupsPage";
import type { AccessGroupResponse } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";

const mockAccessGroups: AccessGroupResponse[] = [
  {
    access_group_id: "ag-1",
    access_group_name: "Admin Group",
    description: "Administrators with full access",
    access_model_names: ["m1", "m2"],
    access_mcp_server_ids: ["s1"],
    access_agent_ids: ["a1"],
    assigned_team_ids: [],
    assigned_key_ids: [],
    created_at: "2024-01-15T10:00:00Z",
    created_by: "user-1",
    updated_at: "2024-01-20T12:00:00Z",
    updated_by: "user-1",
  },
  {
    access_group_id: "ag-2",
    access_group_name: "Read Only",
    description: "Read-only access to models",
    access_model_names: ["m1"],
    access_mcp_server_ids: [],
    access_agent_ids: [],
    assigned_team_ids: [],
    assigned_key_ids: [],
    created_at: "2024-01-10T09:00:00Z",
    created_by: null,
    updated_at: "2024-01-12T11:00:00Z",
    updated_by: null,
  },
];

const mockUseAccessGroups = vi.fn();
const mockUseDeleteAccessGroup = vi.fn();
const mockMutate = vi.fn();

vi.mock("@/app/(dashboard)/hooks/accessGroups/useAccessGroups", () => ({
  useAccessGroups: () => mockUseAccessGroups(),
}));

vi.mock("@/app/(dashboard)/hooks/accessGroups/useDeleteAccessGroup", () => ({
  useDeleteAccessGroup: () => mockUseDeleteAccessGroup(),
}));

vi.mock("./AccessGroupsDetailsPage", () => ({
  AccessGroupDetail: ({
    accessGroupId,
    onBack,
  }: {
    accessGroupId: string;
    onBack: () => void;
  }) => (
    <div data-testid="access-group-detail">
      <span>Detail for {accessGroupId}</span>
      <button onClick={onBack}>Back</button>
    </div>
  ),
}));

vi.mock("./AccessGroupsModal/AccessGroupCreateModal", () => ({
  AccessGroupCreateModal: ({
    visible,
    onCancel,
  }: {
    visible: boolean;
    onCancel: () => void;
  }) =>
    visible ? (
      <div data-testid="create-access-group-modal">
        <button onClick={onCancel}>Cancel</button>
      </div>
    ) : null,
}));

vi.mock("../common_components/IconActionButton/TableIconActionButtons/TableIconActionButton", () => ({
  default: ({
    variant,
    tooltipText,
    onClick,
  }: {
    variant: string;
    tooltipText: string;
    onClick: () => void;
  }) => (
    <button
      data-testid={`action-button-${variant.toLowerCase()}`}
      aria-label={tooltipText}
      onClick={onClick}
    >
      {variant}
    </button>
  ),
}));

describe("AccessGroupsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAccessGroups.mockReturnValue({
      data: mockAccessGroups,
      isLoading: false,
    });
    mockUseDeleteAccessGroup.mockReturnValue({
      mutate: mockMutate,
      isPending: false,
    });
  });

  it("should render", () => {
    renderWithProviders(<AccessGroupsPage />);
    expect(screen.getByRole("heading", { name: "Access Groups" })).toBeInTheDocument();
  });

  it("should display page title and subtitle", () => {
    renderWithProviders(<AccessGroupsPage />);
    expect(screen.getByRole("heading", { name: "Access Groups" })).toBeInTheDocument();
    expect(
      screen.getByText("Manage resource permissions for your organization"),
    ).toBeInTheDocument();
  });

  it("should display Create Access Group button", () => {
    renderWithProviders(<AccessGroupsPage />);
    expect(
      screen.getByRole("button", { name: /create access group/i }),
    ).toBeInTheDocument();
  });

  it("should display search input with placeholder", () => {
    renderWithProviders(<AccessGroupsPage />);
    expect(
      screen.getByPlaceholderText("Search groups by name, ID, or description..."),
    ).toBeInTheDocument();
  });

  it("should display access groups in table", () => {
    renderWithProviders(<AccessGroupsPage />);
    expect(screen.getByText("ag-1")).toBeInTheDocument();
    expect(screen.getByText("Admin Group")).toBeInTheDocument();
    expect(screen.getByText("ag-2")).toBeInTheDocument();
    expect(screen.getByText("Read Only")).toBeInTheDocument();
  });

  it("should display resource counts for each group", () => {
    renderWithProviders(<AccessGroupsPage />);
    const table = screen.getByRole("table");
    expect(table).toHaveTextContent("2");
    expect(table).toHaveTextContent("1");
  });

  it("should filter groups by search text matching name", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    const searchInput = screen.getByPlaceholderText(
      "Search groups by name, ID, or description...",
    );
    await user.type(searchInput, "Admin");
    expect(screen.getByText("Admin Group")).toBeInTheDocument();
    expect(screen.queryByText("Read Only")).not.toBeInTheDocument();
  });

  it("should filter groups by search text matching ID", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    const searchInput = screen.getByPlaceholderText(
      "Search groups by name, ID, or description...",
    );
    await user.type(searchInput, "ag-2");
    expect(screen.getByText("Read Only")).toBeInTheDocument();
    expect(screen.queryByText("Admin Group")).not.toBeInTheDocument();
  });

  it("should filter groups by search text matching description", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    const searchInput = screen.getByPlaceholderText(
      "Search groups by name, ID, or description...",
    );
    await user.type(searchInput, "read-only");
    expect(screen.getByText("Read Only")).toBeInTheDocument();
    expect(screen.queryByText("Admin Group")).not.toBeInTheDocument();
  });

  it("should reset to first page when search text changes", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    const searchInput = screen.getByPlaceholderText(
      "Search groups by name, ID, or description...",
    );
    await user.type(searchInput, "Admin");
    const pagination = screen.getByText(/groups/);
    expect(pagination).toHaveTextContent("1 groups");
  });

  it("should open create modal when Create Access Group button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    await user.click(screen.getByRole("button", { name: /create access group/i }));
    expect(screen.getByTestId("create-access-group-modal")).toBeInTheDocument();
  });

  it("should close create modal when cancel is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    await user.click(screen.getByRole("button", { name: /create access group/i }));
    expect(screen.getByTestId("create-access-group-modal")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.queryByTestId("create-access-group-modal")).not.toBeInTheDocument();
  });

  it("should navigate to detail view when group ID is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    await user.click(screen.getByText("ag-1"));
    expect(screen.getByTestId("access-group-detail")).toBeInTheDocument();
    expect(screen.getByText("Detail for ag-1")).toBeInTheDocument();
  });

  it("should return to list view when Back is clicked from detail", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    await user.click(screen.getByText("ag-1"));
    expect(screen.getByTestId("access-group-detail")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Back" }));
    expect(screen.queryByTestId("access-group-detail")).not.toBeInTheDocument();
    expect(screen.getByText("Admin Group")).toBeInTheDocument();
  });

  it("should open delete modal when delete action is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    const deleteButtons = screen.getAllByRole("button", {
      name: "Delete access group",
    });
    await user.click(deleteButtons[0]);
    const dialog = screen.getByRole("dialog", { name: "Delete Access Group" });
    expect(dialog).toBeInTheDocument();
    expect(
      within(dialog).getByText(
        "Are you sure you want to delete this access group? This action cannot be undone.",
      ),
    ).toBeInTheDocument();
    expect(within(dialog).getByText("Access Group Information")).toBeInTheDocument();
    expect(within(dialog).getByText("ag-1")).toBeInTheDocument();
    expect(within(dialog).getByText("Admin Group")).toBeInTheDocument();
  });

  it("should close delete modal when cancel is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    const deleteButtons = screen.getAllByRole("button", {
      name: "Delete access group",
    });
    await user.click(deleteButtons[0]);
    const dialog = screen.getByRole("dialog", { name: "Delete Access Group" });
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog", { name: "Delete Access Group" })).not.toBeInTheDocument();
  });

  it("should call delete mutation when delete is confirmed", async () => {
    const user = userEvent.setup();
    mockMutate.mockImplementation((_id: string, opts?: { onSuccess?: () => void }) => {
      opts?.onSuccess?.();
    });
    renderWithProviders(<AccessGroupsPage />);
    const deleteButtons = screen.getAllByRole("button", {
      name: "Delete access group",
    });
    await user.click(deleteButtons[0]);
    const dialog = screen.getByRole("dialog", { name: "Delete Access Group" });
    const deleteConfirmButton = within(dialog).getByRole("button", { name: /delete/i });
    await user.click(deleteConfirmButton);
    expect(mockMutate).toHaveBeenCalledWith("ag-1", expect.any(Object));
  });

  it("should display pagination with total count", () => {
    renderWithProviders(<AccessGroupsPage />);
    expect(screen.getByText("2 groups")).toBeInTheDocument();
  });

  it("should show table headers for ID, Name, Resources, and Actions", () => {
    renderWithProviders(<AccessGroupsPage />);
    expect(screen.getByRole("columnheader", { name: /ID/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Name/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Resources/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Actions/i })).toBeInTheDocument();
  });

  it("should display loading state when data is loading", () => {
    mockUseAccessGroups.mockReturnValue({
      data: undefined,
      isLoading: true,
    });
    renderWithProviders(<AccessGroupsPage />);
    const table = screen.getByRole("table");
    expect(table).toBeInTheDocument();
  });

  it("should display empty state when no groups match search", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    const searchInput = screen.getByPlaceholderText(
      "Search groups by name, ID, or description...",
    );
    await user.type(searchInput, "nonexistent-group-xyz");
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("should display empty data when useAccessGroups returns empty array", () => {
    mockUseAccessGroups.mockReturnValue({
      data: [],
      isLoading: false,
    });
    renderWithProviders(<AccessGroupsPage />);
    expect(screen.getByRole("table")).toBeInTheDocument();
  });
});
