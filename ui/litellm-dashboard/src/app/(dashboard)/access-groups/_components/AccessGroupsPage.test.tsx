import { fireEvent, renderWithProviders, screen, within } from "@/../tests/test-utils";
import { waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
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
    access_mcp_servers: [{ id: "s1", name: "Server One" }],
    access_agents: [{ id: "a1", name: "Agent One" }],
    assigned_teams: [],
    assigned_keys: [],
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
    access_mcp_servers: [],
    access_agents: [],
    assigned_teams: [],
    assigned_keys: [],
    created_at: "2024-01-10T09:00:00Z",
    created_by: null,
    updated_at: "2024-01-12T11:00:00Z",
    updated_by: null,
  },
];

const mockUseAccessGroups = vi.fn();
const mockUseDeleteAccessGroup = vi.fn();
const mockMutate = vi.fn();
const mockUseAuthorized = vi.fn();

vi.mock("@/app/(dashboard)/hooks/accessGroups/useAccessGroups", () => ({
  useAccessGroups: () => mockUseAccessGroups(),
}));

vi.mock("@/app/(dashboard)/hooks/accessGroups/useDeleteAccessGroup", () => ({
  useDeleteAccessGroup: () => mockUseDeleteAccessGroup(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

vi.mock("./AccessGroupsDetailsPage", () => ({
  ACCESS_GROUP_DETAIL_TAB_KEY: "detail_tab",
  AccessGroupDetail: ({ accessGroupId, onBack }: { accessGroupId: string; onBack: () => void }) => (
    <div data-testid="access-group-detail">
      <span>Detail for {accessGroupId}</span>
      <button onClick={onBack}>Back</button>
    </div>
  ),
}));

vi.mock("./access-group-create/AccessGroupCreateDialog", () => ({
  AccessGroupCreateDialog: ({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) =>
    open ? (
      <div data-testid="create-access-group-modal">
        <button onClick={() => onOpenChange(false)}>Cancel</button>
      </div>
    ) : null,
}));

const makeGroups = (count: number): AccessGroupResponse[] =>
  Array.from({ length: count }, (_, index) => {
    const suffix = String(index + 1).padStart(2, "0");
    return {
      ...mockAccessGroups[0],
      access_group_id: `ag-${suffix}`,
      access_group_name: `Group ${suffix}`,
      description: `Group ${suffix} description`,
    };
  });

const SEARCH_PLACEHOLDER = "Search groups by name, ID, or description...";

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) => onUrlUpdate.mock.calls.at(-1)?.[0];

const renderedGroupIds = () =>
  screen
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).getAllByRole("cell")[0].textContent);

const openRowMenu = async (user: ReturnType<typeof userEvent.setup>, groupId: string) => {
  await user.click(screen.getByTestId(`access-group-actions-${groupId}`));
  return screen.findByTestId("access-group-action-delete");
};

describe("AccessGroupsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAccessGroups.mockReturnValue({ data: mockAccessGroups, isLoading: false });
    mockUseDeleteAccessGroup.mockReturnValue({ mutate: mockMutate, isPending: false });
    mockUseAuthorized.mockReturnValue({ userRole: "Admin", accessToken: "sk-test" });
  });

  it("renders the page title and subtitle", () => {
    renderWithProviders(<AccessGroupsPage />);
    expect(screen.getByRole("heading", { name: "Access Groups" })).toBeInTheDocument();
    expect(screen.getByText("Manage resource permissions for your organization")).toBeInTheDocument();
    expect(document.querySelector(".lucide-boxes")).not.toBeNull();
  });

  it("shows the Create Access Group button for an admin", () => {
    renderWithProviders(<AccessGroupsPage />);
    expect(screen.getByRole("button", { name: /create access group/i })).toBeInTheDocument();
  });

  it("renders every access group row", () => {
    renderWithProviders(<AccessGroupsPage />);
    expect(screen.getByText("ag-1")).toBeInTheDocument();
    expect(screen.getByText("Admin Group")).toBeInTheDocument();
    expect(screen.getByText("ag-2")).toBeInTheDocument();
    expect(screen.getByText("Read Only")).toBeInTheDocument();
  });

  it("renders resource counts for each group", () => {
    renderWithProviders(<AccessGroupsPage />);
    // ag-1 has 2 models, 1 mcp server, 1 agent.
    const adminRow = screen.getByText("ag-1").closest("tr") as HTMLElement;
    expect(within(adminRow).getByTitle("2 Models")).toHaveTextContent("2");
    expect(within(adminRow).getByTitle("1 MCP Servers")).toHaveTextContent("1");
    expect(within(adminRow).getByTitle("1 Agents")).toHaveTextContent("1");
  });

  it("shows the expected column headers", () => {
    renderWithProviders(<AccessGroupsPage />);
    expect(screen.getByRole("columnheader", { name: /^ID$/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Name/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Resources/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Created/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Updated/i })).toBeInTheDocument();
  });

  it("filters by name", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    fireEvent.change(screen.getByPlaceholderText("Search groups by name, ID, or description..."), {
      target: { value: "Admin" },
    });
    expect(screen.getByText("Admin Group")).toBeInTheDocument();
    expect(screen.queryByText("Read Only")).not.toBeInTheDocument();
  });

  it("filters by ID", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    fireEvent.change(screen.getByPlaceholderText("Search groups by name, ID, or description..."), {
      target: { value: "ag-2" },
    });
    expect(screen.getByText("Read Only")).toBeInTheDocument();
    expect(screen.queryByText("Admin Group")).not.toBeInTheDocument();
  });

  it("filters by description", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    fireEvent.change(screen.getByPlaceholderText("Search groups by name, ID, or description..."), {
      target: { value: "read-only" },
    });
    expect(screen.getByText("Read Only")).toBeInTheDocument();
    expect(screen.queryByText("Admin Group")).not.toBeInTheDocument();
  });

  it("shows the filtered empty state when nothing matches", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    fireEvent.change(screen.getByPlaceholderText("Search groups by name, ID, or description..."), {
      target: { value: "no-such-group" },
    });
    expect(screen.getByText("No matching access groups")).toBeInTheDocument();
    expect(screen.queryByText("Admin Group")).not.toBeInTheDocument();
  });

  it("shows the empty state when there are no groups", () => {
    mockUseAccessGroups.mockReturnValue({ data: [], isLoading: false });
    renderWithProviders(<AccessGroupsPage />);
    expect(screen.getByText("No access groups yet")).toBeInTheDocument();
  });

  it("renders loading skeletons on the initial load", () => {
    mockUseAccessGroups.mockReturnValue({ data: undefined, isLoading: true });
    renderWithProviders(<AccessGroupsPage />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("Admin Group")).not.toBeInTheDocument();
  });

  it("opens and closes the create modal", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    await user.click(screen.getByRole("button", { name: /create access group/i }));
    expect(screen.getByTestId("create-access-group-modal")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.queryByTestId("create-access-group-modal")).not.toBeInTheDocument();
  });

  it("opens the detail view when the ID cell is clicked and returns via Back", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    await user.click(screen.getByText("ag-1"));
    expect(screen.getByTestId("access-group-detail")).toBeInTheDocument();
    expect(screen.getByText("Detail for ag-1")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Back" }));
    expect(screen.queryByTestId("access-group-detail")).not.toBeInTheDocument();
    expect(screen.getByText("Admin Group")).toBeInTheDocument();
  });

  it("opens the delete modal from the row actions menu", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    await user.click(await openRowMenu(user, "ag-1"));
    const dialog = screen.getByRole("dialog", { name: "Delete Access Group" });
    expect(
      within(dialog).getByText("Are you sure you want to delete this access group? This action cannot be undone."),
    ).toBeInTheDocument();
    expect(within(dialog).getByText("Access Group Information")).toBeInTheDocument();
    expect(within(dialog).getByText("ag-1")).toBeInTheDocument();
    expect(within(dialog).getByText("Admin Group")).toBeInTheDocument();
  });

  it("closes the delete modal on cancel without deleting", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AccessGroupsPage />);
    await user.click(await openRowMenu(user, "ag-1"));
    const dialog = screen.getByRole("dialog", { name: "Delete Access Group" });
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "Delete Access Group" })).not.toBeInTheDocument();
    });
    expect(mockMutate).not.toHaveBeenCalled();
  });

  it("calls the delete mutation with the group ID when confirmed", async () => {
    const user = userEvent.setup();
    mockMutate.mockImplementation((_id: string, opts?: { onSuccess?: () => void }) => {
      opts?.onSuccess?.();
    });
    renderWithProviders(<AccessGroupsPage />);
    await user.click(await openRowMenu(user, "ag-1"));
    const dialog = screen.getByRole("dialog", { name: "Delete Access Group" });
    await user.click(within(dialog).getByRole("button", { name: /delete/i }));
    expect(mockMutate).toHaveBeenCalledWith("ag-1", expect.any(Object));
  });

  it("still shows matches when searching from a later page", async () => {
    const user = userEvent.setup();
    mockUseAccessGroups.mockReturnValue({ data: makeGroups(25), isLoading: false });
    renderWithProviders(<AccessGroupsPage />);

    await user.click(screen.getByTestId("pagination-next"));
    expect(screen.getByText("ag-11")).toBeInTheDocument();
    expect(screen.queryByText("ag-01")).not.toBeInTheDocument();

    // The only match lives on page 1, so the page index must reset or the table reads as empty.
    fireEvent.change(screen.getByPlaceholderText("Search groups by name, ID, or description..."), {
      target: { value: "ag-01" },
    });
    expect(await screen.findByText("ag-01")).toBeInTheDocument();
    expect(screen.queryByText("No matching access groups")).not.toBeInTheDocument();
  });

  it("hides the Create button and row actions for a non-admin", () => {
    mockUseAuthorized.mockReturnValue({ userRole: "Admin Viewer", accessToken: "sk-test" });
    renderWithProviders(<AccessGroupsPage />);
    expect(screen.queryByRole("button", { name: /create access group/i })).not.toBeInTheDocument();
    expect(screen.queryByTestId("access-group-actions-ag-1")).not.toBeInTheDocument();
    // The read-only view still lists the groups.
    expect(screen.getByText("Admin Group")).toBeInTheDocument();
  });

  describe("URL state", () => {
    it("opens the detail view for the group named in the URL", () => {
      renderWithProviders(<AccessGroupsPage />, { searchParams: "?group=ag-2" });
      expect(screen.getByText("Detail for ag-2")).toBeInTheDocument();
      expect(screen.queryByPlaceholderText(SEARCH_PLACEHOLDER)).not.toBeInTheDocument();
    });

    it("pushes the clicked group into the URL and resets the detail tab", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<AccessGroupsPage />, { searchParams: "?detail_tab=agents", onUrlUpdate });
      await user.click(screen.getByText("ag-1"));
      expect(lastUrl(onUrlUpdate)?.searchParams.get("group")).toBe("ag-1");
      expect(lastUrl(onUrlUpdate)?.searchParams.has("detail_tab")).toBe(false);
      expect(lastUrl(onUrlUpdate)?.options.history).toBe("push");
    });

    it("clears the group and its detail tab on Back but keeps the list search", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<AccessGroupsPage />, {
        searchParams: "?group=ag-1&detail_tab=mcp&group_search=Read",
        onUrlUpdate,
      });
      await user.click(screen.getByRole("button", { name: "Back" }));
      const url = lastUrl(onUrlUpdate);
      expect(url?.searchParams.has("group")).toBe(false);
      expect(url?.searchParams.has("detail_tab")).toBe(false);
      expect(url?.searchParams.get("group_search")).toBe("Read");
      expect(url?.options.history).toBe("push");
      expect(screen.getByPlaceholderText(SEARCH_PLACEHOLDER)).toHaveValue("Read");
      expect(screen.getByText("Read Only")).toBeInTheDocument();
      expect(screen.queryByText("Admin Group")).not.toBeInTheDocument();
    });

    it("filters by the search in the URL", () => {
      renderWithProviders(<AccessGroupsPage />, { searchParams: "?group_search=admin" });
      expect(screen.getByPlaceholderText(SEARCH_PLACEHOLDER)).toHaveValue("admin");
      expect(screen.getByText("Admin Group")).toBeInTheDocument();
      expect(screen.queryByText("Read Only")).not.toBeInTheDocument();
    });

    it("writes the search to group_search, resets the page, and removes it when cleared", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      mockUseAccessGroups.mockReturnValue({ data: makeGroups(25), isLoading: false });
      renderWithProviders(<AccessGroupsPage />, { searchParams: "?page=2", onUrlUpdate });
      fireEvent.change(screen.getByPlaceholderText(SEARCH_PLACEHOLDER), { target: { value: "Group 0" } });
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("group_search")).toBe("Group 0"));
      expect(lastUrl(onUrlUpdate)?.searchParams.has("search")).toBe(false);
      expect(lastUrl(onUrlUpdate)?.searchParams.has("page")).toBe(false);
      await user.click(screen.getByRole("button", { name: "Clear search" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.has("group_search")).toBe(false));
      expect(screen.getByPlaceholderText(SEARCH_PLACEHOLDER)).toHaveValue("");
    });

    it("opens the page named in the URL and writes page changes back", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      mockUseAccessGroups.mockReturnValue({ data: makeGroups(25), isLoading: false });
      renderWithProviders(<AccessGroupsPage />, { searchParams: "?page=3", onUrlUpdate });
      expect(renderedGroupIds()).toEqual(["ag-21", "ag-22", "ag-23", "ag-24", "ag-25"]);
      await user.click(screen.getByTestId("pagination-prev"));
      expect(lastUrl(onUrlUpdate)?.searchParams.get("page")).toBe("2");
      expect(screen.getByText("ag-11")).toBeInTheDocument();
    });

    it("sorts by the column in the URL and writes header sort changes back", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      mockUseAccessGroups.mockReturnValue({ data: [mockAccessGroups[1], mockAccessGroups[0]], isLoading: false });
      renderWithProviders(<AccessGroupsPage />, { searchParams: "?sort_by=name&sort_order=asc", onUrlUpdate });
      expect(renderedGroupIds()).toEqual(["ag-1", "ag-2"]);
      await user.click(screen.getByTestId("sort-header-name"));
      expect(lastUrl(onUrlUpdate)?.searchParams.get("sort_by")).toBe("name");
      expect(lastUrl(onUrlUpdate)?.searchParams.has("sort_order")).toBe(false);
      expect(renderedGroupIds()).toEqual(["ag-2", "ag-1"]);
    });

    it("sorts oldest first when the URL asks for createdAt ascending", () => {
      mockUseAccessGroups.mockReturnValue({ data: [mockAccessGroups[0], mockAccessGroups[1]], isLoading: false });
      renderWithProviders(<AccessGroupsPage />, { searchParams: "?sort_by=createdAt&sort_order=asc" });
      expect(renderedGroupIds()).toEqual(["ag-2", "ag-1"]);
    });

    it("sorts by name in the default descending order when only sort_by is in the URL", () => {
      mockUseAccessGroups.mockReturnValue({ data: [mockAccessGroups[0], mockAccessGroups[1]], isLoading: false });
      renderWithProviders(<AccessGroupsPage />, { searchParams: "?sort_by=name" });
      expect(renderedGroupIds()).toEqual(["ag-2", "ag-1"]);
    });

    it("falls back to newest first when the URL names an unsortable column", () => {
      mockUseAccessGroups.mockReturnValue({ data: [mockAccessGroups[1], mockAccessGroups[0]], isLoading: false });
      renderWithProviders(<AccessGroupsPage />, { searchParams: "?sort_by=bogus" });
      expect(renderedGroupIds()).toEqual(["ag-1", "ag-2"]);
    });
  });
});
