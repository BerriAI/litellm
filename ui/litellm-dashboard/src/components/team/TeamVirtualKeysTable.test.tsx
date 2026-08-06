import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi, MockedFunction } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { TeamVirtualKeysTable } from "./TeamVirtualKeysTable";
import { KeysResponse, useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { KeyResponse } from "../key_team_helpers/key_list";
import { Organization } from "../networking";

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: vi.fn(),
}));

vi.mock("../key_team_helpers/fetch_available_models_team_key", () => ({
  getModelDisplayName: vi.fn((model: string) => model),
}));

vi.mock("../templates/key_info_view", () => ({
  default: vi.fn(({ onClose }: { onClose: () => void }) => (
    <div>
      <span>Key Info View</span>
      <button onClick={onClose}>Close</button>
    </div>
  )),
}));

// Resolve the debounced search synchronously so typed input lands in the useKeys query within the test tick.
vi.mock("@tanstack/react-pacer/debouncer", () => ({
  useDebouncedValue: (value: unknown) => [value, { cancel: vi.fn(), flush: vi.fn() }],
}));

const mockUseKeys = useKeys as MockedFunction<typeof useKeys>;

const createMockKey = (overrides: Partial<KeyResponse> = {}): KeyResponse =>
  ({
    token: "sk-test123",
    token_id: "key-1",
    key_alias: "alice_key_team1",
    key_name: "sk-...abc",
    user_id: "user-1",
    organization_id: null,
    user: { user_id: "user-1", user_email: "alice@example.com" },
    created_at: "2024-01-01T00:00:00Z",
    team_id: "team-1",
    spend: 0,
    max_budget: 100,
    models: ["gpt-4"],
    ...overrides,
  }) as KeyResponse;

const mockOrganization: Organization = {
  organization_id: "org-123",
  organization_alias: "Test Org",
  budget_id: "budget-1",
  metadata: {},
  models: [],
  spend: 0,
  model_spend: {},
  created_at: "",
  created_by: "",
  updated_at: "",
  updated_by: "",
  litellm_budget_table: {},
  teams: [],
  users: [],
  members: [],
};

describe("TeamVirtualKeysTable", () => {
  const defaultProps = {
    teamId: "team-1",
    teamAlias: "team1",
    organization: null as Organization | null,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseKeys.mockReturnValue({
      data: { keys: [], total_count: 0, current_page: 1, total_pages: 1 } as KeysResponse,
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as any);
  });

  it("should call useKeys with page, pageSize, and expand user for server-side pagination", async () => {
    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await waitFor(() => {
      expect(mockUseKeys).toHaveBeenCalledWith(
        1,
        50,
        expect.objectContaining({
          teamID: "team-1",
          expand: "user",
        }),
      );
    });
  });

  it("should enrich keys with organization_id when organization is provided", async () => {
    const keyWithoutOrg = createMockKey({ organization_id: null });
    mockUseKeys.mockReturnValue({
      data: {
        keys: [keyWithoutOrg],
        total_count: 1,
        current_page: 1,
        total_pages: 1,
      } as KeysResponse,
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as any);

    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} organization={mockOrganization} />);

    // Key with org_id should display in table - org-123 from organization
    await waitFor(() => {
      expect(screen.getByText("org-123")).toBeInTheDocument();
    });
  });

  it("should show table with Key ID column header", async () => {
    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Key ID")).toBeInTheDocument();
    });
  });

  it("should display keys in table when data is loaded", async () => {
    mockUseKeys.mockReturnValue({
      data: {
        keys: [
          createMockKey({ key_alias: "alice_key_team1" }),
          createMockKey({ token: "sk-2", token_id: "key-2", key_alias: "bob_key_team1" }),
        ],
        total_count: 2,
        current_page: 1,
        total_pages: 1,
      } as KeysResponse,
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as any);

    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("alice_key_team1")).toBeInTheDocument();
    });
    expect(screen.getByText("bob_key_team1")).toBeInTheDocument();
  });

  it("should show the current range from total_count when multiple pages exist", async () => {
    mockUseKeys.mockReturnValue({
      data: {
        keys: [createMockKey()],
        total_count: 100,
        current_page: 1,
        total_pages: 3,
      } as KeysResponse,
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as any);

    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-50 of 100");
    });
  });

  it("should fetch page 2 when Next is clicked", async () => {
    const user = userEvent.setup();
    mockUseKeys.mockImplementation(
      (page: number) =>
        ({
          data: {
            keys: page === 1 ? [createMockKey()] : [createMockKey({ token: "sk-page2", key_alias: "page2_key" })],
            total_count: 100,
            current_page: page,
            total_pages: 3,
          } as KeysResponse,
          isPending: false,
          isFetching: false,
          refetch: vi.fn(),
        }) as any,
    );

    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-50 of 100");
    });

    await user.click(screen.getByTestId("pagination-next"));

    await waitFor(() => {
      expect(mockUseKeys).toHaveBeenLastCalledWith(2, 50, expect.objectContaining({ teamID: "team-1" }));
    });
  });

  it("routes a sort-header click to useKeys as a server-side sort", async () => {
    const user = userEvent.setup();
    mockUseKeys.mockReturnValue({
      data: { keys: [createMockKey()], total_count: 1, current_page: 1, total_pages: 1 } as KeysResponse,
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useKeys>);

    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await waitFor(() => expect(screen.getByTestId("sort-header-created_at")).toBeInTheDocument());
    await user.click(screen.getByTestId("sort-header-created_at"));

    await waitFor(() =>
      expect(mockUseKeys).toHaveBeenLastCalledWith(
        1,
        50,
        expect.objectContaining({ sortBy: "created_at", sortOrder: "asc" }),
      ),
    );
  });

  it("resets to the first page when the sort changes", async () => {
    const user = userEvent.setup();
    mockUseKeys.mockImplementation(
      (page: number) =>
        ({
          data: {
            keys: [createMockKey({ token: `sk-p${page}`, key_alias: `page${page}_key` })],
            total_count: 100,
            current_page: page,
            total_pages: 2,
          },
          isPending: false,
          isFetching: false,
          refetch: vi.fn(),
        }) as unknown as ReturnType<typeof useKeys>,
    );

    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await user.click(await screen.findByTestId("pagination-next"));
    await waitFor(() => expect(mockUseKeys).toHaveBeenLastCalledWith(2, 50, expect.anything()));

    await user.click(screen.getByTestId("sort-header-created_at"));
    await waitFor(() => expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.anything()));
  });

  it("maps the User ID drawer filter to a server-side useKeys query and clears it", async () => {
    const user = userEvent.setup();
    mockUseKeys.mockReturnValue({
      data: { keys: [createMockKey()], total_count: 1, current_page: 1, total_pages: 1 },
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useKeys>);

    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await user.click(await screen.findByTestId("datatable-filters-trigger"));
    const drawerBody = await screen.findByTestId("filter-drawer-body");
    const userInput = drawerBody.querySelector("input") as HTMLElement;
    await user.type(userInput, "user-42");
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() =>
      expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ userID: "user-42" })),
    );

    await user.click(screen.getByTestId("datatable-clear-filters"));
    await waitFor(() =>
      expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ userID: undefined })),
    );
  });

  it("maps the search box to a server-side key-alias query", async () => {
    const user = userEvent.setup();
    mockUseKeys.mockReturnValue({
      data: { keys: [createMockKey()], total_count: 1, current_page: 1, total_pages: 1 },
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useKeys>);

    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await user.type(await screen.findByTestId("datatable-search"), "check-002");

    await waitFor(() =>
      expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ selectedKeyAlias: "check-002" })),
    );
  });

  it("should show Loading keys when isPending", async () => {
    mockUseKeys.mockReturnValue({
      data: undefined,
      isPending: true,
      isFetching: true,
      refetch: vi.fn(),
    } as any);

    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Loading keys...")).toBeInTheDocument();
    });
  });

  it("should show the empty state when keys array is empty", async () => {
    mockUseKeys.mockReturnValue({
      data: { keys: [], total_count: 0, current_page: 1, total_pages: 1 } as KeysResponse,
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as any);

    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("No rows match your search or filters.")).toBeInTheDocument();
    });
  });

  it("should open Key Info View when key is clicked", async () => {
    mockUseKeys.mockReturnValue({
      data: {
        keys: [createMockKey({ token: "sk-click-me", key_alias: "clickable_key" })],
        total_count: 1,
        current_page: 1,
        total_pages: 1,
      } as KeysResponse,
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as any);

    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("clickable_key")).toBeInTheDocument();
    });

    const keyButton = screen.getByRole("button", { name: /sk-click-me|clickable_key/ });
    await userEvent.click(keyButton);

    await waitFor(() => {
      expect(screen.getByText("Key Info View")).toBeInTheDocument();
    });
  });
});
