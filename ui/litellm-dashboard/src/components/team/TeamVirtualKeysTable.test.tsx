import { QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactElement, ReactNode } from "react";
import { beforeEach, describe, expect, it, vi, Mock, MockedFunction } from "vitest";
import {
  fireEvent,
  render,
  renderWithProviders,
  screen,
  testQueryClient,
  waitFor,
  within,
} from "../../../tests/test-utils";
import { TeamVirtualKeysTable } from "./TeamVirtualKeysTable";
import { useKeyInfo } from "@/app/(dashboard)/hooks/keys/useKeyInfo";
import { KeysResponse, useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { KeyResponse } from "../key_team_helpers/key_list";
import { Organization } from "../networking";

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/keys/useKeyInfo", () => ({
  useKeyInfo: vi.fn(),
}));

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

vi.mock("../key_team_helpers/fetch_available_models_team_key", () => ({
  getModelDisplayName: vi.fn((model: string) => model),
}));

vi.mock("../templates/key_info_view", () => ({
  default: vi.fn(
    ({
      keyId,
      keyData,
      onClose,
      onKeyDataUpdate,
    }: {
      keyId: string;
      keyData?: { key_alias?: string | null; max_budget?: number | null };
      onClose: () => void;
      onKeyDataUpdate?: (data: { token?: string; spend?: number; max_budget?: number }) => void;
    }) => (
      <div>
        <span>Key Info View</span>
        <span data-testid="key-info-id">{keyId}</span>
        <span data-testid="key-info-alias">{keyData?.key_alias ?? "no key data"}</span>
        <span data-testid="key-info-budget">{keyData?.max_budget ?? "no budget"}</span>
        <button onClick={onClose}>Close</button>
        <button onClick={() => onKeyDataUpdate?.({ token: "sk-rotated", max_budget: 5 })}>Rotate</button>
        <button onClick={() => onKeyDataUpdate?.({ spend: 0 })}>Reset spend</button>
      </div>
    ),
  ),
}));

// Resolve the debounced search synchronously so typed input lands in the useKeys query within the test tick.
vi.mock("@tanstack/react-pacer/debouncer", () => ({
  useDebouncedValue: (value: unknown) => [value, { cancel: vi.fn(), flush: vi.fn() }],
}));

const mockUseKeys = useKeys as MockedFunction<typeof useKeys>;
const mockUseKeyInfo = useKeyInfo as MockedFunction<typeof useKeyInfo>;

const keyInfoResult = (data: KeyResponse | undefined, isError = false) =>
  ({ data, isError }) as unknown as ReturnType<typeof useKeyInfo>;

const keysResult = (keys: KeyResponse[], totalCount = keys.length, refetch = vi.fn()) =>
  ({
    data: { keys, total_count: totalCount, current_page: 1, total_pages: 1 },
    isPending: false,
    isFetching: false,
    isError: false,
    refetch,
  }) as unknown as ReturnType<typeof useKeys>;

const renderKeepingMountUpdates = (ui: ReactElement, searchParams: string, onUrlUpdate: OnUrlUpdateFunction) =>
  render(ui, {
    wrapper: ({ children }: { children: ReactNode }) => (
      <NuqsTestingAdapter
        searchParams={searchParams}
        onUrlUpdate={onUrlUpdate}
        hasMemory
        resetUrlUpdateQueueOnMount={false}
      >
        <QueryClientProvider client={testQueryClient}>{children}</QueryClientProvider>
      </NuqsTestingAdapter>
    ),
  });

const lastUrlUpdate = (onUrlUpdate: Mock<OnUrlUpdateFunction>) => {
  const event = onUrlUpdate.mock.calls.at(-1)?.[0];
  if (!event) throw new Error("no URL update was emitted");
  return event;
};

const KEY_HASH = "88a145505dd6e87e2ea166fcef1e4b53948dbdb32af6431dfd05ec06b571ee52";

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
    mockUseKeyInfo.mockReturnValue(keyInfoResult(undefined));
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

    expect(await screen.findByTestId("sort-header-created_at")).toBeInTheDocument();
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
    const userInput = within(drawerBody).getByPlaceholderText("Filter by user ID…");
    fireEvent.change(userInput, { target: { value: "user-42" } });
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() =>
      expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ userID: "user-42" })),
    );

    await user.click(screen.getByTestId("datatable-clear-filters"));
    await waitFor(() =>
      expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ userID: undefined })),
    );
  });

  it("maps the Key ID drawer filter to a server-side useKeys query and clears it", async () => {
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
    fireEvent.change(within(drawerBody).getByPlaceholderText("Enter Key ID…"), { target: { value: KEY_HASH } });
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() =>
      expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ keyHash: KEY_HASH })),
    );
    expect(screen.getByTestId("filter-chip-key_hash")).toHaveTextContent("Key ID");

    await user.click(screen.getByTestId("datatable-clear-filters"));
    await waitFor(() =>
      expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ keyHash: undefined })),
    );
  });

  it("maps the search box to the combined alias-or-ID search rather than the key-alias filter", async () => {
    mockUseKeys.mockReturnValue({
      data: { keys: [createMockKey()], total_count: 1, current_page: 1, total_pages: 1 },
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useKeys>);

    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    const searchBox = await screen.findByTestId("datatable-search");
    expect(searchBox).toHaveAttribute("placeholder", "Search by key alias or ID…");
    fireEvent.change(searchBox, { target: { value: KEY_HASH } });

    await waitFor(() =>
      expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ search: KEY_HASH })),
    );
    const lastOptions = mockUseKeys.mock.calls.at(-1)?.[2];
    expect(lastOptions?.selectedKeyAlias).toBeUndefined();
    expect(lastOptions?.keyHash).toBeUndefined();
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

  describe("URL state", () => {
    it("reads the keys_ prefixed table state from the URL and ignores unprefixed params", async () => {
      mockUseKeys.mockReturnValue(keysResult([createMockKey()], 100));

      renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />, {
        searchParams: {
          keys_search: "prod",
          keys_page: "3",
          keys_page_size: "25",
          keys_sort_by: "spend",
          keys_sort_order: "asc",
          keys_filter_user: "user-9",
          keys_filter_key_id: KEY_HASH,
          page: "4",
          search: "list-search",
          sort_by: "key_alias",
          filter_user_id: "user-ignored",
        },
      });

      const expectedQuery = { search: "prod", sortBy: "spend", sortOrder: "asc", userID: "user-9", keyHash: KEY_HASH };
      await waitFor(() => expect(mockUseKeys).toHaveBeenLastCalledWith(3, 25, expect.objectContaining(expectedQuery)));
      expect(screen.getByTestId("datatable-search")).toHaveValue("prod");
    });

    it("falls back to the created_at sort for a keys_sort_by the backend cannot sort on", async () => {
      mockUseKeys.mockReturnValue(keysResult([createMockKey()]));

      renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />, {
        searchParams: { keys_sort_by: "user_email", keys_sort_order: "asc" },
      });

      await waitFor(() =>
        expect(mockUseKeys).toHaveBeenLastCalledWith(
          1,
          50,
          expect.objectContaining({ sortBy: "created_at", sortOrder: "asc" }),
        ),
      );
    });

    it.each(["token", "key_alias", "created_at", "updated_at", "spend", "max_budget"])(
      "passes the sortable column %s from keys_sort_by to the keys query",
      async (sortBy) => {
        mockUseKeys.mockReturnValue(keysResult([createMockKey()]));

        renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />, {
          searchParams: { keys_sort_by: sortBy, keys_sort_order: "asc" },
        });

        await waitFor(() =>
          expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ sortBy, sortOrder: "asc" })),
        );
      },
    );

    it("keeps the linked keys_page while the keys request is failing", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      mockUseKeys.mockReturnValue({
        data: undefined,
        isPending: false,
        isFetching: false,
        isError: true,
        refetch: vi.fn(),
      } as unknown as ReturnType<typeof useKeys>);

      renderKeepingMountUpdates(<TeamVirtualKeysTable {...defaultProps} />, "?keys_page=3", onUrlUpdate);

      await new Promise((resolve) => setTimeout(resolve, 100));
      expect(mockUseKeys).toHaveBeenLastCalledWith(3, 50, expect.anything());
      expect(onUrlUpdate).not.toHaveBeenCalled();
    });

    it("writes the search box to keys_search and leaves the list search alone", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      mockUseKeys.mockReturnValue(keysResult([createMockKey()]));

      renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />, {
        searchParams: { team_search: "outer" },
        onUrlUpdate,
      });

      fireEvent.change(await screen.findByTestId("datatable-search"), { target: { value: "alice" } });

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.get("keys_search")).toBe("alice"));
      const params = lastUrlUpdate(onUrlUpdate).searchParams;
      expect(params.get("team_search")).toBe("outer");
      expect(params.has("search")).toBe(false);
    });

    it("writes paging and sorting to keys_page and keys_sort_by", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      mockUseKeys.mockReturnValue(keysResult([createMockKey()], 100));

      renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />, { onUrlUpdate });

      await user.click(await screen.findByTestId("pagination-next"));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.get("keys_page")).toBe("2"));
      expect(lastUrlUpdate(onUrlUpdate).searchParams.has("page")).toBe(false);

      await user.click(screen.getByTestId("sort-header-key_alias"));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.get("keys_sort_by")).toBe("key_alias"));
      await waitFor(() =>
        expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ sortBy: "key_alias" })),
      );
      const params = lastUrlUpdate(onUrlUpdate).searchParams;
      expect(params.has("keys_page")).toBe(false);
      expect(params.has("sort_by")).toBe(false);
    });

    it("writes applied drawer filters to the renamed keys_filter params", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      mockUseKeys.mockReturnValue(keysResult([createMockKey()]));

      renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />, { onUrlUpdate });

      await user.click(await screen.findByTestId("datatable-filters-trigger"));
      const drawerBody = await screen.findByTestId("filter-drawer-body");
      fireEvent.change(within(drawerBody).getByPlaceholderText("Filter by user ID…"), {
        target: { value: "user-42" },
      });
      fireEvent.change(within(drawerBody).getByPlaceholderText("Enter Key ID…"), { target: { value: KEY_HASH } });
      await user.click(screen.getByTestId("filter-drawer-apply"));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.get("keys_filter_user")).toBe("user-42"));
      const params = lastUrlUpdate(onUrlUpdate).searchParams;
      expect(params.get("keys_filter_key_id")).toBe(KEY_HASH);
      expect(params.has("filter_user")).toBe(false);
    });

    it("pushes the clicked key into the key param", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      mockUseKeys.mockReturnValue(keysResult([createMockKey({ token: "sk-click-me", key_alias: "clickable_key" })]));

      renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />, { onUrlUpdate });

      await user.click(await screen.findByRole("button", { name: /sk-click-me/ }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.get("key")).toBe("sk-click-me"));
      expect(lastUrlUpdate(onUrlUpdate).options.history).toBe("push");
      expect(screen.getByTestId("key-info-id")).toHaveTextContent("sk-click-me");
    });

    it("opens a key from the URL using the row already in the list", async () => {
      mockUseKeys.mockReturnValue(keysResult([createMockKey({ token: "sk-listed", key_alias: "listed_key" })]));

      renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />, { searchParams: { key: "sk-listed" } });

      expect(await screen.findByTestId("key-info-alias")).toHaveTextContent("listed_key");
      expect(screen.getByTestId("key-info-id")).toHaveTextContent("sk-listed");
      expect(mockUseKeyInfo).toHaveBeenLastCalledWith("sk-listed", { enabled: false });
    });

    it("fetches a key from the URL that is not on the current page", async () => {
      mockUseKeys.mockReturnValue(keysResult([createMockKey({ token: "sk-listed" })]));
      mockUseKeyInfo.mockReturnValue(keyInfoResult(createMockKey({ token: "sk-elsewhere", key_alias: "far_key" })));

      renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />, { searchParams: { key: "sk-elsewhere" } });

      expect(await screen.findByTestId("key-info-alias")).toHaveTextContent("far_key");
      expect(mockUseKeyInfo).toHaveBeenLastCalledWith("sk-elsewhere", { enabled: true });
    });

    it("shows a loading state while a deep-linked key is fetched and the detail view once the fetch fails", async () => {
      mockUseKeys.mockReturnValue(keysResult([]));

      const { rerender } = renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />, {
        searchParams: { key: "sk-missing" },
      });

      expect(await screen.findByText("Loading key...")).toBeInTheDocument();
      expect(screen.queryByText("Key Info View")).not.toBeInTheDocument();
      expect(screen.queryByTestId("datatable-search")).not.toBeInTheDocument();

      mockUseKeyInfo.mockReturnValue(keyInfoResult(undefined, true));
      rerender(<TeamVirtualKeysTable {...defaultProps} />);

      expect(await screen.findByTestId("key-info-id")).toHaveTextContent("sk-missing");
      expect(screen.getByTestId("key-info-alias")).toHaveTextContent("no key data");
    });

    it("removes the key param and returns to the table on close", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      mockUseKeys.mockReturnValue(keysResult([createMockKey({ token: "sk-listed", key_alias: "listed_key" })]));

      renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />, {
        searchParams: { key: "sk-listed", team: "team-1" },
        onUrlUpdate,
      });

      await user.click(await screen.findByRole("button", { name: "Close" }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.has("key")).toBe(false));
      const { searchParams, options } = lastUrlUpdate(onUrlUpdate);
      expect(searchParams.get("team")).toBe("team-1");
      expect(searchParams.get("team_tab")).toBe("virtual-keys");
      expect(options.history).toBe("push");
      expect(await screen.findByText("listed_key")).toBeInTheDocument();
    });

    it("replaces the key param with the rotated token and keeps the detail open while the list refetches", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      const refetch = vi.fn();
      mockUseKeys.mockReturnValue(
        keysResult([createMockKey({ token: "sk-listed", key_alias: "listed_key", max_budget: 100 })], 1, refetch),
      );

      renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />, {
        searchParams: { key: "sk-listed" },
        onUrlUpdate,
      });

      await user.click(await screen.findByRole("button", { name: "Rotate" }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.get("key")).toBe("sk-rotated"));
      expect(lastUrlUpdate(onUrlUpdate).options.history).toBe("replace");
      expect(refetch).toHaveBeenCalled();
      await waitFor(() => expect(screen.getByTestId("key-info-id")).toHaveTextContent("sk-rotated"));
      expect(screen.queryByText("Loading key...")).not.toBeInTheDocument();
      expect(screen.getByTestId("key-info-alias")).toHaveTextContent("listed_key");
      expect(screen.getByTestId("key-info-budget")).toHaveTextContent("5");
    });

    it("leaves the key param alone for updates that do not rotate the key", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      const refetch = vi.fn();
      mockUseKeys.mockReturnValue(keysResult([createMockKey({ token: "sk-listed" })], 1, refetch));

      renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />, {
        searchParams: { key: "sk-listed" },
        onUrlUpdate,
      });

      await user.click(await screen.findByRole("button", { name: "Reset spend" }));

      await new Promise((resolve) => setTimeout(resolve, 100));
      expect(onUrlUpdate).not.toHaveBeenCalled();
      expect(refetch).not.toHaveBeenCalled();
      expect(screen.getByTestId("key-info-id")).toHaveTextContent("sk-listed");
    });
  });

  describe("entity links out of the key rows", () => {
    const renderRow = async (key: KeyResponse, organization: Organization | null = null) => {
      mockUseKeys.mockReturnValue({
        data: { keys: [key], total_count: 1, current_page: 1, total_pages: 1 } as KeysResponse,
        isPending: false,
        isFetching: false,
        refetch: vi.fn(),
      } as any);
      renderWithProviders(<TeamVirtualKeysTable {...defaultProps} organization={organization} />);
      await screen.findByText(key.key_alias as string);
      return screen.getByRole("row", { name: new RegExp(key.key_alias as string) });
    };

    it("points the Organization ID cell at the org's detail page", async () => {
      const row = await renderRow(createMockKey({ organization_id: null }), mockOrganization);
      expect(within(row).getByRole("link", { name: "org-123" })).toHaveAttribute(
        "href",
        "/ui/organizations?org=org-123",
      );
    });

    it("points the User Email and User ID cells at the owning user's detail page", async () => {
      const row = await renderRow(
        createMockKey({
          user_id: "user-1",
          user: { user_id: "user-1", user_email: "alice@example.com", user_alias: null },
        }),
      );
      expect(within(row).getByRole("link", { name: "alice@example.com" })).toHaveAttribute(
        "href",
        "/ui/users?user=user-1",
      );
      expect(within(row).getByRole("link", { name: "user-1" })).toHaveAttribute("href", "/ui/users?user=user-1");
    });

    it("points the Created By cell at the creator's detail page", async () => {
      const row = await renderRow(
        createMockKey({
          created_by: "creator-1",
          created_by_user: { user_id: "creator-1", user_email: "creator@example.com", user_alias: "The Creator" },
        }),
      );
      expect(within(row).getByRole("link", { name: "The Creator" })).toHaveAttribute(
        "href",
        "/ui/users?user=creator-1",
      );
    });

    it("leaves the default_user_id placeholder unlinked in the User ID and Created By cells", async () => {
      const placeholder = { user_id: "default_user_id", user_email: "admin@example.com", user_alias: "Proxy Admin" };
      const ownedAndCreatedByPlaceholder = {
        user_id: placeholder.user_id,
        user: placeholder,
        created_by: placeholder.user_id,
        created_by_user: placeholder,
      };
      const row = await renderRow(createMockKey(ownedAndCreatedByPlaceholder));
      expect(within(row).getByText("Default Proxy Admin")).toBeInTheDocument();
      expect(within(row).getByText("Proxy Admin")).toBeInTheDocument();
      expect(within(row).queryByRole("link", { name: "Proxy Admin" })).not.toBeInTheDocument();
      expect(within(row).queryByRole("link", { name: placeholder.user_email })).not.toBeInTheDocument();
      expect(within(row).queryByRole("link", { name: "Default Proxy Admin" })).not.toBeInTheDocument();
    });
  });
});
