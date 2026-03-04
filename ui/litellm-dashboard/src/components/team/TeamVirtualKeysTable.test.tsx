import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi, MockedFunction } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { TeamVirtualKeysTable } from "./TeamVirtualKeysTable";
import { KeysResponse, useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { fetchTeamFilterOptions } from "../key_team_helpers/filter_helpers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { KeyResponse } from "../key_team_helpers/key_list";
import { Organization } from "../networking";

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(),
}));

vi.mock("../key_team_helpers/filter_helpers", () => ({
  fetchTeamFilterOptions: vi.fn().mockResolvedValue({
    keyAliases: [],
    organizationIds: [],
    userIds: [],
  }),
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

const mockUseKeys = useKeys as MockedFunction<typeof useKeys>;
const mockUseAuthorized = useAuthorized as MockedFunction<typeof useAuthorized>;

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
  } as KeyResponse);

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
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token" } as any);
    mockUseKeys.mockReturnValue({
      data: { keys: [], total_count: 0, current_page: 1, total_pages: 1 } as KeysResponse,
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as any);
  });

  it("should render successfully", async () => {
    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("0 Members")).toBeInTheDocument();
    });
  });

  it("should display X Members instead of Showing X of Y results", async () => {
    mockUseKeys.mockReturnValue({
      data: {
        keys: [createMockKey(), createMockKey({ token: "sk-2", token_id: "key-2" })],
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
      expect(screen.getByText("2 Members")).toBeInTheDocument();
    });
    expect(screen.queryByText(/Showing.*results/)).not.toBeInTheDocument();
  });

  it("should display 1 Member when singular", async () => {
    mockUseKeys.mockReturnValue({
      data: {
        keys: [createMockKey()],
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
      expect(screen.getByText("1 Member")).toBeInTheDocument();
    });
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
        })
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

    renderWithProviders(
      <TeamVirtualKeysTable {...defaultProps} organization={mockOrganization} />
    );

    await waitFor(() => {
      expect(screen.getByText("1 Member")).toBeInTheDocument();
    });
    // Key with org_id should display in table - org-123 from organization
    await waitFor(() => {
      expect(screen.getByText("org-123")).toBeInTheDocument();
    });
  });

  it("should show table with Key ID column header", async () => {
    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("0 Members")).toBeInTheDocument();
    });
    expect(screen.getByText("Key ID")).toBeInTheDocument();
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
      expect(screen.getByText("2 Members")).toBeInTheDocument();
    });
    expect(screen.getByText("alice_key_team1")).toBeInTheDocument();
    expect(screen.getByText("bob_key_team1")).toBeInTheDocument();
  });

  it("should show Page X of Y when multiple pages exist", async () => {
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
      expect(screen.getByText("Page 1 of 3")).toBeInTheDocument();
    });
    expect(screen.getByText("100 Members")).toBeInTheDocument();
  });

  it("should fetch page 2 when Next is clicked", async () => {
    const user = userEvent.setup();
    mockUseKeys.mockImplementation((page: number) => ({
      data: {
        keys: page === 1 ? [createMockKey()] : [createMockKey({ token: "sk-page2", key_alias: "page2_key" })],
        total_count: 100,
        current_page: page,
        total_pages: 3,
      } as KeysResponse,
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as any));

    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Page 1 of 3")).toBeInTheDocument();
    });

    const nextButton = screen.getByRole("button", { name: "Next" });
    await user.click(nextButton);

    await waitFor(() => {
      expect(mockUseKeys).toHaveBeenLastCalledWith(
        2,
        50,
        expect.objectContaining({ teamID: "team-1" })
      );
    });
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

  it("should show No keys found when keys array is empty", async () => {
    mockUseKeys.mockReturnValue({
      data: { keys: [], total_count: 0, current_page: 1, total_pages: 1 } as KeysResponse,
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as any);

    renderWithProviders(<TeamVirtualKeysTable {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("0 Members")).toBeInTheDocument();
    });
    expect(screen.getByText("No keys found")).toBeInTheDocument();
  });

  it("should fetch team-scoped filter options for Key Alias, Organization ID, and User ID", async () => {
    const mockFetchTeamFilterOptions = vi.mocked(fetchTeamFilterOptions);
    mockFetchTeamFilterOptions.mockResolvedValue({
      keyAliases: ["alice_key_team1", "charlie_key_team1"],
      organizationIds: ["org-123"],
      userIds: [
        { id: "user-1", email: "alice@example.com" },
        { id: "user-2", email: "charlie@example.com" },
      ],
    });

    // Use unique teamId to avoid cache hit from previous tests (refetchOnMount: false)
    renderWithProviders(
      <TeamVirtualKeysTable {...defaultProps} teamId="team-filter-options-test" />
    );

    await waitFor(() => {
      expect(mockFetchTeamFilterOptions).toHaveBeenCalledWith(
        "test-token",
        "team-filter-options-test"
      );
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
