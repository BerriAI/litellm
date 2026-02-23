import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi, MockedFunction } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { TeamVirtualKeysTable } from "./TeamVirtualKeysTable";
import { KeysResponse, useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
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
  fetchAllKeyAliases: vi.fn().mockResolvedValue([]),
  fetchAllOrganizations: vi.fn().mockResolvedValue([]),
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

  it("should call useKeys with expand user to fetch user email", async () => {
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
});
