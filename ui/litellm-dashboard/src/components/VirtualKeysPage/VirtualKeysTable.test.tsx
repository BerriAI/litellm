import { screen, waitFor, within, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, it, expect, beforeEach, describe, MockedFunction } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { VirtualKeysTable } from "./VirtualKeysTable";
import { KeyResponse, Team } from "../key_team_helpers/key_list";
import { KeysResponse, useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import useTeams from "@/app/(dashboard)/hooks/useTeams";

// Resolve debounced values synchronously so an applied filter lands in the useKeys query within the test tick.
vi.mock("@tanstack/react-pacer/debouncer", async () => {
  const React = await vi.importActual<typeof import("react")>("react");
  return {
    useDebouncedValue: (value: unknown) => [value, { cancel: vi.fn(), flush: vi.fn() }],
    useDebouncedState: (initial: unknown) => {
      const [value, setValue] = React.useState(initial);
      return [value, setValue, { cancel: vi.fn(), flush: vi.fn() }];
    },
  };
});

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(() => ({
    accessToken: "test-token",
    userId: "test-user",
    userRole: "Admin",
    premiumUser: true,
    token: "test-token",
  })),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useAllTeams: vi.fn(() => ({
    data: [{ team_id: "team-1", team_alias: "Test Team" }],
    isLoading: false,
  })),
}));

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: vi.fn(),
  keyKeys: { lists: () => ["keys", "list"] },
}));

vi.mock("@/app/(dashboard)/hooks/useTeams", () => ({
  default: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: vi.fn().mockReturnValue({
    data: [
      {
        organization_id: "org-1",
        organization_alias: "Test Organization",
      },
    ],
  }),
}));

const mockKey: KeyResponse = {
  token: "sk-1234567890abcdef",
  token_id: "key-1",
  key_name: "test-key",
  key_alias: "Test Key Alias",
  spend: 5.5,
  max_budget: 100,
  expires: "2999-12-31T23:59:59Z",
  models: ["gpt-3.5-turbo", "gpt-4"],
  aliases: {},
  config: {},
  user_id: "user-1",
  team_id: "team-1",
  project_id: null,
  max_parallel_requests: 10,
  metadata: {},
  tpm_limit: 1000,
  rpm_limit: 100,
  duration: "30d",
  budget_duration: "1m",
  budget_reset_at: "2024-12-01T00:00:00Z",
  allowed_cache_controls: [],
  allowed_routes: [],
  permissions: {},
  model_spend: { "gpt-3.5-turbo": 2.5, "gpt-4": 3.0 },
  model_max_budget: { "gpt-3.5-turbo": 50, "gpt-4": 50 },
  soft_budget_cooldown: false,
  blocked: false,
  litellm_budget_table: {},
  organization_id: "org-1",
  created_at: "2024-11-01T10:00:00Z",
  created_by: "user-1",
  updated_at: "2024-11-15T10:00:00Z",
  last_active: "2024-11-20T14:30:00Z",
  team_spend: 5.5,
  team_alias: "Test Team",
  team_tpm_limit: 5000,
  team_rpm_limit: 500,
  team_max_budget: 500,
  team_models: ["gpt-3.5-turbo", "gpt-4"],
  team_blocked: false,
  soft_budget: 50,
  team_model_aliases: {},
  team_member_spend: 0,
  team_metadata: {},
  end_user_id: "end-user-1",
  end_user_tpm_limit: 100,
  end_user_rpm_limit: 10,
  end_user_max_budget: 10,
  last_refreshed_at: Date.now(),
  api_key: "sk-1234567890abcdef",
  user_role: "user",
  rpm_limit_per_model: {},
  tpm_limit_per_model: {},
  user_tpm_limit: 1000,
  user_rpm_limit: 100,
  user_email: "user@example.com",
  user: {
    user_email: "user@example.com",
    user_id: "user-1",
    user_alias: null,
  },
};

const mockTeam: Team = {
  team_id: "team-1",
  team_alias: "Test Team",
  models: ["gpt-3.5-turbo", "gpt-4"],
  max_budget: 500,
  budget_duration: "1m",
  tpm_limit: 5000,
  rpm_limit: 500,
  organization_id: "org-1",
  created_at: "2024-10-01T10:00:00Z",
  keys: [],
  members_with_roles: [],
  spend: 0,
};

const mockUseKeys = useKeys as MockedFunction<typeof useKeys>;
const mockUseTeams = useTeams as MockedFunction<typeof useTeams>;

const keysResult = (keys: KeyResponse[], data: Partial<KeysResponse> = {}, extra: Record<string, unknown> = {}) =>
  ({
    data: {
      keys,
      total_count: keys.length,
      current_page: 1,
      total_pages: 1,
      ...data,
    } as KeysResponse,
    isPending: false,
    isFetching: false,
    isError: false,
    refetch: vi.fn(),
    ...extra,
  }) as any;

const openFilters = () => fireEvent.click(screen.getByRole("button", { name: "Filters" }));

beforeEach(() => {
  vi.clearAllMocks();

  mockUseKeys.mockReturnValue(keysResult([mockKey]));

  mockUseTeams.mockReturnValue({
    teams: [mockTeam],
    setTeams: vi.fn(),
  });
});

it("should render VirtualKeysTable component", () => {
  renderWithProviders(<VirtualKeysTable />);
  expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
});

it("renders the page header with the create-key action slot", () => {
  renderWithProviders(<VirtualKeysTable headerActions={<button>Create New Key</button>} />);
  expect(screen.getByRole("heading", { name: "Virtual Keys" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Create New Key" })).toBeInTheDocument();
});

it("should display key information correctly", async () => {
  renderWithProviders(<VirtualKeysTable />);

  await waitFor(() => {
    expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
    expect(screen.getByText("Test Team")).toBeInTheDocument();
    expect(screen.getByText("$5.5000")).toBeInTheDocument();
    expect(screen.getByText("of $100")).toBeInTheDocument();
  });
});

it("should display user email correctly", async () => {
  renderWithProviders(<VirtualKeysTable />);

  await waitFor(() => {
    expect(screen.getByText("user@example.com")).toBeInTheDocument();
  });
});

it("shows the user alias over the email in the visible cell when both exist", async () => {
  mockUseKeys.mockReturnValue(
    keysResult([{ ...mockKey, user: { user_id: "user-1", user_email: "user@example.com", user_alias: "The User" } }]),
  );

  renderWithProviders(<VirtualKeysTable />);

  const row = (await screen.findByText("Test Key Alias")).closest("tr") as HTMLElement;
  expect(within(row).getByText("The User")).toBeInTheDocument();
  expect(within(row).queryByText("user@example.com")).not.toBeInTheDocument();
});

it("should show a loading state on the initial load and hide the data", () => {
  mockUseKeys.mockReturnValue(keysResult([], {}, { data: null, isPending: true, isFetching: true }));

  renderWithProviders(<VirtualKeysTable />);

  expect(screen.getByText("Loading keys...")).toBeInTheDocument();
  expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
  expect(screen.queryByText("Test Key Alias")).not.toBeInTheDocument();
});

it("should show 'No keys found' message when the key list is empty", () => {
  mockUseKeys.mockReturnValue(keysResult([]));

  renderWithProviders(<VirtualKeysTable />);

  expect(screen.getByText("No keys found")).toBeInTheDocument();
});

it("collapses models beyond the visible limit into a '+N more' badge", () => {
  mockUseKeys.mockReturnValue(
    keysResult([{ ...mockKey, models: ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "claude-3", "claude-3-5-sonnet"] }]),
  );

  renderWithProviders(<VirtualKeysTable />);

  expect(screen.getByText("+2 more")).toBeInTheDocument();
});

it("should render the redesigned table headers", () => {
  renderWithProviders(<VirtualKeysTable />);

  expect(screen.getByText("Key")).toBeInTheDocument();
  expect(screen.getByText("Team")).toBeInTheDocument();
  expect(screen.getByText("Models")).toBeInTheDocument();
  expect(screen.getByText("Spend / Budget")).toBeInTheDocument();
});

it("sorts by the backend key_alias field (not the column label) when the Key header is clicked", async () => {
  renderWithProviders(<VirtualKeysTable />);

  const keyHeader = screen.getByText("Key").closest("button") as HTMLElement;
  fireEvent.click(keyHeader);

  await waitFor(() => {
    expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ sortBy: "key_alias" }));
  });
});

it("should open KeyInfoView when clicking the key cell", async () => {
  renderWithProviders(<VirtualKeysTable />);

  await waitFor(() => {
    expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
  });

  expect(screen.getByTestId("pagination-range")).toBeInTheDocument();

  fireEvent.click(screen.getByText("Test Key Alias"));

  await waitFor(() => {
    expect(screen.getByText("Back to Keys")).toBeInTheDocument();
  });

  expect(screen.queryByTestId("pagination-range")).not.toBeInTheDocument();
});

it("should display 'Default Proxy Admin' for user_id when value is 'default_user_id'", async () => {
  mockUseKeys.mockReturnValue(
    keysResult([
      {
        ...mockKey,
        user_id: "default_user_id",
        user_email: "",
        user: { user_id: "default_user_id", user_email: "", user_alias: null },
      },
    ]),
  );

  renderWithProviders(<VirtualKeysTable />);

  await waitFor(() => {
    expect(screen.getByText("Default Proxy Admin")).toBeInTheDocument();
  });
});

it("should render table without crashing when models is null", async () => {
  mockUseKeys.mockReturnValue(keysResult([{ ...mockKey, models: null as unknown as string[] }]));

  renderWithProviders(<VirtualKeysTable />);

  await waitFor(() => {
    expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
    expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
  });
});

it("should display 'Unknown' for last_active when value is null", async () => {
  mockUseKeys.mockReturnValue(keysResult([{ ...mockKey, last_active: null }]));

  renderWithProviders(<VirtualKeysTable />);

  await waitFor(() => {
    expect(screen.getByText("Unknown")).toBeInTheDocument();
  });
});

describe("server-side filtering – the LIT-4080 regression guard", () => {
  it("threads an applied User ID filter into the useKeys query so any refetch keeps it", async () => {
    renderWithProviders(<VirtualKeysTable />);

    openFilters();

    const userIdInput = await screen.findByPlaceholderText(/Enter User ID/);
    fireEvent.change(userIdInput, { target: { value: "user-42" } });
    fireEvent.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => {
      expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ userID: "user-42" }));
    });
  });

  it("does not send filter params to useKeys when no filter is active", () => {
    renderWithProviders(<VirtualKeysTable />);

    const lastCall = mockUseKeys.mock.calls[mockUseKeys.mock.calls.length - 1];
    expect(lastCall[2] ?? {}).toMatchObject({ userID: undefined, teamID: undefined, keyHash: undefined });
  });

  it("drops the filter from the useKeys query when it is cleared", async () => {
    renderWithProviders(<VirtualKeysTable />);

    openFilters();
    const userIdInput = await screen.findByPlaceholderText(/Enter User ID/);
    fireEvent.change(userIdInput, { target: { value: "user-42" } });
    fireEvent.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => {
      expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ userID: "user-42" }));
    });

    fireEvent.click(screen.getByTestId("datatable-clear-filters"));

    await waitFor(() => {
      const lastCall = mockUseKeys.mock.calls[mockUseKeys.mock.calls.length - 1];
      expect((lastCall[2] ?? {}).userID).toBeUndefined();
    });
  });
});

describe("pagination display – total count comes from useKeys", () => {
  it("shows total_count and page count from the useKeys response", async () => {
    mockUseKeys.mockReturnValue(keysResult([mockKey], { total_count: 509, total_pages: 11 }));

    renderWithProviders(<VirtualKeysTable />);

    await waitFor(() => {
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-50 of 509");
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 11");
    });
  });

  it("reflects a narrowed total when a filtered fetch returns fewer results", async () => {
    mockUseKeys.mockReturnValue(keysResult([mockKey], { total_count: 1, total_pages: 1 }));

    renderWithProviders(<VirtualKeysTable />);

    await waitFor(() => {
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-1 of 1");
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 1");
    });
  });
});

describe("refresh button", () => {
  it("renders an enabled refresh control in the normal state", () => {
    renderWithProviders(<VirtualKeysTable />);

    const refresh = screen.getByTestId("datatable-refresh");
    expect(refresh).toBeInTheDocument();
    expect(refresh).not.toBeDisabled();
  });

  it("disables the refresh control while a fetch is in flight but keeps data visible", () => {
    mockUseKeys.mockReturnValue(keysResult([mockKey], {}, { isFetching: true }));

    renderWithProviders(<VirtualKeysTable />);

    expect(screen.getByTestId("datatable-refresh")).toBeDisabled();
    expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
  });

  it("calls refetch when the refresh control is clicked", () => {
    const mockRefetch = vi.fn();
    mockUseKeys.mockReturnValue(keysResult([mockKey], {}, { refetch: mockRefetch }));

    renderWithProviders(<VirtualKeysTable />);

    fireEvent.click(screen.getByTestId("datatable-refresh"));

    expect(mockRefetch).toHaveBeenCalledTimes(1);
  });
});

describe("Status column reflects blocked / expiry / scim metadata", () => {
  it("renders Active for a non-blocked, unexpired key", async () => {
    mockUseKeys.mockReturnValue(keysResult([{ ...mockKey, blocked: false, metadata: {} }]));

    renderWithProviders(<VirtualKeysTable />);

    await waitFor(() => {
      expect(screen.getByTestId(`key-status-${mockKey.token_id}`)).toHaveTextContent("Active");
    });
  });

  it("renders Expired when the expiry date has passed", async () => {
    mockUseKeys.mockReturnValue(
      keysResult([{ ...mockKey, blocked: false, metadata: {}, expires: "2020-01-01T00:00:00Z" }]),
    );

    renderWithProviders(<VirtualKeysTable />);

    await waitFor(() => {
      expect(screen.getByTestId(`key-status-${mockKey.token_id}`)).toHaveTextContent("Expired");
    });
  });

  it("renders Blocked when key.blocked is true", async () => {
    mockUseKeys.mockReturnValue(keysResult([{ ...mockKey, blocked: true, metadata: {} }]));

    renderWithProviders(<VirtualKeysTable />);

    await waitFor(() => {
      expect(screen.getByTestId(`key-status-${mockKey.token_id}`)).toHaveTextContent("Blocked");
    });
    expect(screen.queryByText(/Blocked by SCIM/i)).not.toBeInTheDocument();
  });

  it("marks a SCIM-blocked key with the SCIM tooltip reason", async () => {
    mockUseKeys.mockReturnValue(keysResult([{ ...mockKey, blocked: true, metadata: { scim_blocked: true } }]));

    renderWithProviders(<VirtualKeysTable />);

    const tag = await screen.findByTestId(`key-status-${mockKey.token_id}`);
    expect(tag).toHaveTextContent("Blocked");

    const user = userEvent.setup();
    await user.hover(tag);
    await waitFor(() => {
      expect(screen.getByText(/Blocked by SCIM/i)).toBeInTheDocument();
    });
  });
});
