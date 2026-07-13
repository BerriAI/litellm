import { act, screen, waitFor, within, fireEvent } from "@testing-library/react";
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

const mockSetVirtualKeyId = vi.fn();
vi.mock("./useVirtualKeySearchParam", () => ({
  useVirtualKeySearchParam: vi.fn(() => ({
    virtualKeyId: null,
    setVirtualKeyId: mockSetVirtualKeyId,
  })),
  VIRTUAL_KEY_PARAM: "virtual_key",
}));

import { useVirtualKeySearchParam } from "./useVirtualKeySearchParam";

const mockKey: KeyResponse = {
  token: "sk-1234567890abcdef",
  token_id: "key-1",
  key_name: "test-key",
  key_alias: "Test Key Alias",
  spend: 5.5,
  max_budget: 100,
  expires: "2024-12-31T23:59:59Z",
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
const mockUseVirtualKeySearchParam = useVirtualKeySearchParam as MockedFunction<typeof useVirtualKeySearchParam>;

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

beforeEach(() => {
  vi.clearAllMocks();

  mockUseKeys.mockReturnValue(keysResult([mockKey]));

  mockUseTeams.mockReturnValue({
    teams: [mockTeam],
    setTeams: vi.fn(),
  });

  mockUseVirtualKeySearchParam.mockReturnValue({
    virtualKeyId: null,
    setVirtualKeyId: mockSetVirtualKeyId,
  });
});

it("should render VirtualKeysTable component", () => {
  renderWithProviders(<VirtualKeysTable />);
  expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
});

it("should display key information correctly", async () => {
  renderWithProviders(<VirtualKeysTable />);

  await waitFor(() => {
    expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
    expect(screen.getByText("Test Team")).toBeInTheDocument();
    expect(screen.getByText("5.5000")).toBeInTheDocument();
  });
});

it("should display user email correctly", async () => {
  renderWithProviders(<VirtualKeysTable />);

  await waitFor(() => {
    expect(screen.getByText("user@example.com")).toBeInTheDocument();
  });
});

it("should show loading message only on initial load (isPending)", () => {
  mockUseKeys.mockReturnValue(keysResult([], {}, { data: null, isPending: true, isFetching: true }));

  renderWithProviders(<VirtualKeysTable />);

  expect(screen.getByText("🚅 Loading keys...")).toBeInTheDocument();
  expect(screen.queryByText("Test Key Alias")).not.toBeInTheDocument();
  expect(screen.queryByText("Test Team")).not.toBeInTheDocument();
});

it("should show 'No keys found' message when the key list is empty", () => {
  mockUseKeys.mockReturnValue(keysResult([]));

  renderWithProviders(<VirtualKeysTable />);

  expect(screen.getByText("No keys found")).toBeInTheDocument();
});

it("should handle models with more than 3 entries to trigger expansion UI", () => {
  mockUseKeys.mockReturnValue(
    keysResult([{ ...mockKey, models: ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "claude-3", "claude-3-5-sonnet"] }]),
  );

  renderWithProviders(<VirtualKeysTable />);

  expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
});

it("should render table headers correctly", () => {
  renderWithProviders(<VirtualKeysTable />);

  expect(screen.getByText("Key ID")).toBeInTheDocument();
  expect(screen.getByText("Key Alias")).toBeInTheDocument();
  expect(screen.getByText("Team")).toBeInTheDocument();
  expect(screen.getByText("Models")).toBeInTheDocument();
  expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
});

it("should handle column resizing hover events", () => {
  renderWithProviders(<VirtualKeysTable />);

  const headerCell = document.querySelector("[data-header-id]") as HTMLElement;
  expect(headerCell).toBeInTheDocument();

  const resizer = headerCell?.querySelector(".resizer") as HTMLElement;
  expect(resizer).toBeInTheDocument();
  expect(resizer.style.opacity).toBe("0");

  fireEvent.mouseEnter(headerCell);
  expect(resizer.style.opacity).toBe("0.5");

  fireEvent.mouseLeave(headerCell);
  expect(resizer.style.opacity).toBe("0");
});

it("should open KeyInfoView when clicking on a key ID button", async () => {
  renderWithProviders(<VirtualKeysTable />);

  await waitFor(() => {
    expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
  });

  expect(screen.getByText(/Showing.*results/)).toBeInTheDocument();

  const keyIdButton = screen.getByText("sk-1234567890abcdef");
  fireEvent.click(keyIdButton);

  await waitFor(() => {
    expect(screen.getByText("Back to Keys")).toBeInTheDocument();
    expect(screen.getByText("Created At")).toBeInTheDocument();
  });

  expect(screen.queryByText(/Showing.*results/)).not.toBeInTheDocument();
  expect(mockSetVirtualKeyId).toHaveBeenCalledWith("sk-1234567890abcdef");
});

it("should open KeyInfoView from ?virtual_key= without rewriting the URL", async () => {
  mockUseVirtualKeySearchParam.mockReturnValue({
    virtualKeyId: "sk-1234567890abcdef",
    setVirtualKeyId: mockSetVirtualKeyId,
  });

  renderWithProviders(<VirtualKeysTable />);

  await waitFor(() => {
    expect(screen.getByText("Back to Keys")).toBeInTheDocument();
  });
  expect(mockSetVirtualKeyId).not.toHaveBeenCalled();
});

it("should clear ?virtual_key= when closing key detail", async () => {
  renderWithProviders(<VirtualKeysTable />);

  await waitFor(() => {
    expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
  });

  fireEvent.click(screen.getByText("sk-1234567890abcdef"));

  await waitFor(() => {
    expect(screen.getByText("Back to Keys")).toBeInTheDocument();
  });

  fireEvent.click(screen.getByText("Back to Keys"));

  await waitFor(() => {
    expect(screen.getByText(/Showing.*results/)).toBeInTheDocument();
  });
  expect(mockSetVirtualKeyId).toHaveBeenCalledWith(null);
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

it("should display created_by_user email in 'Created By' column when available", async () => {
  mockUseKeys.mockReturnValue(
    keysResult([
      {
        ...mockKey,
        created_by: "some-uuid-1234",
        created_by_user: { user_id: "some-uuid-1234", user_email: "creator@example.com", user_alias: null },
      },
    ]),
  );

  renderWithProviders(<VirtualKeysTable />);

  await waitFor(() => {
    expect(screen.getByText("creator@example.com")).toBeInTheDocument();
  });
});

it("should display created_by_user alias over email when both are available", async () => {
  mockUseKeys.mockReturnValue(
    keysResult([
      {
        ...mockKey,
        created_by: "some-uuid-1234",
        created_by_user: { user_id: "some-uuid-1234", user_email: "creator@example.com", user_alias: "The Creator" },
      },
    ]),
  );

  renderWithProviders(<VirtualKeysTable />);

  // Scope to the key's row so we assert the visible cell value: the hover popover that
  // also holds the email is portaled out of the row, not the displayed "Created By" text.
  const row = (await screen.findByText("Test Key Alias")).closest("tr") as HTMLElement;
  expect(within(row).getByText("The Creator")).toBeInTheDocument();
  expect(within(row).queryByText("creator@example.com")).not.toBeInTheDocument();
});

it("should render table without crashing when models is null", async () => {
  mockUseKeys.mockReturnValue(keysResult([{ ...mockKey, models: null as unknown as string[] }]));

  renderWithProviders(<VirtualKeysTable />);

  await waitFor(() => {
    expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
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
  it("threads an active User ID filter into the useKeys query so any refetch keeps it", async () => {
    renderWithProviders(<VirtualKeysTable />);

    fireEvent.click(screen.getByRole("button", { name: "Filters" }));

    const userIdInput = await screen.findByPlaceholderText("Enter User ID...");
    fireEvent.change(userIdInput, { target: { value: "user-42" } });

    await waitFor(() => {
      expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ userID: "user-42" }));
    });
  });

  it("does not send filter params to useKeys when no filter is active", () => {
    renderWithProviders(<VirtualKeysTable />);

    const lastCall = mockUseKeys.mock.calls[mockUseKeys.mock.calls.length - 1];
    expect(lastCall[2] ?? {}).toMatchObject({ userID: undefined, teamID: undefined, keyHash: undefined });
  });

  it("drops the filter from the useKeys query when Reset Filters is clicked", async () => {
    renderWithProviders(<VirtualKeysTable />);

    fireEvent.click(screen.getByRole("button", { name: "Filters" }));
    const userIdInput = await screen.findByPlaceholderText("Enter User ID...");
    fireEvent.change(userIdInput, { target: { value: "user-42" } });

    await waitFor(() => {
      expect(mockUseKeys).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ userID: "user-42" }));
    });

    fireEvent.click(screen.getByRole("button", { name: "Reset Filters" }));

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
      expect(screen.getByText("Showing 1 - 50 of 509 results")).toBeInTheDocument();
      expect(screen.getByText("Page 1 of 11")).toBeInTheDocument();
    });
  });

  it("reflects a narrowed total when a filtered fetch returns fewer results", async () => {
    mockUseKeys.mockReturnValue(keysResult([mockKey], { total_count: 1, total_pages: 1 }));

    renderWithProviders(<VirtualKeysTable />);

    await waitFor(() => {
      expect(screen.getByText("Showing 1 - 1 of 1 results")).toBeInTheDocument();
      expect(screen.getByText("Page 1 of 1")).toBeInTheDocument();
    });
  });
});

describe("refetch button", () => {
  it("should show Fetch button in normal state", () => {
    renderWithProviders(<VirtualKeysTable />);

    const fetchButton = screen.getByTitle("Fetch data");
    expect(fetchButton).toBeInTheDocument();
    expect(fetchButton).not.toBeDisabled();
    expect(screen.getByText("Fetch")).toBeInTheDocument();
  });

  it("should show Fetching state and keep table data visible during refetch", () => {
    mockUseKeys.mockReturnValue(keysResult([mockKey], {}, { isFetching: true }));

    renderWithProviders(<VirtualKeysTable />);

    expect(screen.getByText("Fetching")).toBeInTheDocument();
    expect(screen.getByTitle("Fetch data")).toBeDisabled();
    expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
    expect(screen.queryByText("🚅 Loading keys...")).not.toBeInTheDocument();
  });

  it("should call refetch when Fetch button is clicked", () => {
    const mockRefetch = vi.fn();
    mockUseKeys.mockReturnValue(keysResult([mockKey], {}, { refetch: mockRefetch }));

    renderWithProviders(<VirtualKeysTable />);

    fireEvent.click(screen.getByTitle("Fetch data"));

    expect(mockRefetch).toHaveBeenCalledTimes(1);
  });

  it("should show Fetch button enabled on error so user can retry", () => {
    mockUseKeys.mockReturnValue(keysResult([], {}, { data: null, isError: true }));

    renderWithProviders(<VirtualKeysTable />);

    const fetchButton = screen.getByTitle("Fetch data");
    expect(fetchButton).not.toBeDisabled();
    expect(screen.getByText("Fetch")).toBeInTheDocument();
  });
});

describe("Status column reflects key.blocked / scim_blocked metadata", () => {
  it("should render Active for a non-blocked key", async () => {
    mockUseKeys.mockReturnValue(keysResult([{ ...mockKey, blocked: false, metadata: {} }]));

    renderWithProviders(<VirtualKeysTable />);

    await waitFor(() => {
      expect(screen.getByTestId(`key-status-${mockKey.token_id}`)).toHaveTextContent("Active");
    });
  });

  it("should render Blocked when key.blocked is true", async () => {
    mockUseKeys.mockReturnValue(keysResult([{ ...mockKey, blocked: true, metadata: {} }]));

    renderWithProviders(<VirtualKeysTable />);

    await waitFor(() => {
      expect(screen.getByTestId(`key-status-${mockKey.token_id}`)).toHaveTextContent("Blocked");
    });
    expect(screen.queryByText(/Blocked by SCIM/i)).not.toBeInTheDocument();
  });

  it("should mark a SCIM-blocked key with the SCIM tooltip reason", async () => {
    mockUseKeys.mockReturnValue(keysResult([{ ...mockKey, blocked: true, metadata: { scim_blocked: true } }]));

    renderWithProviders(<VirtualKeysTable />);

    const tag = await screen.findByTestId(`key-status-${mockKey.token_id}`);
    expect(tag).toHaveTextContent("Blocked");

    act(() => {
      fireEvent.mouseEnter(tag);
    });
    await waitFor(() => {
      expect(screen.getByText(/Blocked by SCIM/i)).toBeInTheDocument();
    });
  });
});
