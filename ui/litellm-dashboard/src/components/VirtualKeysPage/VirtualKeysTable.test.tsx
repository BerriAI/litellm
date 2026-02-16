import { screen, waitFor, fireEvent } from "@testing-library/react";
import { vi, it, expect, beforeEach, MockedFunction } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { VirtualKeysTable } from "./VirtualKeysTable";
import { KeyResponse, Team } from "../key_team_helpers/key_list";
import { Organization } from "../networking";
import { KeysResponse, useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useFilterLogic } from "../key_team_helpers/filter_logic";
import useTeams from "@/app/(dashboard)/hooks/useTeams";

// Mock network calls
vi.mock("./networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../networking")>();
  return {
    ...actual,
    userListCall: vi.fn().mockResolvedValue({
      users: [
        {
          user_id: "user-1",
          user_email: "user@example.com",
          user_role: "user",
        },
      ],
    }),
    teamListCall: vi.fn().mockResolvedValue([]),
  };
});

// Mock filter helpers
vi.mock("./key_team_helpers/filter_helpers", () => ({
  fetchAllKeyAliases: vi.fn().mockResolvedValue(["test-key-alias"]),
  fetchAllTeams: vi.fn().mockResolvedValue([
    {
      team_id: "team-1",
      team_alias: "Test Team",
    },
  ]),
  fetchAllOrganizations: vi.fn().mockResolvedValue([
    {
      organization_id: "org-1",
      organization_alias: "Test Organization",
    },
  ]),
}));

// Mock useKeys hook
vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: vi.fn(),
}));

// Mock useFilterLogic hook
vi.mock("../key_team_helpers/filter_logic", () => ({
  useFilterLogic: vi.fn(),
}));

// Mock useTeams hook (used by KeyInfoView)
vi.mock("@/app/(dashboard)/hooks/useTeams", () => ({
  default: vi.fn(),
}));

// Mock fetchTeams to prevent network calls
vi.mock("@/app/(dashboard)/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/app/(dashboard)/networking")>();
  return {
    ...actual,
    fetchTeams: vi.fn().mockResolvedValue([]),
  };
});

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
};

const mockOrganization: Organization = {
  organization_id: "org-1",
  organization_alias: "Test Organization",
  budget_id: "budget-1",
  metadata: {},
  models: ["gpt-3.5-turbo", "gpt-4"],
  spend: 100,
  model_spend: { "gpt-3.5-turbo": 50, "gpt-4": 50 },
  created_at: "2024-10-01T10:00:00Z",
  created_by: "user-1",
  updated_at: "2024-11-01T10:00:00Z",
  updated_by: "user-1",
  litellm_budget_table: {},
  teams: [],
  users: [],
  members: [],
};

// Mock hook implementations
const mockUseKeys = useKeys as MockedFunction<typeof useKeys>;
const mockUseFilterLogic = useFilterLogic as MockedFunction<typeof useFilterLogic>;
const mockUseTeams = useTeams as MockedFunction<typeof useTeams>;

beforeEach(() => {
  // Reset mocks before each test
  vi.clearAllMocks();

  // Setup default mock implementations
  mockUseKeys.mockReturnValue({
    data: {
      keys: [mockKey],
      total_count: 1,
      current_page: 1,
      total_pages: 1,
    } as KeysResponse,
    isPending: false,
    isFetching: false,
    refetch: vi.fn(),
  } as any);

  mockUseFilterLogic.mockReturnValue({
    filters: {
      "Team ID": "team-1",
      "Organization ID": "org-1",
      "Key Alias": "Test Key Alias",
      "User ID": "user-1",
      "User Email": "user@example.com",
      "User Role": "user",
      "Sort By": "created_at",
      "Sort Order": "desc",
    },
    filteredKeys: [mockKey],
    allKeyAliases: ["test-key-alias"],
    allTeams: [mockTeam],
    allOrganizations: [mockOrganization],
    handleFilterChange: vi.fn(),
    handleFilterReset: vi.fn(),
  });

  // Mock useTeams hook (used by KeyInfoView)
  mockUseTeams.mockReturnValue({
    teams: [mockTeam],
    setTeams: vi.fn(),
  });
});

it("should render VirtualKeysTable component", () => {
  const mockProps = {
    teams: [mockTeam],
    organizations: [mockOrganization],
    onSortChange: vi.fn(),
    currentSort: {
      sortBy: "created_at",
      sortOrder: "desc" as const,
    },
  };

  renderWithProviders(<VirtualKeysTable {...mockProps} />);

  expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
});

it("should display key information correctly", async () => {
  const mockProps = {
    teams: [mockTeam],
    organizations: [mockOrganization],
    onSortChange: vi.fn(),
    currentSort: {
      sortBy: "created_at",
      sortOrder: "desc" as const,
    },
  };

  renderWithProviders(<VirtualKeysTable {...mockProps} />);

  await waitFor(() => {
    expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
    expect(screen.getByText("Test Team")).toBeInTheDocument();
    expect(screen.getByText("5.5000")).toBeInTheDocument();
  });
});

it("should display user email correctly", async () => {
  const mockProps = {
    teams: [mockTeam],
    organizations: [mockOrganization],
    onSortChange: vi.fn(),
    currentSort: {
      sortBy: "created_at",
      sortOrder: "desc" as const,
    },
  };

  renderWithProviders(<VirtualKeysTable {...mockProps} />);

  await waitFor(() => {
    expect(screen.getByText("user@example.com")).toBeInTheDocument();
  });
});

it("should show skeleton loaders when isLoading is true", () => {
  // Mock loading state
  mockUseKeys.mockReturnValue({
    data: null,
    isPending: true,
    isFetching: true,
    refetch: vi.fn(),
  } as any);

  const mockProps = {
    teams: [mockTeam],
    organizations: [mockOrganization],
    onSortChange: vi.fn(),
    currentSort: {
      sortBy: "created_at",
      sortOrder: "desc" as const,
    },
  };

  renderWithProviders(<VirtualKeysTable {...mockProps} />);

  // Check that loading message is shown
  expect(screen.getByText("🚅 Loading keys...")).toBeInTheDocument();

  // Check that actual key data is not shown
  expect(screen.queryByText("Test Key Alias")).not.toBeInTheDocument();
  expect(screen.queryByText("Test Team")).not.toBeInTheDocument();
});

it("should show 'No keys found' message when filteredKeys is empty", () => {
  // Mock empty filteredKeys
  mockUseFilterLogic.mockReturnValue({
    filters: {
      "Team ID": "",
      "Organization ID": "",
      "Key Alias": "",
      "User ID": "",
      "Sort By": "created_at",
      "Sort Order": "desc",
    },
    filteredKeys: [],
    allKeyAliases: [],
    allTeams: [mockTeam],
    allOrganizations: [mockOrganization],
    handleFilterChange: vi.fn(),
    handleFilterReset: vi.fn(),
  });

  const mockProps = {
    teams: [mockTeam],
    organizations: [mockOrganization],
    onSortChange: vi.fn(),
    currentSort: {
      sortBy: "created_at",
      sortOrder: "desc" as const,
    },
  };

  renderWithProviders(<VirtualKeysTable {...mockProps} />);

  expect(screen.getByText("No keys found")).toBeInTheDocument();
});

it("should handle models with more than 3 entries to trigger expansion UI", () => {
  const keyWithManyModels = {
    ...mockKey,
    models: ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "claude-3", "claude-3-5-sonnet"],
  };

  mockUseFilterLogic.mockReturnValue({
    filters: {
      "Team ID": "",
      "Organization ID": "",
      "Key Alias": "",
      "User ID": "",
      "Sort By": "created_at",
      "Sort Order": "desc",
    },
    filteredKeys: [keyWithManyModels],
    allKeyAliases: ["test-key-alias"],
    allTeams: [mockTeam],
    allOrganizations: [mockOrganization],
    handleFilterChange: vi.fn(),
    handleFilterReset: vi.fn(),
  });

  const mockProps = {
    teams: [mockTeam],
    organizations: [mockOrganization],
    onSortChange: vi.fn(),
    currentSort: {
      sortBy: "created_at",
      sortOrder: "desc" as const,
    },
  };

  renderWithProviders(<VirtualKeysTable {...mockProps} />);

  // This test ensures the ChevronDownIcon import (line 6) is used
  // by having a key with > 3 models which triggers the expansion logic
  // that uses ChevronDownIcon and ChevronRightIcon
  expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
});

it("should render table headers correctly", () => {
  const mockProps = {
    teams: [mockTeam],
    organizations: [mockOrganization],
    onSortChange: vi.fn(),
    currentSort: {
      sortBy: "created_at",
      sortOrder: "desc" as const,
    },
  };

  renderWithProviders(<VirtualKeysTable {...mockProps} />);

  // Check that main headers are rendered (testing the header.isPlaceholder condition path)
  expect(screen.getByText("Key ID")).toBeInTheDocument();
  expect(screen.getByText("Key Alias")).toBeInTheDocument();
  expect(screen.getByText("Team Alias")).toBeInTheDocument();
  expect(screen.getByText("Models")).toBeInTheDocument();
  expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
});

it("should handle column resizing hover events", () => {
  const mockProps = {
    teams: [mockTeam],
    organizations: [mockOrganization],
    onSortChange: vi.fn(),
    currentSort: {
      sortBy: "created_at",
      sortOrder: "desc" as const,
    },
  };

  renderWithProviders(<VirtualKeysTable {...mockProps} />);

  // Find a header cell with data-header-id attribute
  const headerCell = document.querySelector("[data-header-id]") as HTMLElement;

  expect(headerCell).toBeInTheDocument();

  // Check that the resizer element exists within the header
  const resizer = headerCell?.querySelector(".resizer") as HTMLElement;
  expect(resizer).toBeInTheDocument();

  // Initially, resizer should have opacity 0
  expect(resizer.style.opacity).toBe("0");

  // Simulate mouse enter using fireEvent - should set opacity to 0.5 (lines 612-616)
  fireEvent.mouseEnter(headerCell);
  expect(resizer.style.opacity).toBe("0.5");

  // Simulate mouse leave using fireEvent - should set opacity back to 0 (lines 618-622)
  fireEvent.mouseLeave(headerCell);
  expect(resizer.style.opacity).toBe("0");
});

it("should open KeyInfoView when clicking on a key ID button", async () => {
  const mockProps = {
    teams: [mockTeam],
    organizations: [mockOrganization],
    onSortChange: vi.fn(),
    currentSort: {
      sortBy: "created_at",
      sortOrder: "desc" as const,
    },
  };

  renderWithProviders(<VirtualKeysTable {...mockProps} />);

  // Wait for the table to render
  await waitFor(() => {
    expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
  });

  // Verify table is visible before clicking - check for table-specific text
  expect(screen.getByText(/Showing.*results/)).toBeInTheDocument();

  // Find the key ID button (it shows the full token value, truncation is CSS-only)
  const keyIdButton = screen.getByText("sk-1234567890abcdef");
  expect(keyIdButton).toBeInTheDocument();

  // Click on the key ID button
  fireEvent.click(keyIdButton);

  // Wait for KeyInfoView to appear - check for unique elements that only exist in KeyInfoView
  await waitFor(() => {
    expect(screen.getByText("Back to Keys")).toBeInTheDocument();
    // KeyInfoView shows "Created:" or "Updated:" which is unique to it
    expect(screen.getByText(/Created:|Updated:/)).toBeInTheDocument();
  });

  // Verify that table-specific elements are no longer visible
  // The "Showing X of Y results" text should not be visible when KeyInfoView is open
  expect(screen.queryByText(/Showing.*results/)).not.toBeInTheDocument();
});

it("should display 'Default Proxy Admin' for user_id when value is 'default_user_id'", async () => {
  const keyWithDefaultUserId = {
    ...mockKey,
    user_id: "default_user_id",
  };

  mockUseFilterLogic.mockReturnValue({
    filters: {
      "Team ID": "",
      "Organization ID": "",
      "Key Alias": "",
      "User ID": "",
      "Sort By": "created_at",
      "Sort Order": "desc",
    },
    filteredKeys: [keyWithDefaultUserId],
    allKeyAliases: ["test-key-alias"],
    allTeams: [mockTeam],
    allOrganizations: [mockOrganization],
    handleFilterChange: vi.fn(),
    handleFilterReset: vi.fn(),
  });

  const mockProps = {
    teams: [mockTeam],
    organizations: [mockOrganization],
    onSortChange: vi.fn(),
    currentSort: {
      sortBy: "created_at",
      sortOrder: "desc" as const,
    },
  };

  renderWithProviders(<VirtualKeysTable {...mockProps} />);

  await waitFor(() => {
    expect(screen.getByText("Default Proxy Admin")).toBeInTheDocument();
  });
});

it("should display 'Default Proxy Admin' for created_by when value is 'default_user_id'", async () => {
  const keyWithDefaultCreatedBy = {
    ...mockKey,
    created_by: "default_user_id",
  };

  mockUseFilterLogic.mockReturnValue({
    filters: {
      "Team ID": "",
      "Organization ID": "",
      "Key Alias": "",
      "User ID": "",
      "Sort By": "created_at",
      "Sort Order": "desc",
    },
    filteredKeys: [keyWithDefaultCreatedBy],
    allKeyAliases: ["test-key-alias"],
    allTeams: [mockTeam],
    allOrganizations: [mockOrganization],
    handleFilterChange: vi.fn(),
    handleFilterReset: vi.fn(),
  });

  const mockProps = {
    teams: [mockTeam],
    organizations: [mockOrganization],
    onSortChange: vi.fn(),
    currentSort: {
      sortBy: "created_at",
      sortOrder: "desc" as const,
    },
  };

  renderWithProviders(<VirtualKeysTable {...mockProps} />);

  await waitFor(() => {
    // The created_by column should display "Default Proxy Admin"
    const defaultProxyAdminElements = screen.getAllByText("Default Proxy Admin");
    expect(defaultProxyAdminElements.length).toBeGreaterThan(0);
  });
});


it("should render table without crashing when models is null", async () => {
  const keyWithNullModels = {
    ...mockKey,
    models: null as unknown as string[],
  };

  mockUseFilterLogic.mockReturnValue({
    filters: {
      "Team ID": "",
      "Organization ID": "",
      "Key Alias": "",
      "User ID": "",
      "Sort By": "created_at",
      "Sort Order": "desc",
    },
    filteredKeys: [keyWithNullModels],
    allKeyAliases: ["test-key-alias"],
    allTeams: [mockTeam],
    allOrganizations: [mockOrganization],
    handleFilterChange: vi.fn(),
    handleFilterReset: vi.fn(),
  });

  const mockProps = {
    teams: [mockTeam],
    organizations: [mockOrganization],
    onSortChange: vi.fn(),
    currentSort: {
      sortBy: "created_at",
      sortOrder: "desc" as const,
    },
  };

  // This should not throw an error
  renderWithProviders(<VirtualKeysTable {...mockProps} />);

  await waitFor(() => {
    expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
  });
});

it("should render table without crashing when models is undefined", async () => {
  const keyWithUndefinedModels = {
    ...mockKey,
    models: undefined as unknown as string[],
  };

  mockUseFilterLogic.mockReturnValue({
    filters: {
      "Team ID": "",
      "Organization ID": "",
      "Key Alias": "",
      "User ID": "",
      "Sort By": "created_at",
      "Sort Order": "desc",
    },
    filteredKeys: [keyWithUndefinedModels],
    allKeyAliases: ["test-key-alias"],
    allTeams: [mockTeam],
    allOrganizations: [mockOrganization],
    handleFilterChange: vi.fn(),
    handleFilterReset: vi.fn(),
  });

  const mockProps = {
    teams: [mockTeam],
    organizations: [mockOrganization],
    onSortChange: vi.fn(),
    currentSort: {
      sortBy: "created_at",
      sortOrder: "desc" as const,
    },
  };

  // This should not throw an error
  renderWithProviders(<VirtualKeysTable {...mockProps} />);

  await waitFor(() => {
    expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
  });
});
