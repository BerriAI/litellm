import { screen, waitFor } from "@testing-library/react";
import { vi, it, expect } from "vitest";
import { renderWithProviders } from "../../tests/test-utils";
import { AllKeysTable } from "./all_keys_table";
import { KeyResponse, Team } from "./key_team_helpers/key_list";
import { Organization } from "./networking";

// Mock network calls
vi.mock("./networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./networking")>();
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

it("should render AllKeysTable component", () => {
  const mockProps = {
    keys: [mockKey],
    setKeys: vi.fn(),
    isLoading: false,
    pagination: {
      currentPage: 1,
      totalPages: 1,
      totalCount: 1,
    },
    onPageChange: vi.fn(),
    pageSize: 50,
    teams: [mockTeam],
    selectedTeam: null,
    setSelectedTeam: vi.fn(),
    selectedKeyAlias: null,
    setSelectedKeyAlias: vi.fn(),
    accessToken: "test-token",
    userID: "user-1",
    userRole: "admin",
    organizations: [mockOrganization],
    setCurrentOrg: vi.fn(),
    premiumUser: false,
  };

  renderWithProviders(<AllKeysTable {...mockProps} />);

  expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
});

it("should display key information correctly", async () => {
  const mockProps = {
    keys: [mockKey],
    setKeys: vi.fn(),
    isLoading: false,
    pagination: {
      currentPage: 1,
      totalPages: 1,
      totalCount: 1,
    },
    onPageChange: vi.fn(),
    pageSize: 50,
    teams: [mockTeam],
    selectedTeam: null,
    setSelectedTeam: vi.fn(),
    selectedKeyAlias: null,
    setSelectedKeyAlias: vi.fn(),
    accessToken: "test-token",
    userID: "user-1",
    userRole: "admin",
    organizations: [mockOrganization],
    setCurrentOrg: vi.fn(),
    premiumUser: false,
  };

  renderWithProviders(<AllKeysTable {...mockProps} />);

  await waitFor(() => {
    expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
    expect(screen.getByText("Test Team")).toBeInTheDocument();
    expect(screen.getByText("5.5000")).toBeInTheDocument();
  });
});
