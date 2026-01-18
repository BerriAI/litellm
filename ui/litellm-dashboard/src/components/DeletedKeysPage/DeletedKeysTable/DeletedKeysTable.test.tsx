import { screen } from "@testing-library/react";
import { vi, it, expect, beforeEach } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import { DeletedKeysTable } from "./DeletedKeysTable";
import { DeletedKeyResponse } from "@/app/(dashboard)/hooks/keys/useKeys";

const mockDeletedKey: DeletedKeyResponse = {
  token: "sk-1234567890abcdef",
  token_id: "key-1",
  key_name: "test-key",
  key_alias: "Test Key Alias",
  spend: 5.5,
  max_budget: 100,
  expires: "2024-12-31T23:59:59Z",
  models: ["gpt-3.5-turbo"],
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
  model_spend: {},
  model_max_budget: {},
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
  team_models: ["gpt-3.5-turbo"],
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
  deleted_at: "2024-11-15T10:00:00Z",
  deleted_by: "user-1",
};

beforeEach(() => {
  vi.clearAllMocks();
});

it("should render DeletedKeysTable component", () => {
  renderWithProviders(
    <DeletedKeysTable
      keys={[mockDeletedKey]}
      totalCount={1}
      isLoading={false}
      isFetching={false}
      pageIndex={0}
      pageSize={50}
      onPageChange={vi.fn()}
    />,
  );

  expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
});

it("should display key information correctly", () => {
  renderWithProviders(
    <DeletedKeysTable
      keys={[mockDeletedKey]}
      totalCount={1}
      isLoading={false}
      isFetching={false}
      pageIndex={0}
      pageSize={50}
      onPageChange={vi.fn()}
    />,
  );

  expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
  expect(screen.getByText("sk-1234567890abcdef")).toBeInTheDocument();
  expect(screen.getByText("Showing 1 - 1 of 1 results")).toBeInTheDocument();
});
