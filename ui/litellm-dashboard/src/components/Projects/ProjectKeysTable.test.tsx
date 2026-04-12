import { describe, it, expect, vi } from "vitest";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import { ProjectKeysTable } from "./ProjectKeysTable";
import { KeyResponse } from "@/components/key_team_helpers/key_list";

vi.mock("@/components/common_components/DefaultProxyAdminTag", () => ({
  default: ({ userId }: { userId: string }) => <span data-testid="owner-tag">{userId}</span>,
}));

function makeKey(overrides: Partial<KeyResponse> = {}): KeyResponse {
  return {
    token: "tok-abc123",
    token_id: "tid-abc123",
    key_name: "sk-...abc",
    key_alias: "Test Key",
    spend: 0,
    max_budget: 0,
    expires: "",
    models: [],
    aliases: {},
    config: {},
    user_id: null as any,
    team_id: null,
    project_id: null,
    max_parallel_requests: 0,
    metadata: {},
    tpm_limit: 0,
    rpm_limit: 0,
    duration: "",
    budget_duration: "",
    budget_reset_at: "",
    allowed_cache_controls: [],
    allowed_routes: [],
    permissions: {},
    model_spend: {},
    model_max_budget: {},
    soft_budget_cooldown: false,
    blocked: false,
    litellm_budget_table: {},
    organization_id: null,
    created_at: "2024-03-01T00:00:00Z",
    updated_at: "2024-03-01T00:00:00Z",
    last_active: null,
    team_spend: 0,
    team_alias: "",
    team_tpm_limit: 0,
    team_rpm_limit: 0,
    team_max_budget: 0,
    team_models: [],
    team_blocked: false,
    soft_budget: 0,
    team_model_aliases: {},
    team_member_spend: 0,
    team_metadata: {},
    end_user_id: "",
    end_user_tpm_limit: 0,
    end_user_rpm_limit: 0,
    end_user_max_budget: 0,
    last_refreshed_at: 0,
    api_key: "",
    user_role: "user",
    rpm_limit_per_model: {},
    tpm_limit_per_model: {},
    user_tpm_limit: 0,
    user_rpm_limit: 0,
    user_email: "",
    ...overrides,
  } as KeyResponse;
}

describe("ProjectKeysTable", () => {
  it("should render", () => {
    renderWithProviders(<ProjectKeysTable keys={[]} />);
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("should display 'No keys found' when the keys list is empty", () => {
    renderWithProviders(<ProjectKeysTable keys={[]} />);
    expect(screen.getByText("No keys found")).toBeInTheDocument();
  });

  it("should display the key alias when provided", () => {
    renderWithProviders(<ProjectKeysTable keys={[makeKey({ key_alias: "My API Key" })]} />);
    expect(screen.getByText("My API Key")).toBeInTheDocument();
  });

  it("should display '—' when the key alias is null", () => {
    // Provide a user_id so only the alias column shows "—" (not the owner column too)
    renderWithProviders(
      <ProjectKeysTable keys={[makeKey({ key_alias: null as any, user_id: "owner-1" })]} />
    );
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("should display the owner using user.user_email when available", () => {
    const key = makeKey({ user: { user_id: "u1", user_email: "alice@example.com" } });
    renderWithProviders(<ProjectKeysTable keys={[key]} />);
    expect(screen.getByTestId("owner-tag")).toHaveTextContent("alice@example.com");
  });

  it("should fall back to user_id when user.user_email is absent", () => {
    const key = makeKey({ user_id: "user-99" });
    renderWithProviders(<ProjectKeysTable keys={[key]} />);
    expect(screen.getByTestId("owner-tag")).toHaveTextContent("user-99");
  });

  it("should display 'Never' in the Last Active column when last_active is null", () => {
    renderWithProviders(<ProjectKeysTable keys={[makeKey({ last_active: null })]} />);
    expect(screen.getByText("Never")).toBeInTheDocument();
  });

  it("should display a formatted date in the Last Active column when last_active is provided", () => {
    renderWithProviders(
      <ProjectKeysTable keys={[makeKey({ last_active: "2024-06-15T10:00:00Z" })]} />
    );
    expect(screen.queryByText("Never")).not.toBeInTheDocument();
  });

  it("should render multiple keys as separate rows", () => {
    const keys = [
      makeKey({ token: "tok-1", key_alias: "Key One" }),
      makeKey({ token: "tok-2", key_alias: "Key Two" }),
    ];
    renderWithProviders(<ProjectKeysTable keys={keys} />);
    expect(screen.getByText("Key One")).toBeInTheDocument();
    expect(screen.getByText("Key Two")).toBeInTheDocument();
  });
});
