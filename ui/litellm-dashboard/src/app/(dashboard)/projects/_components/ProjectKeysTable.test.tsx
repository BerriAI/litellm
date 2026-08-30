import { describe, it, expect, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen } from "../../../../../tests/test-utils";
import { ProjectKeysTable } from "./ProjectKeysTable";
import { KeyResponse } from "@/components/key_team_helpers/key_list";

vi.mock("@/components/common_components/DefaultProxyAdminTag", () => ({
  default: ({ userId }: { userId: string }) => <span data-testid="owner-tag">{userId}</span>,
}));

const push = vi.fn();
vi.mock("next/navigation", async () => ({
  ...(await vi.importActual("next/navigation")),
  useRouter: () => ({ push }),
}));

const defaultProps = {
  totalCount: 0,
  isLoading: false,
  pagination: { pageIndex: 0, pageSize: 5 },
  onPaginationChange: vi.fn(),
};

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
    renderWithProviders(<ProjectKeysTable {...defaultProps} keys={[]} />);
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("should display 'No keys found' when the keys list is empty", () => {
    renderWithProviders(<ProjectKeysTable {...defaultProps} keys={[]} />);
    expect(screen.getByText("No keys found")).toBeInTheDocument();
  });

  it("should display the key alias when provided", () => {
    renderWithProviders(<ProjectKeysTable {...defaultProps} keys={[makeKey({ key_alias: "My API Key" })]} />);
    expect(screen.getByText("My API Key")).toBeInTheDocument();
  });

  it("should display '—' when the key alias is null", () => {
    // Provide a user_id so only the alias column shows "—" (not the owner column too)
    renderWithProviders(
      <ProjectKeysTable {...defaultProps} keys={[makeKey({ key_alias: null as any, user_id: "owner-1" })]} />,
    );
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("should link the key name to that key's detail on the Virtual Keys page", () => {
    renderWithProviders(
      <ProjectKeysTable {...defaultProps} keys={[makeKey({ token: "tok-abc123", key_alias: "My API Key" })]} />,
    );
    expect(screen.getByRole("link", { name: "My API Key" })).toHaveAttribute("href", "/ui/api-keys?key=tok-abc123");
  });

  it("should navigate to the key detail without a full page load when the key name is clicked", async () => {
    const user = userEvent.setup();
    push.mockClear();
    renderWithProviders(
      <ProjectKeysTable {...defaultProps} keys={[makeKey({ token: "tok-abc123", key_alias: "My API Key" })]} />,
    );
    await user.click(screen.getByRole("link", { name: "My API Key" }));
    expect(push).toHaveBeenCalledWith("/ui/api-keys?key=tok-abc123");
  });

  it("should still link a key that has no alias", () => {
    renderWithProviders(
      <ProjectKeysTable
        {...defaultProps}
        keys={[makeKey({ token: "tok-no-alias", key_alias: "", user_id: "owner-1" })]}
      />,
    );
    expect(screen.getByRole("link", { name: "—" })).toHaveAttribute("href", "/ui/api-keys?key=tok-no-alias");
  });

  it("should give each row a link to its own key", () => {
    const keys = [makeKey({ token: "tok-1", key_alias: "Key One" }), makeKey({ token: "tok-2", key_alias: "Key Two" })];
    renderWithProviders(<ProjectKeysTable {...defaultProps} keys={keys} />);
    expect(screen.getByRole("link", { name: "Key One" })).toHaveAttribute("href", "/ui/api-keys?key=tok-1");
    expect(screen.getByRole("link", { name: "Key Two" })).toHaveAttribute("href", "/ui/api-keys?key=tok-2");
  });

  it("should display the owner using user.user_email when available", () => {
    const key = makeKey({ user: { user_id: "u1", user_email: "alice@example.com", user_alias: null } });
    renderWithProviders(<ProjectKeysTable {...defaultProps} keys={[key]} />);
    expect(screen.getByTestId("owner-tag")).toHaveTextContent("alice@example.com");
  });

  it("should fall back to user_id when user.user_email is absent", () => {
    const key = makeKey({ user_id: "user-99" });
    renderWithProviders(<ProjectKeysTable {...defaultProps} keys={[key]} />);
    expect(screen.getByTestId("owner-tag")).toHaveTextContent("user-99");
  });

  it("should display 'Never' in the Last Active column when last_active is null", () => {
    renderWithProviders(<ProjectKeysTable {...defaultProps} keys={[makeKey({ last_active: null })]} />);
    expect(screen.getByText("Never")).toBeInTheDocument();
  });

  it("should display a formatted date in the Last Active column when last_active is provided", () => {
    renderWithProviders(
      <ProjectKeysTable {...defaultProps} keys={[makeKey({ last_active: "2024-06-15T10:00:00Z" })]} />,
    );
    expect(screen.queryByText("Never")).not.toBeInTheDocument();
  });

  it("should render multiple keys as separate rows", () => {
    const keys = [makeKey({ token: "tok-1", key_alias: "Key One" }), makeKey({ token: "tok-2", key_alias: "Key Two" })];
    renderWithProviders(<ProjectKeysTable {...defaultProps} keys={keys} />);
    expect(screen.getByText("Key One")).toBeInTheDocument();
    expect(screen.getByText("Key Two")).toBeInTheDocument();
  });

  it("should show skeleton rows while loading", () => {
    renderWithProviders(<ProjectKeysTable {...defaultProps} keys={[]} isLoading />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("No keys found")).not.toBeInTheDocument();
  });

  it("should show the server-side total in the pagination footer", () => {
    renderWithProviders(<ProjectKeysTable {...defaultProps} keys={[makeKey()]} totalCount={42} />);
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-5 of 42");
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 9");
  });

  it("should request the next page through the pagination footer", async () => {
    const user = userEvent.setup();
    const onPaginationChange = vi.fn();
    renderWithProviders(
      <ProjectKeysTable {...defaultProps} keys={[makeKey()]} totalCount={42} onPaginationChange={onPaginationChange} />,
    );
    await user.click(screen.getByTestId("pagination-next"));
    expect(onPaginationChange).toHaveBeenCalled();
  });
});
