import type { ProxyModel } from "@/app/(dashboard)/hooks/models/useModels";
import type { Organization } from "@/components/networking";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { ModelSelect } from "./ModelSelect";

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeam: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganization: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/users/useCurrentUser", () => ({
  useCurrentUser: vi.fn(),
}));

import { useAllProxyModels } from "@/app/(dashboard)/hooks/models/useModels";
import { useOrganization } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";

const mockUseAllProxyModels = vi.mocked(useAllProxyModels);
const mockUseTeam = vi.mocked(useTeam);
const mockUseOrganization = vi.mocked(useOrganization);
const mockUseCurrentUser = vi.mocked(useCurrentUser);

const createMockOrganization = (models: string[]): Organization => ({
  organization_id: "org-1",
  organization_alias: "Test Org",
  budget_id: "budget-1",
  metadata: {},
  models,
  spend: 0,
  model_spend: {},
  created_at: "2024-01-01",
  created_by: "user-1",
  updated_at: "2024-01-01",
  updated_by: "user-1",
  litellm_budget_table: null,
  teams: null,
  users: null,
  members: null,
});

describe("ModelSelect", () => {
  const mockProxyModels: ProxyModel[] = [
    { id: "gpt-4", object: "model", created: 1234567890, owned_by: "openai" },
    { id: "claude-3", object: "model", created: 1234567890, owned_by: "anthropic" },
    { id: "openai/*", object: "model", created: 1234567890, owned_by: "openai" },
    { id: "anthropic/*", object: "model", created: 1234567890, owned_by: "anthropic" },
  ];

  const mockOnChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAllProxyModels.mockReturnValue({
      data: { data: mockProxyModels },
      isLoading: false,
    } as any);
    mockUseTeam.mockReturnValue({
      data: undefined,
      isLoading: false,
    } as any);
    mockUseOrganization.mockReturnValue({
      data: undefined,
      isLoading: false,
    } as any);
    mockUseCurrentUser.mockReturnValue({
      data: { models: [] },
      isLoading: false,
    } as any);
  });

  const openDropdown = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(screen.getByRole("combobox"));
    await waitFor(() => {
      expect(screen.getByPlaceholderText("Search models...")).toBeInTheDocument();
    });
  };

  it("should render with all option groups when opened", async () => {
    const user = userEvent.setup({ delay: null });
    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        context="user"
        options={{ showAllProxyModelsOverride: true }}
      />,
    );
    await openDropdown(user);

    expect(screen.getByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByText("claude-3")).toBeInTheDocument();
    expect(screen.getByText("All Openai models")).toBeInTheDocument();
    expect(screen.getByText("All Anthropic models")).toBeInTheDocument();
  });

  it("should show skeleton loader when any data is loading", () => {
    mockUseAllProxyModels.mockReturnValue({
      data: undefined,
      isLoading: true,
    } as any);

    renderWithProviders(<ModelSelect onChange={mockOnChange} context="user" />);
    // Skeleton renders as a div with the skeleton classes — no test id, but
    // the trigger combobox should not be rendered.
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  });

  it("should handle model selection and onChange", async () => {
    const user = userEvent.setup({ delay: null });
    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        context="user"
        options={{ showAllProxyModelsOverride: true }}
      />,
    );
    await openDropdown(user);

    const gpt4Button = screen.getByRole("button", { name: /^gpt-4$/ });
    await user.click(gpt4Button);
    expect(mockOnChange).toHaveBeenCalledWith(["gpt-4"]);
  });

  it("should handle special options correctly", async () => {
    const user = userEvent.setup({ delay: null });
    mockUseOrganization.mockReturnValue({
      data: createMockOrganization(["all-proxy-models"]),
      isLoading: false,
    } as any);

    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        context="organization"
        organizationID="org-1"
        options={{
          showAllProxyModelsOverride: true,
          includeSpecialOptions: true,
        }}
      />,
    );
    await openDropdown(user);

    expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
    expect(screen.getByText("No Default Models")).toBeInTheDocument();
  });

  it("should filter models based on context", async () => {
    const user = userEvent.setup({ delay: null });
    mockUseOrganization.mockReturnValue({
      data: createMockOrganization(["gpt-4"]),
      isLoading: false,
    } as any);
    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        context="team"
        organizationID="org-1"
      />,
    );
    await openDropdown(user);

    expect(screen.getByText("gpt-4")).toBeInTheDocument();
    expect(screen.queryByText("claude-3")).not.toBeInTheDocument();
  });

  it("should deduplicate models with same id", async () => {
    const user = userEvent.setup({ delay: null });
    const duplicatedModels: ProxyModel[] = [
      ...mockProxyModels,
      { id: "gpt-4", object: "model", created: 1234567891, owned_by: "openai" },
    ];
    mockUseAllProxyModels.mockReturnValue({
      data: { data: duplicatedModels },
      isLoading: false,
    } as any);

    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        context="user"
        options={{ showAllProxyModelsOverride: true }}
      />,
    );
    await openDropdown(user);

    const gpt4Matches = screen.getAllByText("gpt-4");
    expect(gpt4Matches).toHaveLength(1);
  });

  it("should use custom dataTestId when provided", async () => {
    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        context="user"
        dataTestId="my-custom-id"
      />,
    );
    expect(screen.getByTestId("my-custom-id")).toBeInTheDocument();
  });

  it("should return all proxy models for team context when organization has empty models array", async () => {
    const user = userEvent.setup({ delay: null });
    mockUseOrganization.mockReturnValue({
      data: createMockOrganization([]),
      isLoading: false,
    } as any);
    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        context="team"
        organizationID="org-1"
      />,
    );
    await openDropdown(user);

    expect(screen.getByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByText("claude-3")).toBeInTheDocument();
  });

  it("should render selected chip when value is provided", () => {
    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        value={["gpt-4"]}
        context="user"
        options={{ showAllProxyModelsOverride: true }}
      />,
    );
    // Chip appears inside the trigger button.
    expect(screen.getByRole("combobox")).toHaveTextContent("gpt-4");
  });

  it("should show +N more indicator when many items selected", () => {
    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        value={["gpt-4", "claude-3", "openai/*", "anthropic/*"]}
        context="user"
        options={{ showAllProxyModelsOverride: true }}
      />,
    );
    // First 3 shown as chips, rest in "+N more"
    expect(screen.getByText(/\+1 more/)).toBeInTheDocument();
  });
});
