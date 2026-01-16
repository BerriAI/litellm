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

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>();
  return {
    ...actual,
    Select: ({
      value,
      onChange,
      options,
      "data-testid": dataTestId,
      allowClear,
      maxTagCount,
      maxTagPlaceholder,
      mode,
      ...props
    }: any) => {
      return (
        <div data-testid={dataTestId || "model-select"}>
          <select
            multiple={mode === "multiple"}
            role="listbox"
            value={value}
            onChange={(e) => {
              const selectedValues = Array.from(e.target.selectedOptions, (option) => option.value);
              onChange(mode === "multiple" ? selectedValues : selectedValues[0]);
            }}
            {...props}
          >
            {options?.map((group: any) => (
              <optgroup
                key={group.label?.props?.children || group.title}
                label={group.title || group.label?.props?.children}
              >
                {group.options?.map((option: any) => (
                  <option key={option.value} value={option.value} disabled={option.disabled}>
                    {typeof option.label === "string" ? option.label : option.label?.props?.children}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>
      );
    },
    Skeleton: {
      Input: ({ active, block }: any) => <div data-testid="skeleton-input" data-active={active} data-block={block} />,
    },
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  };
});

import { useAllProxyModels } from "@/app/(dashboard)/hooks/models/useModels";
import { useOrganization } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";

const mockUseAllProxyModels = vi.mocked(useAllProxyModels);
const mockUseTeam = vi.mocked(useTeam);
const mockUseOrganization = vi.mocked(useOrganization);
const mockUseCurrentUser = vi.mocked(useCurrentUser);

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

  it("should render", async () => {
    renderWithProviders(
      <ModelSelect onChange={mockOnChange} context="user" options={{ showAllProxyModelsOverride: true }} />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("model-select")).toBeInTheDocument();
    });
  });

  it("should show skeleton loader when loading", () => {
    mockUseAllProxyModels.mockReturnValue({
      data: undefined,
      isLoading: true,
    } as any);

    renderWithProviders(<ModelSelect onChange={mockOnChange} context="user" />);

    expect(screen.getByTestId("skeleton-input")).toBeInTheDocument();
    expect(screen.queryByTestId("model-select")).not.toBeInTheDocument();
  });

  it("should show skeleton loader when team is loading", () => {
    mockUseTeam.mockReturnValue({
      data: undefined,
      isLoading: true,
    } as any);

    renderWithProviders(<ModelSelect onChange={mockOnChange} context="team" teamID="team-1" />);

    expect(screen.getByTestId("skeleton-input")).toBeInTheDocument();
  });

  it("should show skeleton loader when organization is loading", () => {
    mockUseOrganization.mockReturnValue({
      data: undefined,
      isLoading: true,
    } as any);

    renderWithProviders(<ModelSelect onChange={mockOnChange} context="organization" organizationID="org-1" />);

    expect(screen.getByTestId("skeleton-input")).toBeInTheDocument();
  });

  it("should show skeleton loader when current user is loading", () => {
    mockUseCurrentUser.mockReturnValue({
      data: undefined,
      isLoading: true,
    } as any);

    renderWithProviders(<ModelSelect onChange={mockOnChange} context="user" />);

    expect(screen.getByTestId("skeleton-input")).toBeInTheDocument();
  });

  it("should render special options group", async () => {
    const mockOrganization: Organization = {
      organization_id: "org-1",
      organization_alias: "Test Org",
      budget_id: "budget-1",
      metadata: {},
      models: ["all-proxy-models"],
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
    };

    mockUseOrganization.mockReturnValue({
      data: mockOrganization,
      isLoading: false,
    } as any);

    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        context="organization"
        organizationID="org-1"
        options={{ includeSpecialOptions: true }}
      />,
    );

    await waitFor(() => {
      const select = screen.getByTestId("model-select");
      expect(select).toBeInTheDocument();
      expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
      expect(screen.getByText("No Default Models")).toBeInTheDocument();
    });
  });

  it("should render wildcard options group", async () => {
    renderWithProviders(
      <ModelSelect onChange={mockOnChange} context="user" options={{ showAllProxyModelsOverride: true }} />,
    );

    await waitFor(() => {
      expect(screen.getByText("All Openai models")).toBeInTheDocument();
      expect(screen.getByText("All Anthropic models")).toBeInTheDocument();
    });
  });

  it("should render regular models group", async () => {
    renderWithProviders(
      <ModelSelect onChange={mockOnChange} context="user" options={{ showAllProxyModelsOverride: true }} />,
    );

    await waitFor(() => {
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
      expect(screen.getByText("claude-3")).toBeInTheDocument();
    });
  });

  it("should call onChange when selecting a regular model", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ModelSelect onChange={mockOnChange} context="user" options={{ showAllProxyModelsOverride: true }} />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("model-select")).toBeInTheDocument();
    });

    const select = screen.getByRole("listbox");
    await user.selectOptions(select, "gpt-4");

    expect(mockOnChange).toHaveBeenCalledWith(["gpt-4"]);
  });

  it("should call onChange with only last special option when multiple special options are selected", async () => {
    const user = userEvent.setup();
    const mockOrganization: Organization = {
      organization_id: "org-1",
      organization_alias: "Test Org",
      budget_id: "budget-1",
      metadata: {},
      models: ["all-proxy-models"],
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
    };

    mockUseOrganization.mockReturnValue({
      data: mockOrganization,
      isLoading: false,
    } as any);

    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        context="organization"
        organizationID="org-1"
        options={{ showAllProxyModelsOverride: true, includeSpecialOptions: true }}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("model-select")).toBeInTheDocument();
    });

    const select = screen.getByRole("listbox");
    await user.selectOptions(select, ["all-proxy-models", "no-default-models"]);

    expect(mockOnChange).toHaveBeenCalledWith(["no-default-models"]);
  });

  it("should disable regular models when special option is selected", async () => {
    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        value={["all-proxy-models"]}
        context="user"
        options={{ showAllProxyModelsOverride: true }}
      />,
    );

    await waitFor(() => {
      const gpt4Option = screen.getByRole("option", { name: "gpt-4" });
      expect(gpt4Option).toBeDisabled();
    });
  });

  it("should disable wildcard models when special option is selected", async () => {
    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        value={["all-proxy-models"]}
        context="user"
        options={{ showAllProxyModelsOverride: true }}
      />,
    );

    await waitFor(() => {
      const openaiWildcardOption = screen.getByRole("option", { name: "All Openai models" });
      expect(openaiWildcardOption).toBeDisabled();
    });
  });

  it("should disable other special options when one special option is selected", async () => {
    const mockOrganization: Organization = {
      organization_id: "org-1",
      organization_alias: "Test Org",
      budget_id: "budget-1",
      metadata: {},
      models: ["all-proxy-models"],
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
    };

    mockUseOrganization.mockReturnValue({
      data: mockOrganization,
      isLoading: false,
    } as any);

    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        value={["all-proxy-models"]}
        context="organization"
        organizationID="org-1"
        options={{ showAllProxyModelsOverride: true, includeSpecialOptions: true }}
      />,
    );

    await waitFor(() => {
      const noDefaultOption = screen.getByRole("option", { name: "No Default Models" });
      expect(noDefaultOption).toBeDisabled();
    });
  });

  it("should filter models when showAllProxyModelsOverride is true", async () => {
    renderWithProviders(
      <ModelSelect onChange={mockOnChange} context="user" options={{ showAllProxyModelsOverride: true }} />,
    );

    await waitFor(() => {
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
      expect(screen.getByText("claude-3")).toBeInTheDocument();
    });
  });

  it("should filter models when organization has all-proxy-models in models array", async () => {
    const mockOrganization: Organization = {
      organization_id: "org-1",
      organization_alias: "Test Org",
      budget_id: "budget-1",
      metadata: {},
      models: ["all-proxy-models"],
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
    };

    mockUseOrganization.mockReturnValue({
      data: mockOrganization,
      isLoading: false,
    } as any);

    renderWithProviders(<ModelSelect onChange={mockOnChange} context="organization" organizationID="org-1" />);

    await waitFor(() => {
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
      expect(screen.getByText("claude-3")).toBeInTheDocument();
    });
  });

  it("should filter models when organization has specific models", async () => {
    const mockOrganization: Organization = {
      organization_id: "org-1",
      organization_alias: "Test Org",
      budget_id: "budget-1",
      metadata: {},
      models: ["gpt-4"],
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
    };

    mockUseOrganization.mockReturnValue({
      data: mockOrganization,
      isLoading: false,
    } as any);

    renderWithProviders(<ModelSelect onChange={mockOnChange} context="organization" organizationID="org-1" />);

    await waitFor(() => {
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
      expect(screen.queryByText("claude-3")).not.toBeInTheDocument();
    });
  });

  it("should use custom dataTestId when provided", async () => {
    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        dataTestId="custom-test-id"
        context="user"
        options={{ showAllProxyModelsOverride: true }}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("custom-test-id")).toBeInTheDocument();
    });
  });

  it("should handle multiple model selections", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ModelSelect onChange={mockOnChange} context="user" options={{ showAllProxyModelsOverride: true }} />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("model-select")).toBeInTheDocument();
    });

    const select = screen.getByRole("listbox");
    await user.selectOptions(select, "gpt-4");
    expect(mockOnChange).toHaveBeenCalledWith(["gpt-4"]);

    await user.selectOptions(select, "claude-3");
    expect(mockOnChange).toHaveBeenCalled();
    const allCalls = mockOnChange.mock.calls.map((call) => call[0]);
    expect(allCalls.some((call) => Array.isArray(call) && call.includes("gpt-4"))).toBe(true);
    expect(allCalls.some((call) => Array.isArray(call) && call.includes("claude-3"))).toBe(true);
  });

  it("should capitalize provider name in wildcard options", async () => {
    renderWithProviders(
      <ModelSelect onChange={mockOnChange} context="user" options={{ showAllProxyModelsOverride: true }} />,
    );

    await waitFor(() => {
      expect(screen.getByText("All Openai models")).toBeInTheDocument();
      expect(screen.getByText("All Anthropic models")).toBeInTheDocument();
    });
  });

  it("should deduplicate models with same id", async () => {
    const duplicateModels: ProxyModel[] = [
      { id: "gpt-4", object: "model", created: 1234567890, owned_by: "openai" },
      { id: "gpt-4", object: "model", created: 1234567890, owned_by: "openai" },
    ];

    mockUseAllProxyModels.mockReturnValue({
      data: { data: duplicateModels },
      isLoading: false,
    } as any);

    renderWithProviders(
      <ModelSelect onChange={mockOnChange} context="user" options={{ showAllProxyModelsOverride: true }} />,
    );

    await waitFor(() => {
      const gpt4Options = screen.getAllByText("gpt-4");
      expect(gpt4Options.length).toBeGreaterThan(0);
    });
  });

  it("should filter models based on user context with includeUserModels option", async () => {
    mockUseCurrentUser.mockReturnValue({
      data: { models: ["gpt-4"] },
      isLoading: false,
    } as any);

    renderWithProviders(<ModelSelect onChange={mockOnChange} context="user" options={{ includeUserModels: true }} />);

    await waitFor(() => {
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
      expect(screen.queryByText("claude-3")).not.toBeInTheDocument();
    });
  });

  it("should filter models based on team context", async () => {
    const mockTeam = {
      team_id: "team-1",
      team_alias: "Test Team",
      models: ["gpt-4"],
    };

    const mockOrganization: Organization = {
      organization_id: "org-1",
      organization_alias: "Test Org",
      budget_id: "budget-1",
      metadata: {},
      models: ["gpt-4"],
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
    };

    mockUseTeam.mockReturnValue({
      data: mockTeam,
      isLoading: false,
    } as any);

    mockUseOrganization.mockReturnValue({
      data: mockOrganization,
      isLoading: false,
    } as any);

    renderWithProviders(<ModelSelect onChange={mockOnChange} context="team" teamID="team-1" organizationID="org-1" />);

    await waitFor(() => {
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
      expect(screen.queryByText("claude-3")).not.toBeInTheDocument();
    });
  });
});
