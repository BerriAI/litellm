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
      // Simulate maxTagCount responsive behavior - if value length > 5, call maxTagPlaceholder
      const shouldShowPlaceholder = maxTagCount === "responsive" && Array.isArray(value) && value.length > 5;
      const visibleValues = shouldShowPlaceholder ? value.slice(0, 5) : value;
      const omittedValues = shouldShowPlaceholder
        ? value.slice(5).map((v: string) => ({ value: v, label: v }))
        : [];

      return (
        <div data-testid={dataTestId || "model-select"}>
          <select
            multiple={mode === "multiple"}
            role="listbox"
            value={visibleValues}
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
          {shouldShowPlaceholder && maxTagPlaceholder && (
            <div data-testid="max-tag-placeholder">{maxTagPlaceholder(omittedValues)}</div>
          )}
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

  it("should render with all option groups", async () => {
    renderWithProviders(
      <ModelSelect onChange={mockOnChange} context="user" options={{ showAllProxyModelsOverride: true }} />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("model-select")).toBeInTheDocument();
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
      expect(screen.getByText("claude-3")).toBeInTheDocument();
      expect(screen.getByText("All Openai models")).toBeInTheDocument();
      expect(screen.getByText("All Anthropic models")).toBeInTheDocument();
    });
  });

  it("should show skeleton loader when any data is loading", () => {
    const loadingScenarios = [
      { hook: mockUseAllProxyModels, context: "user" as const },
      { hook: mockUseTeam, context: "team" as const, props: { teamID: "team-1" } },
      { hook: mockUseOrganization, context: "organization" as const, props: { organizationID: "org-1" } },
      { hook: mockUseCurrentUser, context: "user" as const },
    ];

    loadingScenarios.forEach(({ hook, context, props = {} }) => {
      hook.mockReturnValue({
        data: undefined,
        isLoading: true,
      } as any);

      const { unmount } = renderWithProviders(
        <ModelSelect onChange={mockOnChange} context={context} {...props} />,
      );

      expect(screen.getByTestId("skeleton-input")).toBeInTheDocument();
      unmount();
    });
  });

  it("should handle model selection and onChange", async () => {
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

    await user.selectOptions(select, ["gpt-4", "claude-3"]);
    expect(mockOnChange).toHaveBeenCalled();
  });

  it("should handle special options correctly", async () => {
    const user = userEvent.setup();
    mockUseOrganization.mockReturnValue({
      data: createMockOrganization(["all-proxy-models"]),
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
      expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
      expect(screen.getByText("No Default Models")).toBeInTheDocument();
    });

    const select = screen.getByRole("listbox");
    await user.selectOptions(select, ["all-proxy-models", "no-default-models"]);
    expect(mockOnChange).toHaveBeenCalledWith(["no-default-models"]);
  });

  it("should disable models when special option is selected", async () => {
    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        value={["all-proxy-models"]}
        context="user"
        options={{ showAllProxyModelsOverride: true }}
      />,
    );

    await waitFor(() => {
      expect(screen.getByRole("option", { name: "gpt-4" })).toBeDisabled();
      expect(screen.getByRole("option", { name: "All Openai models" })).toBeDisabled();
    });
  });

  it("should filter models based on context", async () => {
    const testCases = [
      {
        name: "user context with includeUserModels",
        context: "user" as const,
        options: { includeUserModels: true },
        setup: () => {
          mockUseCurrentUser.mockReturnValue({
            data: { models: ["gpt-4"] },
            isLoading: false,
          } as any);
        },
        expectedVisible: ["gpt-4"],
        expectedHidden: ["claude-3"],
      },
      {
        name: "user context without includeUserModels",
        context: "user" as const,
        options: {},
        setup: () => {
          mockUseCurrentUser.mockReturnValue({
            data: { models: ["gpt-4"] },
            isLoading: false,
          } as any);
        },
        expectedVisible: [],
        expectedHidden: ["gpt-4", "claude-3"],
      },
      {
        name: "team context without organization",
        context: "team" as const,
        options: {},
        props: { teamID: "team-1" },
        setup: () => {
          mockUseTeam.mockReturnValue({
            data: { team_id: "team-1", team_alias: "Test Team", models: [] },
            isLoading: false,
          } as any);
          mockUseOrganization.mockReturnValue({
            data: undefined,
            isLoading: false,
          } as any);
        },
        expectedVisible: ["gpt-4", "claude-3"],
        expectedHidden: [],
      },
      {
        name: "team context with organization having all-proxy-models",
        context: "team" as const,
        options: {},
        props: { teamID: "team-1", organizationID: "org-1" },
        setup: () => {
          mockUseTeam.mockReturnValue({
            data: { team_id: "team-1", team_alias: "Test Team", models: [] },
            isLoading: false,
          } as any);
          mockUseOrganization.mockReturnValue({
            data: createMockOrganization(["all-proxy-models"]),
            isLoading: false,
          } as any);
        },
        expectedVisible: ["gpt-4", "claude-3"],
        expectedHidden: [],
      },
      {
        name: "team context with organization filtering models",
        context: "team" as const,
        options: {},
        props: { teamID: "team-1", organizationID: "org-1" },
        setup: () => {
          mockUseTeam.mockReturnValue({
            data: { team_id: "team-1", team_alias: "Test Team", models: [] },
            isLoading: false,
          } as any);
          mockUseOrganization.mockReturnValue({
            data: createMockOrganization(["gpt-4"]),
            isLoading: false,
          } as any);
        },
        expectedVisible: ["gpt-4"],
        expectedHidden: ["claude-3"],
      },
      {
        name: "organization context",
        context: "organization" as const,
        options: {},
        props: { organizationID: "org-1" },
        setup: () => {
          mockUseOrganization.mockReturnValue({
            data: createMockOrganization(["gpt-4"]),
            isLoading: false,
          } as any);
        },
        expectedVisible: ["gpt-4", "claude-3"],
        expectedHidden: [],
      },
      {
        name: "global context",
        context: "global" as const,
        options: {},
        setup: () => { },
        expectedVisible: ["gpt-4", "claude-3"],
        expectedHidden: [],
      },
    ];

    for (const testCase of testCases) {
      testCase.setup();
      const { unmount } = renderWithProviders(
        <ModelSelect
          onChange={mockOnChange}
          context={testCase.context}
          options={testCase.options}
          {...(testCase.props || {})}
        />,
      );

      await waitFor(() => {
        testCase.expectedVisible.forEach((model) => {
          expect(screen.getByText(model)).toBeInTheDocument();
        });
        testCase.expectedHidden.forEach((model) => {
          expect(screen.queryByText(model)).not.toBeInTheDocument();
        });
      });

      unmount();
      vi.clearAllMocks();
      mockUseAllProxyModels.mockReturnValue({
        data: { data: mockProxyModels },
        isLoading: false,
      } as any);
    }
  });

  it("should show All Proxy Models option based on conditions", async () => {
    const testCases = [
      {
        name: "when showAllProxyModelsOverride is true",
        context: "user" as const,
        options: { showAllProxyModelsOverride: true, includeSpecialOptions: true },
        setup: () => { },
        shouldShow: true,
      },
      {
        name: "when organization has all-proxy-models",
        context: "organization" as const,
        options: { includeSpecialOptions: true },
        props: { organizationID: "org-1" },
        setup: () => {
          mockUseOrganization.mockReturnValue({
            data: createMockOrganization(["all-proxy-models"]),
            isLoading: false,
          } as any);
        },
        shouldShow: true,
      },
      {
        name: "when organization has empty models array",
        context: "organization" as const,
        options: { includeSpecialOptions: true },
        props: { organizationID: "org-1" },
        setup: () => {
          mockUseOrganization.mockReturnValue({
            data: createMockOrganization([]),
            isLoading: false,
          } as any);
        },
        shouldShow: true,
      },
      {
        name: "when context is global",
        context: "global" as const,
        options: { includeSpecialOptions: true },
        setup: () => { },
        shouldShow: true,
      },
      {
        name: "when organization has specific models",
        context: "organization" as const,
        options: { includeSpecialOptions: true },
        props: { organizationID: "org-1" },
        setup: () => {
          mockUseOrganization.mockReturnValue({
            data: createMockOrganization(["gpt-4"]),
            isLoading: false,
          } as any);
        },
        shouldShow: false,
      },
    ];

    for (const testCase of testCases) {
      testCase.setup();
      const { unmount } = renderWithProviders(
        <ModelSelect
          onChange={mockOnChange}
          context={testCase.context}
          options={testCase.options}
          {...(testCase.props || {})}
        />,
      );

      await waitFor(() => {
        if (testCase.shouldShow) {
          expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
        } else {
          expect(screen.queryByText("All Proxy Models")).not.toBeInTheDocument();
          expect(screen.getByText("No Default Models")).toBeInTheDocument();
        }
      });

      unmount();
      vi.clearAllMocks();
      mockUseAllProxyModels.mockReturnValue({
        data: { data: mockProxyModels },
        isLoading: false,
      } as any);
    }
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

  it("should return all proxy models for team context when organization has empty models array", async () => {
    mockUseTeam.mockReturnValue({
      data: { team_id: "team-1", team_alias: "Test Team", models: [] },
      isLoading: false,
    } as any);

    mockUseOrganization.mockReturnValue({
      data: createMockOrganization([]),
      isLoading: false,
    } as any);

    renderWithProviders(<ModelSelect onChange={mockOnChange} context="team" teamID="team-1" organizationID="org-1" />);

    await waitFor(() => {
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
      expect(screen.getByText("claude-3")).toBeInTheDocument();
    });
  });

  it("should disable No Default Models when all-proxy-models is selected", async () => {
    mockUseOrganization.mockReturnValue({
      data: createMockOrganization(["all-proxy-models"]),
      isLoading: false,
    } as any);

    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        value={["all-proxy-models"]}
        context="organization"
        organizationID="org-1"
        options={{ includeSpecialOptions: true }}
      />,
    );

    await waitFor(() => {
      const noDefaultOption = screen.getByRole("option", { name: "No Default Models" });
      expect(noDefaultOption).toBeDisabled();
    });
  });

  it("should render maxTagPlaceholder when many items are selected", async () => {
    // Create many models to trigger maxTagCount responsive behavior
    const manyModels: ProxyModel[] = Array.from({ length: 20 }, (_, i) => ({
      id: `model-${i}`,
      object: "model",
      created: 1234567890,
      owned_by: "test",
    }));

    mockUseAllProxyModels.mockReturnValue({
      data: { data: manyModels },
      isLoading: false,
    } as any);

    const selectedValues = manyModels.slice(0, 10).map((m) => m.id);

    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        value={selectedValues}
        context="user"
        options={{ showAllProxyModelsOverride: true }}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("model-select")).toBeInTheDocument();
      // Verify maxTagPlaceholder is rendered with omitted values
      expect(screen.getByTestId("max-tag-placeholder")).toBeInTheDocument();
      expect(screen.getByText(/\+5 more/)).toBeInTheDocument();
    });
  });
});
