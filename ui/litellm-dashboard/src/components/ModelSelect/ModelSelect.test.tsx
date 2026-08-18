import type { ProxyModel } from "@/app/(dashboard)/hooks/models/useModels";
import type { Organization } from "@/components/networking";
import { screen } from "@testing-library/react";
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

const openModelList = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getAllByRole("combobox")[0]);
  await screen.findByRole("listbox");
};

const expectOffered = (label: string) => expect(screen.queryAllByText(label).length).toBeGreaterThan(0);
const expectNotOffered = (label: string) => expect(screen.queryAllByText(label)).toHaveLength(0);

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

  it("should offer every model and wildcard under its group heading", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ModelSelect onChange={mockOnChange} context="user" options={{ showAllProxyModelsOverride: true }} />,
    );

    await openModelList(user);

    expectOffered("Wildcard Options");
    expectOffered("gpt-4");
    expectOffered("claude-3");
    expectOffered("All Openai models");
    expectOffered("All Anthropic models");
  });

  it("should offer nothing to select while any dependency is loading", () => {
    const { unmount: unmountReady } = renderWithProviders(<ModelSelect onChange={mockOnChange} context="user" />);
    expect(screen.getAllByRole("combobox")).toHaveLength(1);
    unmountReady();

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

      const { unmount } = renderWithProviders(<ModelSelect onChange={mockOnChange} context={context} {...props} />);

      expect(screen.queryAllByRole("combobox")).toHaveLength(0);
      unmount();
    });
  });

  it("should report the picked model to onChange", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ModelSelect onChange={mockOnChange} context="user" options={{ showAllProxyModelsOverride: true }} />,
    );

    await openModelList(user);
    await user.click(screen.getAllByText("gpt-4")[0]);

    expect(mockOnChange).toHaveBeenCalledWith(["gpt-4"]);
  });

  it("should append a second model to the existing selection", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        value={["gpt-4"]}
        context="user"
        options={{ showAllProxyModelsOverride: true }}
      />,
    );

    await openModelList(user);
    await user.click(screen.getAllByText("claude-3")[0]);

    expect(mockOnChange).toHaveBeenCalledWith(["gpt-4", "claude-3"]);
  });

  it("should offer both special options when they are enabled", async () => {
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

    await openModelList(user);

    expectOffered("Special Options");
    expectOffered("All Proxy Models");
    expectOffered("No Default Models");
  });

  it("should replace an existing selection when a special option is picked", async () => {
    const user = userEvent.setup();

    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        value={["gpt-4"]}
        context="user"
        options={{ showAllProxyModelsOverride: true, includeSpecialOptions: true }}
      />,
    );

    await openModelList(user);
    await user.click(screen.getAllByText("No Default Models")[0]);

    expect(mockOnChange).toHaveBeenCalledWith(["no-default-models"]);
  });

  it("should filter the offered models by context", async () => {
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
        setup: () => {},
        expectedVisible: ["gpt-4", "claude-3"],
        expectedHidden: [],
      },
    ];

    for (const testCase of testCases) {
      const user = userEvent.setup();
      testCase.setup();
      const { unmount } = renderWithProviders(
        <ModelSelect
          onChange={mockOnChange}
          context={testCase.context}
          options={testCase.options}
          {...(testCase.props || {})}
        />,
      );

      await openModelList(user);
      testCase.expectedVisible.forEach(expectOffered);
      testCase.expectedHidden.forEach(expectNotOffered);

      unmount();
      vi.clearAllMocks();
      mockUseAllProxyModels.mockReturnValue({
        data: { data: mockProxyModels },
        isLoading: false,
      } as any);
    }
  });

  it("should offer All Proxy Models only when the context allows it", async () => {
    const testCases = [
      {
        name: "when showAllProxyModelsOverride is true",
        context: "user" as const,
        options: { showAllProxyModelsOverride: true, includeSpecialOptions: true },
        setup: () => {},
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
        setup: () => {},
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
      const user = userEvent.setup();
      testCase.setup();
      const { unmount } = renderWithProviders(
        <ModelSelect
          onChange={mockOnChange}
          context={testCase.context}
          options={testCase.options}
          {...(testCase.props || {})}
        />,
      );

      await openModelList(user);
      if (testCase.shouldShow) {
        expectOffered("All Proxy Models");
      } else {
        expectNotOffered("All Proxy Models");
        expectOffered("No Default Models");
      }

      unmount();
      vi.clearAllMocks();
      mockUseAllProxyModels.mockReturnValue({
        data: { data: mockProxyModels },
        isLoading: false,
      } as any);
    }
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

    expect(await screen.findByTestId("custom-test-id")).toBeInTheDocument();
  });

  it("should return all proxy models for team context when organization has empty models array", async () => {
    const user = userEvent.setup();
    mockUseTeam.mockReturnValue({
      data: { team_id: "team-1", team_alias: "Test Team", models: [] },
      isLoading: false,
    } as any);

    mockUseOrganization.mockReturnValue({
      data: createMockOrganization([]),
      isLoading: false,
    } as any);

    renderWithProviders(<ModelSelect onChange={mockOnChange} context="team" teamID="team-1" organizationID="org-1" />);

    await openModelList(user);

    expectOffered("gpt-4");
    expectOffered("claude-3");
  });

  it("should not offer a special options group when includeSpecialOptions is omitted", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ModelSelect onChange={mockOnChange} context="global" />);

    await openModelList(user);

    expectNotOffered("Special Options");
    expectNotOffered("All Proxy Models");
    expectNotOffered("No Default Models");
    expectOffered("Models");
  });

  it("should mark models and wildcards unselectable while a special option is selected", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        value={["all-proxy-models"]}
        context="user"
        options={{ showAllProxyModelsOverride: true, includeSpecialOptions: true }}
      />,
    );

    await openModelList(user);

    expect(screen.getByRole("option", { name: "gpt-4" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("option", { name: "All Openai models" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("option", { name: "No Default Models" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("option", { name: "All Proxy Models" })).not.toHaveAttribute("aria-disabled", "true");
  });

  it("should list a duplicated proxy model only once", async () => {
    const user = userEvent.setup();
    mockUseAllProxyModels.mockReturnValue({
      data: {
        data: [
          { id: "gpt-4", object: "model", created: 1234567890, owned_by: "openai" },
          { id: "gpt-4", object: "model", created: 1234567890, owned_by: "openai" },
        ],
      },
      isLoading: false,
    } as any);

    renderWithProviders(
      <ModelSelect onChange={mockOnChange} context="user" options={{ showAllProxyModelsOverride: true }} />,
    );

    await openModelList(user);

    expect(screen.getAllByRole("option", { name: "gpt-4" })).toHaveLength(1);
  });

  it("should collapse selections past the chip limit into a labelled overflow count", async () => {
    const manyModels: ProxyModel[] = Array.from({ length: 8 }, (_, i) => ({
      id: `model-${i}`,
      object: "model",
      created: 1234567890,
      owned_by: "test",
    }));

    mockUseAllProxyModels.mockReturnValue({
      data: { data: manyModels },
      isLoading: false,
    } as any);

    renderWithProviders(
      <ModelSelect
        onChange={mockOnChange}
        value={manyModels.map((m) => m.id)}
        context="user"
        options={{ showAllProxyModelsOverride: true }}
      />,
    );

    expect(await screen.findByText("+3 more")).toBeInTheDocument();
    expect(screen.getByLabelText("model-0")).toBeInTheDocument();
    expect(screen.getByLabelText("model-4")).toBeInTheDocument();
    expect(screen.queryByLabelText("model-5")).not.toBeInTheDocument();
  });
});
