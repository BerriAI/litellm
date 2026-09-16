import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useCustomers } from "@/app/(dashboard)/hooks/customers/useCustomers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useIsOrgAdmin from "@/app/(dashboard)/hooks/useIsOrgAdmin";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";
import { useInfiniteUsers } from "@/app/(dashboard)/hooks/users/useUsers";
import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import type { Organization } from "@/components/networking";
import * as networking from "@/components/networking";
import type { TagListResponse } from "@/components/tag_management/types";
import UsagePage from "./UsagePageView";

// Polyfill ResizeObserver for test environment
beforeAll(() => {
  if (typeof window !== "undefined" && !window.ResizeObserver) {
    window.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as any;
  }
});

// Mock the networking module
vi.mock("@/components/networking", () => ({
  userDailyActivityCall: vi.fn(),
  userDailyActivityAggregatedCall: vi.fn(),
  gatewayDailyActivityCall: vi.fn(),
  tagListCall: vi.fn(),
}));

// Mock child components to simplify testing
vi.mock("@/components/activity_metrics", () => ({
  ActivityMetrics: ({ modelMetrics }: { modelMetrics?: { __source?: string } }) => (
    <div>{`activity-source:${modelMetrics?.__source ?? "none"}`}</div>
  ),
  processActivityData: (_data: unknown, key: string) => ({ __source: key }),
}));

vi.mock("@/components/view_user_spend", () => ({
  default: () => <div>View User Spend</div>,
}));

vi.mock("@/components/UsagePage/components/EntityUsage/TopKeyView", () => ({
  default: ({ topKeysLimit, setTopKeysLimit }: { topKeysLimit: number; setTopKeysLimit: (limit: number) => void }) => (
    <div>
      <span>Top Keys</span>
      <span>{`top-keys-limit:${topKeysLimit}`}</span>
      <button onClick={() => setTopKeysLimit(25)}>set-top-keys-limit</button>
    </div>
  ),
}));

vi.mock("./EntityUsage/EntityUsage", () => ({
  default: ({ entityType, entityList }: { entityType: string; entityList: unknown }) => (
    <div data-testid="entity-usage" data-entity-type={entityType} data-entity-list={JSON.stringify(entityList ?? null)}>
      Entity Usage
    </div>
  ),
  EntityList: [],
}));

vi.mock("./EntityUsage/SpendByProvider", () => ({
  default: () => <div>Spend By Provider</div>,
}));

vi.mock("./EndpointUsage/EndpointUsage", () => ({
  default: () => <div>Endpoint Usage</div>,
}));

vi.mock("./UsageViewSelect/UsageViewSelect", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./UsageViewSelect/UsageViewSelect")>();
  const React = await import("react");
  const UsageViewSelect = ({ value, onChange, canViewTagUsage = false }: any) => {
    const tagOption = canViewTagUsage ? React.createElement("option", { value: "tag" }, "Tag Usage") : null;
    return React.createElement(
      "select",
      {
        value,
        onChange: (e: any) => onChange?.(e.target.value),
        role: "combobox",
        "data-testid": "usage-view-select",
        "data-value": value,
      },
      React.createElement("option", { value: "global" }, "Global Usage"),
      React.createElement("option", { value: "team" }, "Team Usage"),
      React.createElement("option", { value: "organization" }, "Organization Usage"),
      React.createElement("option", { value: "customer" }, "Customer Usage"),
      tagOption,
      React.createElement("option", { value: "agent" }, "Agent Usage"),
      React.createElement("option", { value: "user" }, "User Usage"),
      React.createElement("option", { value: "user-agent-activity" }, "User Agent Activity"),
    );
  };
  UsageViewSelect.displayName = "UsageViewSelect";
  return { ...actual, UsageViewSelect };
});

vi.mock("@/components/shared/advanced_date_picker", async () => {
  const React = await import("react");
  // The button is how a test drives a range change; the real picker's own UI is
  // not what any test here is asserting on.
  const AdvancedDatePicker = ({ onValueChange }: { onValueChange?: (value: unknown) => void }) =>
    React.createElement(
      "div",
      { "data-testid": "advanced-date-picker" },
      "Date Picker",
      React.createElement(
        "button",
        {
          "data-testid": "pick-a-different-range",
          onClick: () => onValueChange?.({ from: new Date(2024, 0, 1), to: new Date(2024, 0, 8, 23, 59, 59, 999) }),
        },
        "pick",
      ),
    );
  AdvancedDatePicker.displayName = "AdvancedDatePicker";
  return { default: AdvancedDatePicker };
});

vi.mock("@/components/user_agent_activity", () => ({
  default: () => <div>User Agent Activity</div>,
}));

vi.mock("@/components/cloudzero_export_modal", () => ({
  default: () => <div>CloudZero Export Modal</div>,
}));

vi.mock("@/components/EntityUsageExport", () => ({
  default: () => <div>Entity Usage Export Modal</div>,
}));

vi.mock("./UsageAIChatPanel", () => ({
  default: () => <div data-testid="usage-ai-chat-panel">Usage AI Chat Panel</div>,
}));

vi.mock("@/app/(dashboard)/hooks/customers/useCustomers", () => ({
  useCustomers: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/agents/useAgents", () => ({
  useAgents: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useIsOrgAdmin", () => ({
  __esModule: true,
  default: vi.fn(() => false),
}));

vi.mock("@/app/(dashboard)/hooks/users/useCurrentUser", () => ({
  useCurrentUser: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/users/useUsers", () => ({
  useInfiniteUsers: vi.fn(),
  useUserLookup: vi.fn(() => ({ data: null })),
}));

describe("UsagePage", () => {
  const mockUserDailyActivityAggregatedCall = vi.mocked(networking.userDailyActivityAggregatedCall);
  const mockUserDailyActivityCall = vi.mocked(networking.userDailyActivityCall);
  const mockTagListCall = vi.mocked(networking.tagListCall);
  const mockGatewayDailyActivityCall = vi.mocked(networking.gatewayDailyActivityCall);
  const mockUseCustomers = vi.mocked(useCustomers);
  const mockUseAgents = vi.mocked(useAgents);
  const mockUseAuthorized = vi.mocked(useAuthorized);
  const mockUseCurrentUser = vi.mocked(useCurrentUser);
  const mockUseInfiniteUsers = vi.mocked(useInfiniteUsers);

  const mockSpendData = {
    results: [
      {
        date: "2025-01-01",
        metrics: {
          spend: 125.75,
          api_requests: 1500,
          successful_requests: 1450,
          failed_requests: 50,
          total_tokens: 75000,
          prompt_tokens: 45000,
          completion_tokens: 30000,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
        breakdown: {
          models: {
            "gpt-4": {
              metrics: {
                spend: 75.5,
                api_requests: 800,
                successful_requests: 780,
                failed_requests: 20,
                total_tokens: 40000,
                prompt_tokens: 24000,
                completion_tokens: 16000,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
              },
              metadata: {},
              api_key_breakdown: {},
            },
          },
          model_groups: {
            "gpt-4": {
              metrics: {
                spend: 75.5,
                api_requests: 800,
                successful_requests: 780,
                failed_requests: 20,
                total_tokens: 40000,
                prompt_tokens: 24000,
                completion_tokens: 16000,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
              },
              metadata: {},
              api_key_breakdown: {},
            },
          },
          api_keys: {
            "sk-test123": {
              metrics: {
                spend: 125.75,
                api_requests: 1500,
                successful_requests: 1450,
                failed_requests: 50,
                total_tokens: 75000,
                prompt_tokens: 45000,
                completion_tokens: 30000,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
              },
              metadata: {
                key_alias: "Test Key",
                tags: ["production"],
              },
            },
          },
          providers: {
            openai: {
              metrics: {
                spend: 125.75,
                api_requests: 1500,
                successful_requests: 1450,
                failed_requests: 50,
                total_tokens: 75000,
                prompt_tokens: 45000,
                completion_tokens: 30000,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
              },
            },
          },
          mcp_servers: {},
        },
      },
    ],
    metadata: {
      total_spend: 125.75,
      total_api_requests: 1500,
      total_successful_requests: 1450,
      total_failed_requests: 50,
      total_tokens: 75000,
    },
  };

  const mockOrganizations: Organization[] = [
    {
      organization_id: "org-123",
      organization_alias: "Acme Org",
      budget_id: "budget-1",
      metadata: {},
      models: [],
      spend: 0,
      model_spend: {},
      created_at: "2025-01-01T00:00:00Z",
      created_by: "user-123",
      updated_at: "2025-01-02T00:00:00Z",
      updated_by: "user-123",
      litellm_budget_table: null,
      teams: null,
      users: null,
      members: null,
    },
  ];

  const mockCustomers = [
    {
      user_id: "customer-123",
      alias: "Test Customer",
      spend: 0,
      blocked: false,
      allowed_model_region: null,
      default_model: null,
      budget_id: null,
      litellm_budget_table: null,
    },
  ];

  const mockAgents = [
    {
      agent_id: "agent-123",
      agent_name: "Test Agent",
    },
  ];

  // The same session the suite runs as, minus the admin role. Named rather than
  // inlined so the test reads as "this session, but not an admin".
  const nonAdminSession = {
    isLoading: false,
    isAuthorized: true,
    token: "mock-token",
    accessToken: "test-token",
    userId: "user-123",
    userEmail: "test@example.com",
    userRole: "Internal User",
    userRoleLabel: "Internal User",
    isViewOnly: false,
    premiumUser: true,
    disabledPersonalKeyCreation: false,
    showSSOBanner: false,
  };
  const adminSession = { ...nonAdminSession, userRole: "Admin", userRoleLabel: "Admin" };

  const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
    new URLSearchParams(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams);
  const tab = (name: string): HTMLElement => screen.getByRole("tab", { name });
  const selectedView = (): string | null => screen.getByTestId("usage-view-select").getAttribute("data-value");

  // Counts deliberately unlike anything in mockSpendData: the gateway tile must be
  // readable as coming from /gateway/daily/activity and from nothing else.
  const mockGatewayActivity = {
    total_successful_requests: 424242,
    total_failed_requests: 909,
    by_date: [{ date: "2025-01-01", successful_requests: 424242, failed_requests: 909 }],
    by_route: [{ category: "llm", route: "/chat/completions", successful_requests: 424242, failed_requests: 909 }],
  };

  const defaultProps = {
    teams: [
      {
        team_id: "team-1",
        team_alias: "Test Team",
        models: [],
        max_budget: null,
        spend: 0,
        tpm_limit: null,
        rpm_limit: null,
        blocked: false,
        metadata: {},
        budget_duration: null,
        organization_id: "org-123",
        created_at: "2025-01-01T00:00:00Z",
        keys: [],
        members_with_roles: [],
      },
    ],
    organizations: [],
  };

  beforeEach(() => {
    mockUseAuthorized.mockReturnValue(adminSession);
    mockUseCurrentUser.mockReturnValue({
      data: {
        user_id: "user-123",
        max_budget: null,
      },
      isLoading: false,
      error: null,
    } as any);
    mockUserDailyActivityAggregatedCall.mockClear();
    mockUserDailyActivityCall.mockClear();
    mockTagListCall.mockClear();
    mockGatewayDailyActivityCall.mockClear();
    mockUserDailyActivityAggregatedCall.mockResolvedValue(mockSpendData);
    mockGatewayDailyActivityCall.mockResolvedValue(mockGatewayActivity);
    mockUseInfiniteUsers.mockReturnValue({
      data: {
        pages: [
          {
            users: [
              { user_id: "user-001", user_alias: "Alice", user_email: "alice@example.com" },
              { user_id: "user-002", user_alias: null, user_email: "bob@example.com" },
              { user_id: "user-003", user_alias: null, user_email: null },
            ],
            page: 1,
            total_pages: 1,
            total_count: 3,
          },
        ],
        pageParams: [1],
      },
      fetchNextPage: vi.fn(),
      hasNextPage: false,
      isFetchingNextPage: false,
      isLoading: false,
    } as any);
    mockTagListCall.mockResolvedValue({});
    mockUseCustomers.mockReturnValue({
      data: [],
      isLoading: false,
      error: null,
    } as any);
    mockUseAgents.mockReturnValue({
      data: { agents: [] },
      isLoading: false,
      error: null,
    } as any);
  });

  it("should render and fetch usage data on mount", async () => {
    renderWithProviders(<UsagePage {...defaultProps} />);

    // Wait for data to be fetched
    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    // Check that key metrics are displayed
    const totalRequestElements = screen.getAllByText("Total Requests");
    expect(totalRequestElements.length).toBeGreaterThan(0);
    const successfulRequestLabelElements = screen.getAllByText("Successful Requests");
    expect(successfulRequestLabelElements.length).toBeGreaterThan(0);
    await waitFor(() => {
      expect(screen.getAllByText("424,242").length).toBeGreaterThan(0);
    });
    expect(screen.getAllByText("909").length).toBeGreaterThan(0);
    expect(screen.getByText("425,151")).toBeInTheDocument();
    expect(screen.queryByText("1,500")).not.toBeInTheDocument();
    expect(screen.queryByText("1,450")).not.toBeInTheDocument();
  });

  it("should stop showing the previous range's totals while a new range is in flight", async () => {
    // The request tiles read the gateway counts and fall through to the
    // spend-derived ones. Withholding a superseded gateway result is only worth
    // something if the fallback is withheld too, otherwise the tile keeps
    // showing the previous range's number by the other route.
    let releaseSecondFetch: () => void = () => {};
    mockUserDailyActivityAggregatedCall.mockReset();
    mockUserDailyActivityAggregatedCall.mockResolvedValueOnce(mockSpendData).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          releaseSecondFetch = () => resolve(mockSpendData);
        }),
    );

    renderWithProviders(<UsagePage {...defaultProps} />);
    await waitFor(() => {
      expect(screen.getAllByText("75,000").length).toBeGreaterThan(0);
    });

    await act(async () => {
      fireEvent.click(screen.getByTestId("pick-a-different-range"));
    });

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalledTimes(2);
    });
    expect(screen.queryByText("75,000")).not.toBeInTheDocument();

    await act(async () => {
      releaseSecondFetch();
    });
    await waitFor(() => {
      expect(screen.getAllByText("75,000").length).toBeGreaterThan(0);
    });
  });

  it("should fall back to the spend-derived count when the gateway endpoint is unavailable", async () => {
    mockGatewayDailyActivityCall.mockRejectedValue(new Error("gateway activity unavailable"));

    renderWithProviders(<UsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockGatewayDailyActivityCall).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(screen.getAllByText("1,450").length).toBeGreaterThan(0);
    });
    expect(screen.getByText("1,500")).toBeInTheDocument();
    expect(screen.queryByText("424,242")).not.toBeInTheDocument();
    expect(screen.queryByText("909")).not.toBeInTheDocument();
    expect(screen.queryByText("425,151")).not.toBeInTheDocument();
    expect(screen.queryByTestId("gateway-requests-by-endpoint")).not.toBeInTheDocument();
  });

  it("should not request deployment-wide gateway counts for a non-admin", async () => {
    mockUseAuthorized.mockReturnValue(nonAdminSession);

    renderWithProviders(<UsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });
    expect(mockGatewayDailyActivityCall).not.toHaveBeenCalled();
    expect(screen.getByText("1,500")).toBeInTheDocument();
    expect(screen.queryByText("424,242")).not.toBeInTheDocument();
    expect(screen.queryByTestId("gateway-requests-by-endpoint")).not.toBeInTheDocument();
  });

  it("should display usage metrics and charts", async () => {
    renderWithProviders(<UsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    // Check for usage metrics cards
    const totalRequestElements = screen.getAllByText("Total Requests");
    expect(totalRequestElements.length).toBeGreaterThan(0);
    const successfulRequestElements = screen.getAllByText("Successful Requests");
    expect(successfulRequestElements.length).toBeGreaterThan(0);
    const failedRequestElements = screen.getAllByText("Failed Requests");
    expect(failedRequestElements.length).toBeGreaterThan(0);
    const totalTokensElements = screen.getAllByText("Total Tokens");
    expect(totalTokensElements.length).toBeGreaterThan(0);

    // Check for chart titles (these are in the Cost tab)
    expect(screen.getByText("Daily Spend")).toBeInTheDocument();
    expect(screen.getByText("Top Virtual Keys")).toBeInTheDocument();
  });

  it("should render the daily spend and top models charts with cyan bars", async () => {
    const { container } = renderWithProviders(<UsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    // The gateway endpoint breakdown is a separate chart with its own palette,
    // so it is excluded rather than allowed to widen the expected fill set.
    const spendBars = () => {
      const gatewayCard = container.querySelector('[data-testid="gateway-requests-by-endpoint"]');
      return Array.from(container.querySelectorAll("path.recharts-rectangle")).filter(
        (rect) => !gatewayCard?.contains(rect),
      );
    };

    await waitFor(() => {
      expect(spendBars()).toHaveLength(2);
    });

    const fills = new Set(spendBars().map((rect) => rect.getAttribute("fill")));
    expect(fills).toEqual(new Set(["var(--color-cyan-500, #06b6d4)"]));

    expect(screen.getAllByText("2025-01-01").length).toBeGreaterThan(0);
    expect(screen.getAllByText("gpt-4").length).toBeGreaterThan(0);
  });

  it("should switch between usage views correctly", async () => {
    renderWithProviders(<UsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    // Default view should show Global Usage (for admin)
    expect(screen.getByText("Daily Spend")).toBeInTheDocument();

    // Switch to Team Usage view
    const usageSelect = screen.getByTestId("usage-view-select");
    act(() => {
      fireEvent.change(usageSelect, { target: { value: "team" } });
    });

    // Should render EntityUsage component
    await waitFor(() => {
      const entityUsageElements = screen.getAllByText("Entity Usage");
      expect(entityUsageElements.length).toBeGreaterThan(0);
    });

    // Switch to Tag Usage view (admin only)
    act(() => {
      fireEvent.change(usageSelect, { target: { value: "tag" } });
    });

    // Should still render EntityUsage component for tags
    await waitFor(() => {
      const entityUsageElements = screen.getAllByText("Entity Usage");
      expect(entityUsageElements.length).toBeGreaterThan(0);
    });
  });

  it("should withhold the tag list until it resolves so no empty state is shown while loading", async () => {
    let resolveTagList: (tags: TagListResponse) => void = () => {};
    mockTagListCall.mockReturnValue(
      new Promise((resolve) => {
        resolveTagList = resolve;
      }) as ReturnType<typeof networking.tagListCall>,
    );

    renderWithProviders(<UsagePage {...defaultProps} />);

    act(() => {
      fireEvent.change(screen.getByTestId("usage-view-select"), { target: { value: "tag" } });
    });

    const entityUsage = await screen.findByTestId("entity-usage");
    expect(entityUsage).toHaveAttribute("data-entity-list", "null");

    await act(async () => {
      resolveTagList({});
    });

    expect(screen.getByTestId("entity-usage")).toHaveAttribute("data-entity-list", "[]");
  });

  it("should drop the previous range's tags as soon as the range changes", async () => {
    mockTagListCall.mockResolvedValue({ "old-range-tag": { name: "old-range-tag" } } as never);

    renderWithProviders(<UsagePage {...defaultProps} />);

    act(() => {
      fireEvent.change(screen.getByTestId("usage-view-select"), { target: { value: "tag" } });
    });

    await waitFor(() => {
      expect(screen.getByTestId("entity-usage")).toHaveAttribute(
        "data-entity-list",
        JSON.stringify([{ label: "old-range-tag", value: "old-range-tag" }]),
      );
    });

    let resolveNewRange: (tags: TagListResponse) => void = () => {};
    mockTagListCall.mockReturnValue(
      new Promise((resolve) => {
        resolveNewRange = resolve;
      }) as ReturnType<typeof networking.tagListCall>,
    );

    act(() => {
      fireEvent.click(screen.getByTestId("pick-a-different-range"));
    });

    expect(screen.getByTestId("entity-usage")).toHaveAttribute("data-entity-list", "null");

    await act(async () => {
      resolveNewRange({});
    });

    expect(screen.getByTestId("entity-usage")).toHaveAttribute("data-entity-list", "[]");
  });

  it("should show tag usage selector option for internal users", async () => {
    mockUseAuthorized.mockReturnValue({ ...nonAdminSession, userRole: "internal_user" });

    renderWithProviders(<UsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    expect(screen.getByRole("option", { name: "Tag Usage" })).toBeInTheDocument();
  });

  it("should show organization usage banner and view for admins", async () => {
    renderWithProviders(<UsagePage {...defaultProps} organizations={mockOrganizations} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    const usageSelect = screen.getByTestId("usage-view-select");
    act(() => {
      fireEvent.change(usageSelect, { target: { value: "organization" } });
    });

    await waitFor(() => {
      const entityUsageElements = screen.getAllByText("Entity Usage");
      expect(entityUsageElements.length).toBeGreaterThan(0);
    });
  });

  // Org-admin membership comes from the server, so it can be revoked while the
  // page is open. The Organization Usage option and its panel both disappear,
  // and without a fallback the selector keeps a value it no longer offers,
  // leaving the user on a blank trigger over a blank panel with nothing to
  // click. An internal user is used because that is the session role an org
  // admin actually carries.
  it("should leave the organization view when org-admin membership is revoked mid-session", async () => {
    const mockUseIsOrgAdmin = vi.mocked(useIsOrgAdmin);
    mockUseIsOrgAdmin.mockReturnValue(true);
    mockUseAuthorized.mockReturnValue({
      isLoading: false,
      isAuthorized: true,
      token: "mock-token",
      accessToken: "test-token",
      userId: "user-123",
      userEmail: "test@example.com",
      userRole: "Internal User",
      premiumUser: true,
      disabledPersonalKeyCreation: false,
      showSSOBanner: false,
    } as any);

    const { rerender } = renderWithProviders(<UsagePage {...defaultProps} organizations={mockOrganizations} />);

    const usageSelect = screen.getByTestId("usage-view-select");
    act(() => {
      fireEvent.change(usageSelect, { target: { value: "organization" } });
    });
    await waitFor(() => {
      expect(screen.getAllByText("Entity Usage").length).toBeGreaterThan(0);
    });
    expect((usageSelect as HTMLSelectElement).value).toBe("organization");

    mockUseIsOrgAdmin.mockReturnValue(false);
    act(() => {
      rerender(<UsagePage {...defaultProps} organizations={mockOrganizations} />);
    });

    await waitFor(() => {
      expect((screen.getByTestId("usage-view-select") as HTMLSelectElement).value).toBe("global");
    });
  });

  it("should show customer usage view for admins", async () => {
    mockUseCustomers.mockReturnValue({
      data: mockCustomers,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<UsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    const usageSelect = screen.getByTestId("usage-view-select");
    act(() => {
      fireEvent.change(usageSelect, { target: { value: "customer" } });
    });

    await waitFor(() => {
      const entityUsageElements = screen.getAllByText("Entity Usage");
      expect(entityUsageElements.length).toBeGreaterThan(0);
    });
  });

  it("should withhold the customer list while it is still loading", async () => {
    mockUseCustomers.mockReturnValue({ data: undefined, isLoading: true, error: null } as any);

    renderWithProviders(<UsagePage {...defaultProps} />);

    act(() => {
      fireEvent.change(screen.getByTestId("usage-view-select"), { target: { value: "customer" } });
    });

    const entityUsage = await screen.findByTestId("entity-usage");
    expect(entityUsage).toHaveAttribute("data-entity-list", "null");
  });

  it("should show agent usage view for admins", async () => {
    mockUseAgents.mockReturnValue({
      data: { agents: mockAgents },
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<UsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    const usageSelect = screen.getByTestId("usage-view-select");
    act(() => {
      fireEvent.change(usageSelect, { target: { value: "agent" } });
    });

    await waitFor(() => {
      const entityUsageElements = screen.getAllByText("Entity Usage");
      expect(entityUsageElements.length).toBeGreaterThan(0);
    });
  });

  it.each(["organization", "agent"])("should not render the %s usage view for an internal user", async (usageView) => {
    mockUseAuthorized.mockReturnValue(nonAdminSession);

    renderWithProviders(<UsagePage {...defaultProps} organizations={mockOrganizations} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    const usageSelect = screen.getByTestId("usage-view-select");
    act(() => {
      fireEvent.change(usageSelect, { target: { value: "team" } });
    });
    expect(screen.getAllByText("Entity Usage").length).toBeGreaterThan(0);

    act(() => {
      fireEvent.change(usageSelect, { target: { value: usageView } });
    });
    expect(screen.queryByText("Entity Usage")).not.toBeInTheDocument();
  });

  describe("admin user selector", () => {
    // Anchored on the field's own label, so it does not depend on which library draws the control.
    const userSelectCombobox = (): HTMLElement => {
      let node: HTMLElement | null = screen.getByText("Filter by user");
      while (node && !node.querySelector('[role="combobox"]')) {
        node = node.parentElement;
      }
      const combobox = node?.querySelector('[role="combobox"]') ?? null;
      expect(combobox).not.toBeNull();
      return combobox as HTMLElement;
    };

    const openUserSelect = async () => {
      await userEvent.setup().click(userSelectCombobox());
    };

    // One library paints the prompt as its own text node and the other leaves it on the input's
    // placeholder attribute, so either one means the user is being told what to type.
    const promptsWith = (text: string) =>
      screen.queryAllByText(text).length + screen.queryAllByPlaceholderText(text).length > 0;

    it("should render user selector for admin users in global view", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      expect(userSelectCombobox()).toBeInTheDocument();
      expect(promptsWith("Search users by email…")).toBe(true);
    });

    it("should format user options with alias when available", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      await openUserSelect();

      // User with alias should show "alias (id)"
      expect(screen.getByText("Alice (user-001)")).toBeInTheDocument();
      // User without alias but with email should show "email (id)"
      expect(screen.getByText("bob@example.com (user-002)")).toBeInTheDocument();
      // User with neither alias nor email should show just the id
      expect(screen.getByText("user-003")).toBeInTheDocument();
    });

    it("should call useInfiniteUsers with debounced search", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      // useInfiniteUsers should be called with default page size
      expect(mockUseInfiniteUsers).toHaveBeenCalledWith(50, undefined);
    });

    it("should deduplicate users across pages", async () => {
      mockUseInfiniteUsers.mockReturnValue({
        data: {
          pages: [
            {
              users: [{ user_id: "user-dup", user_alias: "DupUser", user_email: null }],
              page: 1,
              total_pages: 2,
              total_count: 2,
            },
            {
              users: [
                { user_id: "user-dup", user_alias: "DupUser", user_email: null },
                { user_id: "user-unique", user_alias: "UniqueUser", user_email: null },
              ],
              page: 2,
              total_pages: 2,
              total_count: 2,
            },
          ],
          pageParams: [1, 2],
        },
        fetchNextPage: vi.fn(),
        hasNextPage: false,
        isFetchingNextPage: false,
        isLoading: false,
      } as any);

      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      await openUserSelect();

      // Duplicate user should appear only once
      const dupElements = screen.getAllByText("DupUser (user-dup)");
      expect(dupElements).toHaveLength(1);
      // Unique user should also appear
      expect(screen.getByText("UniqueUser (user-unique)")).toBeInTheDocument();
    });

    it("should pass selected userId to aggregated call", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      // Initially called with null (global view for admin)
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalledWith(
        "test-token",
        expect.any(Date),
        expect.any(Date),
        null,
      );
    });

    it("should filter by the user named in ?user=", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />, { searchParams: "?user=user-002" });

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalledWith(
          "test-token",
          expect.any(Date),
          expect.any(Date),
          "user-002",
        );
      });
      expect(mockUserDailyActivityAggregatedCall).not.toHaveBeenCalledWith(
        "test-token",
        expect.any(Date),
        expect.any(Date),
        null,
      );
    });

    it("should write the picked user to ?user=", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<UsagePage {...defaultProps} />, { onUrlUpdate });

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      await openUserSelect();
      await userEvent.setup().click(screen.getByText("Alice (user-001)"));

      await waitFor(() => {
        expect(lastUrl(onUrlUpdate).get("user")).toBe("user-001");
      });
      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalledWith(
          "test-token",
          expect.any(Date),
          expect.any(Date),
          "user-001",
        );
      });
    });
  });

  describe("user usage view", () => {
    it("should hand EntityUsage no user list so its own filter can search every user", async () => {
      mockUseInfiniteUsers.mockReturnValue({
        data: {
          pages: [
            {
              users: Array.from({ length: 50 }, (_, index) => ({
                user_id: `user-${index}`,
                user_alias: null,
                user_email: `user${index}@example.com`,
              })),
              page: 1,
              total_pages: 4,
              total_count: 200,
            },
          ],
          pageParams: [1],
        },
        fetchNextPage: vi.fn(),
        hasNextPage: true,
        isFetchingNextPage: false,
        isLoading: false,
      } as unknown as ReturnType<typeof useInfiniteUsers>);

      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      act(() => {
        fireEvent.change(screen.getByTestId("usage-view-select"), { target: { value: "user" } });
      });

      const entityUsage = await screen.findByTestId("entity-usage");
      expect(entityUsage).toHaveAttribute("data-entity-type", "user");
      expect(entityUsage).toHaveAttribute("data-entity-list", "null");
    });
  });

  describe("non-admin user behavior", () => {
    it("should not render user selector for non-admin users", async () => {
      mockUseAuthorized.mockReturnValue({ ...nonAdminSession, premiumUser: false });

      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      // The admin case above proves this label is rendered when the selector exists, so its
      // absence here is a live assertion rather than a query that can never match.
      expect(screen.queryByText("Filter by user")).not.toBeInTheDocument();
    });

    it("should always pass own userId for non-admin users", async () => {
      mockUseAuthorized.mockReturnValue({ ...nonAdminSession, premiumUser: false });

      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalledWith(
          "test-token",
          expect.any(Date),
          expect.any(Date),
          "user-123",
        );
      });
    });
  });

  describe("aggregated endpoint fallback", () => {
    it("should fall back to paginated calls when aggregated endpoint fails", async () => {
      mockUserDailyActivityAggregatedCall.mockRejectedValue(new Error("Aggregated endpoint not available"));
      mockUserDailyActivityCall.mockResolvedValue({
        ...mockSpendData,
        metadata: {
          ...mockSpendData.metadata,
          total_pages: 1,
          page: 1,
        },
      });

      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
        expect(mockUserDailyActivityCall).toHaveBeenCalled();
      });

      // Should still render the data from the paginated fallback, which lands a render after the call
      expect(await screen.findByText("75,000")).toBeInTheDocument();
    });

    it("should stop showing the previous range's paginated pages while a new range is in flight", async () => {
      // Same rule as the aggregate, one fallback further down. The flag that
      // decides whether these pages are read belongs to the range the failure
      // happened on, or the previous range's pages reach the tile through it.
      let releaseSecondAggregated: () => void = () => {};
      mockUserDailyActivityAggregatedCall.mockReset();
      mockUserDailyActivityAggregatedCall
        .mockRejectedValueOnce(new Error("Aggregated endpoint not available"))
        .mockImplementationOnce(
          () =>
            new Promise((_resolve, reject) => {
              releaseSecondAggregated = () => reject(new Error("Aggregated endpoint not available"));
            }),
        );
      mockUserDailyActivityCall.mockResolvedValue({
        ...mockSpendData,
        metadata: { ...mockSpendData.metadata, total_pages: 1, page: 1 },
      });

      renderWithProviders(<UsagePage {...defaultProps} />);
      await waitFor(() => {
        expect(screen.getAllByText("75,000").length).toBeGreaterThan(0);
      });

      await act(async () => {
        fireEvent.click(screen.getByTestId("pick-a-different-range"));
      });

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalledTimes(2);
      });
      expect(screen.queryByText("75,000")).not.toBeInTheDocument();

      await act(async () => {
        releaseSecondAggregated();
      });
      await waitFor(() => {
        expect(screen.getAllByText("75,000").length).toBeGreaterThan(0);
      });
    });

    it("should aggregate multiple pages when paginated endpoint has more than 1 page", async () => {
      mockUserDailyActivityAggregatedCall.mockRejectedValue(new Error("Not available"));

      const page1Data = {
        results: [mockSpendData.results[0]],
        metadata: {
          total_spend: 60,
          total_api_requests: 700,
          total_successful_requests: 680,
          total_failed_requests: 20,
          total_tokens: 35000,
          total_pages: 2,
          page: 1,
        },
      };

      const page2Data = {
        results: [
          {
            ...mockSpendData.results[0],
            date: "2025-01-02",
          },
        ],
        metadata: {
          total_spend: 65.75,
          total_api_requests: 800,
          total_successful_requests: 770,
          total_failed_requests: 30,
          total_tokens: 40000,
          total_pages: 2,
          page: 2,
        },
      };

      mockUserDailyActivityCall.mockResolvedValueOnce(page1Data).mockResolvedValueOnce(page2Data);

      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        // Both pages should have been fetched
        expect(mockUserDailyActivityCall).toHaveBeenCalledTimes(2);
      });

      // Verify first page call
      expect(mockUserDailyActivityCall).toHaveBeenCalledWith("test-token", expect.any(Date), expect.any(Date), 1, null);

      // Verify second page call
      expect(mockUserDailyActivityCall).toHaveBeenCalledWith("test-token", expect.any(Date), expect.any(Date), 2, null);
    });
  });

  describe("MCP Server Activity tab", () => {
    it("should render MCP Server Activity tab", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      // The tab list should contain MCP Server Activity
      expect(screen.getByText("MCP Server Activity")).toBeInTheDocument();
    });
  });

  describe("User Agent Activity view", () => {
    it("should render User Agent Activity component when view is selected", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      const usageSelect = screen.getByTestId("usage-view-select");
      act(() => {
        fireEvent.change(usageSelect, { target: { value: "user-agent-activity" } });
      });

      await waitFor(() => {
        // "User Agent Activity" appears both in the select option and in the rendered component
        const elements = screen.getAllByText("User Agent Activity");
        expect(elements.length).toBeGreaterThanOrEqual(2);
      });
    });
  });

  describe("Export Data button", () => {
    it("should render Export Data button in global view for admin", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      expect(screen.getByText("Export Data")).toBeInTheDocument();
    });
  });

  describe("Ask AI button", () => {
    it("should render Ask AI button in global view", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      expect(screen.getByText("Ask AI")).toBeInTheDocument();
    });

    it("should render AI chat panel component", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      expect(screen.getByTestId("usage-ai-chat-panel")).toBeInTheDocument();
    });
  });

  describe("model view toggle", () => {
    it("should show Public Model Name view by default", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      // Default should be "groups" view showing "Top Public Model Names"
      expect(screen.getByText("Top Public Model Names")).toBeInTheDocument();
      expect(screen.getAllByText("Public Model Name").length).toBeGreaterThan(0);
      expect(screen.getAllByText("Litellm Model Name").length).toBeGreaterThan(0);
    });

    it("should switch to Litellm Model Name view on toggle click", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      // Click the "Litellm Model Name" toggle
      const litellmToggle = screen.getAllByText("Litellm Model Name")[0];
      act(() => {
        fireEvent.click(litellmToggle);
      });

      // Title should change to "Top Litellm Models"
      await waitFor(() => {
        expect(screen.getByText("Top Litellm Models")).toBeInTheDocument();
      });
    });

    it("should switch back to Public Model Name view", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<UsagePage {...defaultProps} />, { onUrlUpdate });

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      // Switch to individual first
      const litellmToggle = screen.getAllByText("Litellm Model Name")[0];
      act(() => {
        fireEvent.click(litellmToggle);
      });

      await waitFor(() => {
        expect(screen.getByText("Top Litellm Models")).toBeInTheDocument();
      });
      await waitFor(() => {
        expect(lastUrl(onUrlUpdate).get("model_view")).toBe("individual");
      });

      // Switch back to groups
      const publicToggle = screen.getAllByText("Public Model Name")[0];
      act(() => {
        fireEvent.click(publicToggle);
      });

      await waitFor(() => {
        expect(screen.getByText("Top Public Model Names")).toBeInTheDocument();
      });
    });

    it("should feed the Model Activity tab from the model_groups breakdown by default", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      expect(screen.getByText("activity-source:model_groups")).toBeInTheDocument();
      expect(screen.queryByText("activity-source:models")).not.toBeInTheDocument();
    });

    it("should switch the Model Activity tab to the litellm models breakdown on toggle click", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      act(() => {
        fireEvent.click(screen.getAllByText("Litellm Model Name")[0]);
      });

      await waitFor(() => {
        expect(screen.getByText("activity-source:models")).toBeInTheDocument();
      });
      expect(screen.queryByText("activity-source:model_groups")).not.toBeInTheDocument();
    });
  });

  describe("customer usage banner", () => {
    it("should show and be dismissible in customer view", async () => {
      mockUseCustomers.mockReturnValue({
        data: mockCustomers,
        isLoading: false,
        error: null,
      } as any);

      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      const usageSelect = screen.getByTestId("usage-view-select");
      act(() => {
        fireEvent.change(usageSelect, { target: { value: "customer" } });
      });

      await waitFor(() => {
        const entityUsageElements = screen.getAllByText("Entity Usage");
        expect(entityUsageElements.length).toBeGreaterThan(0);
      });
    });
  });

  describe("agent usage banner", () => {
    it("should show agent usage banner with A2A info", async () => {
      mockUseAgents.mockReturnValue({
        data: { agents: mockAgents },
        isLoading: false,
        error: null,
      } as any);

      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      const usageSelect = screen.getByTestId("usage-view-select");
      act(() => {
        fireEvent.change(usageSelect, { target: { value: "agent" } });
      });

      await waitFor(() => {
        const entityUsageElements = screen.getAllByText("Entity Usage");
        expect(entityUsageElements.length).toBeGreaterThan(0);
      });
    });
  });

  describe("tab navigation in global view", () => {
    it("should render all expected tabs", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      expect(screen.getByText("Cost")).toBeInTheDocument();
      expect(screen.getByText("Model Activity")).toBeInTheDocument();
      expect(screen.getByText("Key Activity")).toBeInTheDocument();
      expect(screen.getByText("MCP Server Activity")).toBeInTheDocument();
      expect(screen.getByText("Endpoint Activity")).toBeInTheDocument();
    });
  });

  describe("URL state", () => {
    const renderKeepingMountWrites = (ui: ReactNode, searchParams: string, onUrlUpdate: OnUrlUpdateFunction) =>
      render(ui, {
        wrapper: ({ children }: { children: ReactNode }) => (
          <NuqsTestingAdapter
            searchParams={searchParams}
            onUrlUpdate={onUrlUpdate}
            hasMemory
            resetUrlUpdateQueueOnMount={false}
          >
            <QueryClientProvider client={testQueryClient}>{children}</QueryClientProvider>
          </NuqsTestingAdapter>
        ),
      });

    it("opens the view named by ?view=", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />, { searchParams: "?view=team" });

      const entityUsage = await screen.findByTestId("entity-usage");
      expect(entityUsage).toHaveAttribute("data-entity-type", "team");
      expect(selectedView()).toBe("team");
      expect(screen.queryByText("Daily Spend")).not.toBeInTheDocument();
    });

    it("writes the picked view to the URL and drops the previous view's entity filter", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<UsagePage {...defaultProps} />, { searchParams: "?view=team&filter=team-1", onUrlUpdate });
      await screen.findByTestId("entity-usage");

      act(() => {
        fireEvent.change(screen.getByTestId("usage-view-select"), { target: { value: "tag" } });
      });

      await waitFor(() => {
        expect(lastUrl(onUrlUpdate).get("view")).toBe("tag");
      });
      expect(lastUrl(onUrlUpdate).has("filter")).toBe(false);
      expect(screen.getByTestId("entity-usage")).toHaveAttribute("data-entity-type", "tag");
    });

    it.each(["organization", "customer", "agent", "user", "user-agent-activity", "my-usage", "nope"])(
      "falls back to global usage when ?view=%s names a view this session cannot open",
      async (view) => {
        mockUseAuthorized.mockReturnValue(nonAdminSession);

        renderWithProviders(<UsagePage {...defaultProps} organizations={mockOrganizations} />, {
          searchParams: `?view=${view}`,
        });

        await waitFor(() => {
          expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalledWith(
            "test-token",
            expect.any(Date),
            expect.any(Date),
            "user-123",
          );
        });
        expect(screen.queryByTestId("entity-usage")).not.toBeInTheDocument();
        expect(screen.queryByText("User Agent Activity", { selector: "div" })).not.toBeInTheDocument();
        expect(screen.getByText("Daily Spend")).toBeInTheDocument();
        expect(selectedView()).toBe("global");
      },
    );

    it("opens ?view=tag for an internal user, who may see tag usage", async () => {
      mockUseAuthorized.mockReturnValue(nonAdminSession);

      renderWithProviders(<UsagePage {...defaultProps} />, { searchParams: "?view=tag" });

      expect(await screen.findByTestId("entity-usage")).toHaveAttribute("data-entity-type", "tag");
      expect(selectedView()).toBe("tag");
    });

    it("keeps ?view=organization for an org admin whose membership resolves after the first render", async () => {
      const mockUseIsOrgAdmin = vi.mocked(useIsOrgAdmin);
      mockUseIsOrgAdmin.mockReturnValue(false);
      mockUseAuthorized.mockReturnValue(nonAdminSession);
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();

      const { rerender } = renderKeepingMountWrites(
        <UsagePage {...defaultProps} organizations={mockOrganizations} />,
        "?view=organization&tab=keys",
        onUrlUpdate,
      );
      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });
      expect(screen.queryByTestId("entity-usage")).not.toBeInTheDocument();

      try {
        mockUseIsOrgAdmin.mockReturnValue(true);
        act(() => {
          rerender(<UsagePage {...defaultProps} organizations={mockOrganizations} />);
        });

        expect(await screen.findByTestId("entity-usage")).toHaveAttribute("data-entity-type", "organization");
        expect(selectedView()).toBe("organization");
        expect(onUrlUpdate).not.toHaveBeenCalled();
      } finally {
        mockUseIsOrgAdmin.mockReturnValue(false);
      }
    });

    it("requests the range named by ?from= and ?to= as whole local days", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />, { searchParams: "?from=2025-03-01&to=2025-03-05" });

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalledWith(
          "test-token",
          new Date(2025, 2, 1),
          new Date(2025, 2, 5, 23, 59, 59, 999),
          null,
        );
      });
    });

    it("writes a picked range to the URL as calendar days in one update", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<UsagePage {...defaultProps} />, { onUrlUpdate });
      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      await act(async () => {
        fireEvent.click(screen.getByTestId("pick-a-different-range"));
      });

      await waitFor(() => {
        expect(onUrlUpdate).toHaveBeenCalledTimes(1);
      });
      expect(lastUrl(onUrlUpdate).get("from")).toBe("2024-01-01");
      expect(lastUrl(onUrlUpdate).get("to")).toBe("2024-01-08");
    });

    it("selects the tab named by ?tab= and writes tab changes back", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<UsagePage {...defaultProps} />, { searchParams: "?tab=keys", onUrlUpdate });
      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });
      expect(tab("Key Activity")).toHaveAttribute("aria-selected", "true");
      expect(tab("Cost")).toHaveAttribute("aria-selected", "false");

      act(() => {
        fireEvent.click(tab("MCP Server Activity"));
      });

      await waitFor(() => {
        expect(lastUrl(onUrlUpdate).get("tab")).toBe("mcp");
      });
      expect(tab("MCP Server Activity")).toHaveAttribute("aria-selected", "true");
    });

    it("falls back to the cost tab and clears ?tab= when it names a tab the global view does not render", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderKeepingMountWrites(<UsagePage {...defaultProps} />, "?tab=agents&top_models=10", onUrlUpdate);
      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      expect(tab("Cost")).toHaveAttribute("aria-selected", "true");
      await waitFor(() => {
        expect(onUrlUpdate).toHaveBeenCalled();
      });
      expect(lastUrl(onUrlUpdate).has("tab")).toBe(false);
      expect(lastUrl(onUrlUpdate).get("top_models")).toBe("10");
    });

    it("drops ?tab=agents from the URL when leaving an entity view for global usage", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<UsagePage {...defaultProps} />, { searchParams: "?view=team&tab=agents", onUrlUpdate });
      await screen.findByTestId("entity-usage");

      act(() => {
        fireEvent.change(screen.getByTestId("usage-view-select"), { target: { value: "global" } });
      });

      await waitFor(() => {
        expect(onUrlUpdate).toHaveBeenCalled();
      });
      expect(lastUrl(onUrlUpdate).has("view")).toBe(false);
      expect(lastUrl(onUrlUpdate).has("tab")).toBe(false);
      expect(tab("Cost")).toHaveAttribute("aria-selected", "true");
    });

    it("leaves ?tab=agents alone for an entity view, whose panel does render that tab", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderKeepingMountWrites(<UsagePage {...defaultProps} />, "?view=team&tab=agents", onUrlUpdate);
      await screen.findByTestId("entity-usage");

      await act(async () => {
        fireEvent.click(screen.getByTestId("pick-a-different-range"));
      });

      await waitFor(() => {
        expect(onUrlUpdate).toHaveBeenCalled();
      });
      expect(onUrlUpdate.mock.calls.map(([update]) => update.searchParams.get("tab"))).toEqual(
        onUrlUpdate.mock.calls.map(() => "agents"),
      );
      expect(lastUrl(onUrlUpdate).get("view")).toBe("team");
      expect(lastUrl(onUrlUpdate).get("from")).toBe("2024-01-01");
    });

    it("shows litellm model names when ?model_view=individual", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />, { searchParams: "?model_view=individual" });
      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      expect(screen.getByText("Top Litellm Models")).toBeInTheDocument();
      expect(screen.getByText("activity-source:models")).toBeInTheDocument();
      expect(screen.queryByText("activity-source:model_groups")).not.toBeInTheDocument();
    });

    it("writes the model view toggle to ?model_view= and clears it on the default", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<UsagePage {...defaultProps} />, { onUrlUpdate });
      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      act(() => {
        fireEvent.click(screen.getAllByText("Litellm Model Name")[0]);
      });
      await waitFor(() => {
        expect(lastUrl(onUrlUpdate).get("model_view")).toBe("individual");
      });

      act(() => {
        fireEvent.click(screen.getAllByText("Public Model Name")[0]);
      });
      await waitFor(() => {
        expect(lastUrl(onUrlUpdate).has("model_view")).toBe(false);
      });
      expect(screen.getByText("Top Public Model Names")).toBeInTheDocument();
    });

    it("reads the top-models limit from ?top_models= and writes a picked limit back", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<UsagePage {...defaultProps} />, { searchParams: "?top_models=25", onUrlUpdate });
      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });
      expect(tab("25")).toHaveAttribute("aria-selected", "true");

      act(() => {
        fireEvent.click(tab("10"));
      });

      await waitFor(() => {
        expect(lastUrl(onUrlUpdate).get("top_models")).toBe("10");
      });
      expect(tab("10")).toHaveAttribute("aria-selected", "true");
    });

    it("falls back to the default top-models limit when ?top_models= is not an offered size", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />, { searchParams: "?top_models=7" });
      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      expect(tab("5")).toHaveAttribute("aria-selected", "true");
    });

    it("hands the top-keys table the limit from ?top_keys= and writes its picks back", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<UsagePage {...defaultProps} />, { searchParams: "?top_keys=10", onUrlUpdate });
      expect(await screen.findByText("top-keys-limit:10")).toBeInTheDocument();

      act(() => {
        fireEvent.click(screen.getByText("set-top-keys-limit"));
      });

      await waitFor(() => {
        expect(lastUrl(onUrlUpdate).get("top_keys")).toBe("25");
      });
      expect(screen.getByText("top-keys-limit:25")).toBeInTheDocument();
    });
  });
});
