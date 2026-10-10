import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useCustomers } from "@/app/(dashboard)/hooks/customers/useCustomers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useIsOrgAdmin from "@/app/(dashboard)/hooks/useIsOrgAdmin";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";
import { useInfiniteUsers } from "@/app/(dashboard)/hooks/users/useUsers";
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import { processActivityData } from "@/components/activity_metrics";
import type { Organization } from "@/components/networking";
import type { ModelActivityData } from "@/components/UsagePage/types";
import * as networking from "@/components/networking";
import { STACKED_USAGE_PALETTE } from "@/components/shared/charts";
import { OTHER_COLOR } from "./overview/overviewData";
import UsagePage from "./UsagePageView";

// The Overview tab. The Key Activity tab stays mounted and carries its own spend-derived totals,
// so assertions about the overview's request tiles are scoped here rather than to the whole page.
const overview = () => within(screen.getByRole("tabpanel", { name: "Overview" }));

// The Total Requests stat cell: its label, the count, and the "N ok / N failed" line beneath it.
const totalRequestsCell = (): HTMLElement => {
  const cell = overview().getByText("Total Requests").parentElement?.parentElement;
  expect(cell).toBeTruthy();
  return cell as HTMLElement;
};

const modelActivity = (label: string): ModelActivityData => ({
  label,
  total_requests: 1,
  total_successful_requests: 1,
  total_failed_requests: 0,
  total_cache_read_input_tokens: 0,
  total_cache_creation_input_tokens: 0,
  total_tokens: 10,
  prompt_tokens: 5,
  completion_tokens: 5,
  total_spend: 0.01,
  top_models: [],
  daily_data: [],
});

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
  dailyActivityAggregatedCall: vi.fn(),
  dailyActivityKeyPageCall: vi.fn(),
  dailyActivityKeySearchCall: vi.fn(),
  dailyActivityModelTopKeysCall: vi.fn(),
  dailyActivityExportCall: vi.fn(),
  gatewayDailyActivityCall: vi.fn(),
  requestErrorActivityCall: vi.fn(),
  tagListCall: vi.fn(),
}));

// Mock child components to simplify testing
vi.mock("@/components/activity_metrics", () => ({
  ActivityMetrics: ({ modelMetrics }: { modelMetrics?: Record<string, { label: string }> }) => (
    <div>
      {Object.entries(modelMetrics ?? {}).map(([model, metrics]) => (
        <span key={model}>{metrics.label}</span>
      ))}
    </div>
  ),
  processActivityData: vi.fn(),
}));

vi.mock("@/components/view_user_spend", () => ({
  default: () => <div>View User Spend</div>,
}));

vi.mock("@/components/UsagePage/components/EntityUsage/TopKeyView", () => ({
  default: () => <div>Top Keys</div>,
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

vi.mock("./UsageViewSelect/UsageViewSelect", async () => {
  const React = await import("react");
  const UsageViewSelect = ({
    value,
    onChange,
    canViewTagUsage = false,
  }: {
    value: string;
    onChange?: (value: string) => void;
    canViewTagUsage?: boolean;
  }) => {
    const tagOption = canViewTagUsage ? React.createElement("option", { value: "tag" }, "Tag Usage") : null;
    const selectProps = {
      value,
      onChange: (e: React.ChangeEvent<HTMLSelectElement>) => onChange?.(e.target.value),
      role: "combobox",
      "data-testid": "usage-view-select",
    };
    return React.createElement(
      "select",
      selectProps,
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
  return { UsageViewSelect };
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
          onClick: () =>
            onValueChange?.({ from: new Date("2024-01-01T00:00:00Z"), to: new Date("2024-01-08T00:00:00Z") }),
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
  const mockUserDailyActivityAggregatedCall = vi.fn();
  const mockDailyActivityAggregatedCall = vi.mocked(networking.dailyActivityAggregatedCall);
  const mockDailyActivityKeyPageCall = vi.mocked(networking.dailyActivityKeyPageCall);
  const mockTagListCall = vi.mocked(networking.tagListCall);
  const mockGatewayDailyActivityCall = vi.mocked(networking.gatewayDailyActivityCall);
  const mockRequestErrorActivityCall = vi.mocked(networking.requestErrorActivityCall);
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

  // Counts deliberately unlike anything in mockSpendData: the gateway tile must be
  // readable as coming from /gateway/daily/activity and from nothing else.
  const mockRequestErrorActivity = {
    total_successful_requests: 900,
    total_failed_requests: 100,
    by_date: [
      {
        date: "2025-01-01",
        successful_requests: 900,
        failed_requests: 100,
        client_errors: 90,
        server_errors: 10,
        by_status_code: [
          { status_code: 429, failed_requests: 90 },
          { status_code: 500, failed_requests: 10 },
        ],
      },
    ],
    by_status_code: [
      { status_code: 429, failed_requests: 90 },
      { status_code: 500, failed_requests: 10 },
    ],
    by_key: [
      {
        id: "hash-1",
        label: "prod-key",
        api_requests: 400,
        failed_requests: 100,
        top_status_code: 429,
        top_status_code_requests: 90,
      },
    ],
    by_team: [],
    by_user: [],
    by_model: [],
  };

  const mockGatewayActivity = {
    total_successful_requests: 424242,
    total_failed_requests: 909,
    by_date: [{ date: "2025-01-01", successful_requests: 424242, failed_requests: 909 }],
    by_route: [{ category: "llm", route: "/chat/completions", successful_requests: 424242, failed_requests: 909 }],
    by_status_code: [
      { status_code: 429, failed_requests: 3 },
      { status_code: 500, failed_requests: 2 },
    ],
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
    vi.mocked(processActivityData).mockReset();
    vi.mocked(processActivityData).mockImplementation((_data, key) => ({
      [key]: modelActivity(`activity-source:${key}`),
    }));
    mockUseAuthorized.mockReturnValue({
      isLoading: false,
      isAuthorized: true,
      token: "mock-token",
      accessToken: "test-token",
      userId: "user-123",
      userEmail: "test@example.com",
      userRole: "Admin",
      premiumUser: true,
      disabledPersonalKeyCreation: false,
      showSSOBanner: false,
    });
    mockUseCurrentUser.mockReturnValue({
      data: {
        user_id: "user-123",
        max_budget: null,
      },
      isLoading: false,
      error: null,
    } as any);
    mockUserDailyActivityAggregatedCall.mockClear();
    mockDailyActivityAggregatedCall.mockReset();
    mockDailyActivityKeyPageCall.mockReset();
    mockDailyActivityKeyPageCall.mockResolvedValue({
      api_keys: [],
      total_api_keys: 0,
      offset: 0,
      limit: 50,
    });
    mockDailyActivityAggregatedCall.mockImplementation((entity: string, request: unknown) =>
      entity === "user" ? mockUserDailyActivityAggregatedCall(request) : Promise.resolve({ results: [], metadata: {} }),
    );
    mockTagListCall.mockClear();
    mockGatewayDailyActivityCall.mockClear();
    mockUserDailyActivityAggregatedCall.mockResolvedValue(mockSpendData);
    mockGatewayDailyActivityCall.mockResolvedValue(mockGatewayActivity);
    mockRequestErrorActivityCall.mockClear();
    mockRequestErrorActivityCall.mockResolvedValue(mockRequestErrorActivity);
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
    const totalRequestElements = overview().getAllByText("Total Requests");
    expect(totalRequestElements.length).toBeGreaterThan(0);
    await waitFor(() => {
      expect(overview().getAllByText("424,242").length).toBeGreaterThan(0);
    });
    expect(overview().getAllByText("909").length).toBeGreaterThan(0);
    expect(overview().getByText("425,151")).toBeInTheDocument();
    // Successful and failed counts now sit under Total Requests rather than in their own tiles.
    expect(totalRequestsCell()).toHaveTextContent("425,151");
    expect(totalRequestsCell()).toHaveTextContent("424,242 ok");
    expect(totalRequestsCell()).toHaveTextContent("909 failed");
    expect(overview().queryByText("1,500")).not.toBeInTheDocument();
    expect(overview().queryByText("1,450")).not.toBeInTheDocument();
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
    // Total Tokens reads compact (75K); the exact count stays on hover.
    await waitFor(() => {
      expect(overview().getAllByText("75K").length).toBeGreaterThan(0);
    });
    expect(overview().getByTitle("75,000 tokens")).toHaveTextContent("75K");

    await act(async () => {
      fireEvent.click(screen.getByTestId("pick-a-different-range"));
    });

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalledTimes(2);
    });
    expect(overview().queryByText("75K")).not.toBeInTheDocument();
    expect(overview().queryByText("$0.00")).not.toBeInTheDocument();
    expect(overview().getByText("Total Tokens")).toBeInTheDocument();
    expect(await overview().findByText("425,151")).toBeInTheDocument();
    // The Total Tokens stat (label, value and hint) must hold no number while its range is in flight.
    expect(overview().getByText("Total Tokens").parentElement).not.toHaveTextContent(/\d/);
    expect(overview().queryByText("$0.0000")).not.toBeInTheDocument();

    await act(async () => {
      releaseSecondFetch();
    });
    await waitFor(() => {
      expect(overview().getAllByText("75K").length).toBeGreaterThan(0);
    });
    expect(overview().getByText("Total Tokens")).toBeInTheDocument();
    expect(overview().getByText("$0.0838")).toBeInTheDocument();
  });

  it("loads key pages separately from the aggregate and refreshes them when the range changes", async () => {
    renderWithProviders(<UsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });
    fireEvent.click(screen.getByText("Key Activity"));
    await waitFor(() => {
      expect(mockDailyActivityKeyPageCall).toHaveBeenCalledWith("user", expect.any(Object), 0, 50);
    });
    expect(mockUserDailyActivityAggregatedCall.mock.lastCall?.[0]).not.toHaveProperty("apiKeyLimit");
    const pageCallsBeforeRangeChange = mockDailyActivityKeyPageCall.mock.calls.length;
    fireEvent.click(screen.getByTestId("pick-a-different-range"));

    await waitFor(() => {
      expect(mockDailyActivityKeyPageCall.mock.calls.length).toBeGreaterThan(pageCallsBeforeRangeChange);
    });
    expect(mockUserDailyActivityAggregatedCall.mock.lastCall?.[0]).not.toHaveProperty("apiKeyLimit");
  });

  it("should fall back to the spend-derived count when the gateway endpoint is unavailable", async () => {
    mockGatewayDailyActivityCall.mockRejectedValue(new Error("gateway activity unavailable"));

    renderWithProviders(<UsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockGatewayDailyActivityCall).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(overview().getAllByText("1,450").length).toBeGreaterThan(0);
    });
    expect(overview().getByText("1,500")).toBeInTheDocument();
    expect(totalRequestsCell()).toHaveTextContent("1,500");
    expect(totalRequestsCell()).toHaveTextContent("1,450 ok");
    expect(totalRequestsCell()).toHaveTextContent("50 failed");
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
    expect(mockRequestErrorActivityCall).not.toHaveBeenCalled();
    expect(screen.queryByRole("tab", { name: "Errors" })).not.toBeInTheDocument();
    await waitFor(() => {
      expect(totalRequestsCell()).toHaveTextContent("1,500");
    });
    expect(screen.queryByText("424,242")).not.toBeInTheDocument();
    expect(screen.queryByTestId("gateway-requests-by-endpoint")).not.toBeInTheDocument();
  });

  it("keeps failure analytics off the cost overview and on the Errors tab", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockRequestErrorActivityCall).toHaveBeenCalled();
    });
    expect(overview().queryByText(/Client errors \(4xx\)/)).not.toBeInTheDocument();
    expect(overview().queryByRole("button", { name: /Failed Requests/ })).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Errors" }));
    const errorsTab = within(await screen.findByTestId("usage-errors-tab"));
    expect(await errorsTab.findByText("10.0%")).toBeInTheDocument();
    expect(errorsTab.getByRole("list", { name: "Failed requests by status code" })).toHaveTextContent("429");
    const keys = errorsTab.getByRole("table", { name: "Virtual keys ranked by failed requests" });
    expect(within(keys).getAllByRole("row")[1]).toHaveTextContent("prod-key");
  });

  it("should display usage metrics and charts", async () => {
    renderWithProviders(<UsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    // Check for usage metrics cards
    const totalRequestElements = screen.getAllByText("Total Requests");
    expect(totalRequestElements.length).toBeGreaterThan(0);
    // Successful and failed counts now sit under Total Requests rather than in their own tiles.
    await waitFor(() => {
      expect(totalRequestsCell()).toHaveTextContent(/\d+ ok/);
    });
    expect(totalRequestsCell()).toHaveTextContent(/\d+ failed/);
    const totalTokensElements = screen.getAllByText("Total Tokens");
    expect(totalTokensElements.length).toBeGreaterThan(0);

    // Check for chart titles (these are in the Overview tab)
    expect(screen.getByText("Daily usage")).toBeInTheDocument();
    expect(screen.getByText("Daily spend by model (top 8, rest grouped as Other)")).toBeInTheDocument();
    expect(screen.getByText("Top models")).toBeInTheDocument();
    expect(screen.getByText("Top Virtual Keys")).toBeInTheDocument();
  });

  it("should rename the usage chart when the weekly bucket is selected", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    expect(screen.getByText("Daily usage")).toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: "Weekly" }));

    expect(screen.getByText("Weekly usage")).toBeInTheDocument();
    expect(screen.getByText("Weekly spend by model (top 8, rest grouped as Other)")).toBeInTheDocument();
    expect(screen.queryByText("Daily usage")).not.toBeInTheDocument();
  });

  it("should render the top models chart stacked in the shared usage palette", async () => {
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

    // One day stacked as the top model (gpt-4) over the "Other" remainder of that day's spend.
    await waitFor(() => {
      expect(spendBars()).toHaveLength(2);
    });

    const fills = new Set(spendBars().map((rect) => rect.getAttribute("fill")));
    expect(fills).toEqual(new Set([STACKED_USAGE_PALETTE[0], OTHER_COLOR]));

    expect(screen.getAllByText("Jan 1").length).toBeGreaterThan(0);
    expect(screen.getAllByText("gpt-4").length).toBeGreaterThan(0);
  });

  it("should switch between usage views correctly", async () => {
    renderWithProviders(<UsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    // Default view should show Global Usage (for admin)
    expect(screen.getByText("Daily usage")).toBeInTheDocument();

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
    let resolveTagList: (tags: Record<string, unknown>) => void = () => {};
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

    let resolveNewRange: (tags: Record<string, unknown>) => void = () => {};
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
    mockUseAuthorized.mockReturnValue({
      isLoading: false,
      isAuthorized: true,
      token: "mock-token",
      accessToken: "test-token",
      userId: "user-123",
      userEmail: "test@example.com",
      userRole: "internal_user",
      premiumUser: true,
      disabledPersonalKeyCreation: false,
      showSSOBanner: false,
    });

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
    mockDailyActivityKeyPageCall.mockImplementation(() => new Promise(() => {}));
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
    // Anchored on the header dropdown's own test id, so it does not depend on which library draws
    // the control.
    const userSelectCombobox = (): HTMLElement => {
      const combobox = screen.getByTestId("user-dropdown").querySelector('[role="combobox"]');
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
        expect.objectContaining({ accessToken: "test-token", entityIds: null }),
      );
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
      mockUseAuthorized.mockReturnValue({
        isLoading: false,
        isAuthorized: true,
        token: "mock-token",
        accessToken: "test-token",
        userId: "user-123",
        userEmail: "test@example.com",
        userRole: "Internal User",
        premiumUser: false,
        disabledPersonalKeyCreation: false,
        showSSOBanner: false,
      });

      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      // The admin case above proves this test id is rendered when the selector exists, so its
      // absence here is a live assertion rather than a query that can never match.
      expect(screen.queryByTestId("user-dropdown")).not.toBeInTheDocument();
    });

    it("should always pass own userId for non-admin users", async () => {
      mockUseAuthorized.mockReturnValue({
        isLoading: false,
        isAuthorized: true,
        token: "mock-token",
        accessToken: "test-token",
        userId: "user-123",
        userEmail: "test@example.com",
        userRole: "Internal User",
        premiumUser: false,
        disabledPersonalKeyCreation: false,
        showSSOBanner: false,
      });

      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalledWith(
          expect.objectContaining({ accessToken: "test-token", entityIds: ["user-123"] }),
        );
      });
    });
  });

  describe("aggregated endpoint failure", () => {
    it("shows the failure alert instead of retrying other routes when the aggregated call fails", async () => {
      mockUserDailyActivityAggregatedCall.mockRejectedValue(new Error("Aggregated endpoint not available"));

      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });
      expect(mockDailyActivityAggregatedCall.mock.calls.filter((c) => c[0] === "user")).toHaveLength(1);
      expect(await screen.findByText(/Fetching spend data failed/)).toBeInTheDocument();
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

      // Default should be the "groups" view, which feeds Model Activity from model_groups
      expect(screen.getByText("activity-source:model_groups")).toBeInTheDocument();
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

      // Model Activity should switch to the litellm models breakdown
      await waitFor(() => {
        expect(screen.getByText("activity-source:models")).toBeInTheDocument();
      });
    });

    it("should switch back to Public Model Name view", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      // Switch to individual first
      const litellmToggle = screen.getAllByText("Litellm Model Name")[0];
      act(() => {
        fireEvent.click(litellmToggle);
      });

      await waitFor(() => {
        expect(screen.getByText("activity-source:models")).toBeInTheDocument();
      });

      // Switch back to groups
      const publicToggle = screen.getAllByText("Public Model Name")[0];
      act(() => {
        fireEvent.click(publicToggle);
      });

      await waitFor(() => {
        expect(screen.getByText("activity-source:model_groups")).toBeInTheDocument();
      });
      expect(screen.queryByText("activity-source:models")).not.toBeInTheDocument();
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

    it("filters Model Activity by model name and shows an empty state when there are no matches", async () => {
      vi.mocked(processActivityData).mockReturnValue({
        "openai/gpt-4o": modelActivity("GPT-4o"),
        "anthropic/claude-3": modelActivity("Claude 3 Sonnet"),
      });
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });
      fireEvent.click(screen.getByRole("tab", { name: "Model Activity" }));

      const modelActivityTab = screen.getByRole("tabpanel", { name: "Model Activity" });
      const searchInput = within(modelActivityTab).getByRole("textbox", { name: "Search models" });
      fireEvent.change(searchInput, { target: { value: "GPT-4o" } });

      expect(within(modelActivityTab).getByText("GPT-4o")).toBeInTheDocument();
      expect(within(modelActivityTab).queryByText("Claude 3 Sonnet")).not.toBeInTheDocument();

      fireEvent.change(searchInput, { target: { value: "missing model" } });

      expect(
        within(modelActivityTab).getByText('No models match "missing model" in this date range'),
      ).toBeInTheDocument();
    });

    it("does not show the no-match state when model data is empty", async () => {
      vi.mocked(processActivityData).mockReturnValue({});
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });
      fireEvent.click(screen.getByRole("tab", { name: "Model Activity" }));

      const modelActivityTab = screen.getByRole("tabpanel", { name: "Model Activity" });
      const searchInput = within(modelActivityTab).getByRole("textbox", { name: "Search models" });
      fireEvent.change(searchInput, { target: { value: "missing model" } });

      expect(
        within(modelActivityTab).queryByText('No models match "missing model" in this date range'),
      ).not.toBeInTheDocument();
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

      expect(screen.getByText("Overview")).toBeInTheDocument();
      expect(screen.getByText("Model Activity")).toBeInTheDocument();
      expect(screen.getByText("Key Activity")).toBeInTheDocument();
      expect(screen.getByText("MCP Server Activity")).toBeInTheDocument();
      expect(screen.getByText("Endpoint Activity")).toBeInTheDocument();
    });
  });
});
