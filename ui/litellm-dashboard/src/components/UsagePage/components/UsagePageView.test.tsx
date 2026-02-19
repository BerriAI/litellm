import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useCustomers } from "@/app/(dashboard)/hooks/customers/useCustomers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";
import { useInfiniteUsers } from "@/app/(dashboard)/hooks/users/useUsers";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import type { Organization } from "../../networking";
import * as networking from "../../networking";
import UsagePage from "./UsagePageView";

// Polyfill ResizeObserver for test environment
beforeAll(() => {
  if (typeof window !== "undefined" && !window.ResizeObserver) {
    window.ResizeObserver = class ResizeObserver {
      observe() { }
      unobserve() { }
      disconnect() { }
    } as any;
  }
});

// Mock the networking module
vi.mock("../../networking", () => ({
  userDailyActivityCall: vi.fn(),
  userDailyActivityAggregatedCall: vi.fn(),
  tagListCall: vi.fn(),
}));

// Mock child components to simplify testing
vi.mock("../../activity_metrics", () => ({
  ActivityMetrics: () => <div>Activity Metrics</div>,
  processActivityData: () => ({ data: [], metadata: {} }),
}));

vi.mock("../../view_user_spend", () => ({
  default: () => <div>View User Spend</div>,
}));

vi.mock("./EntityUsage/TopKeyView", () => ({
  default: () => <div>Top Keys</div>,
}));

vi.mock("./EntityUsage/EntityUsage", () => ({
  default: () => <div>Entity Usage</div>,
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
  const UsageViewSelect = ({ value, onChange }: any) => {
    return React.createElement(
      "select",
      {
        value,
        onChange: (e: any) => onChange?.(e.target.value),
        role: "combobox",
        "data-testid": "usage-view-select",
      },
      React.createElement("option", { value: "global" }, "Global Usage"),
      React.createElement("option", { value: "team" }, "Team Usage"),
      React.createElement("option", { value: "organization" }, "Organization Usage"),
      React.createElement("option", { value: "customer" }, "Customer Usage"),
      React.createElement("option", { value: "tag" }, "Tag Usage"),
      React.createElement("option", { value: "agent" }, "Agent Usage"),
      React.createElement("option", { value: "user-agent-activity" }, "User Agent Activity"),
    );
  };
  UsageViewSelect.displayName = "UsageViewSelect";
  return { UsageViewSelect };
});

vi.mock("../../shared/advanced_date_picker", async () => {
  const React = await import("react");
  const AdvancedDatePicker = () => {
    return React.createElement("div", { "data-testid": "advanced-date-picker" }, "Date Picker");
  };
  AdvancedDatePicker.displayName = "AdvancedDatePicker";
  return { default: AdvancedDatePicker };
});

vi.mock("../../user_agent_activity", () => ({
  default: () => <div>User Agent Activity</div>,
}));

vi.mock("../../cloudzero_export_modal", () => ({
  default: () => <div>CloudZero Export Modal</div>,
}));

vi.mock("../../EntityUsageExport", () => ({
  default: () => <div>Entity Usage Export Modal</div>,
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

vi.mock("@/app/(dashboard)/hooks/users/useCurrentUser", () => ({
  useCurrentUser: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/users/useUsers", () => ({
  useInfiniteUsers: vi.fn(),
}));

vi.mock("antd", async (importOriginal) => {
  const React = await import("react");
  const actual = await importOriginal<typeof import("antd")>();

  function Select(props: any) {
    const { value, onChange, options, ...rest } = props;
    return React.createElement(
      "select",
      {
        ...rest,
        value,
        onChange: (e: any) => onChange?.(e.target.value),
        role: "combobox",
      },
      options?.map((opt: any) => React.createElement("option", { key: opt.value, value: opt.value }, opt.label)),
    );
  }
  (Select as any).displayName = "AntdSelect";

  function Alert(props: any) {
    const { message, description, type, closable, onClose, ...rest } = props;
    return React.createElement(
      "div",
      { ...rest, "data-testid": "antd-alert", "data-type": type },
      message && React.createElement("div", null, message),
      description && React.createElement("div", null, description),
      closable && React.createElement("button", { onClick: onClose, "aria-label": "Close" }, "×"),
    );
  }
  (Alert as any).displayName = "AntdAlert";

  function Badge(props: any) {
    const { count, color, children, ...rest } = props;
    return React.createElement(
      "div",
      { ...rest, "data-testid": "antd-badge", "data-color": color },
      count && React.createElement("span", { "data-testid": "antd-badge-count" }, count),
      children,
    );
  }
  (Badge as any).displayName = "AntdBadge";

  function Table({ columns, dataSource, ...rest }: any) {
    return React.createElement(
      "div",
      { ...rest, "data-testid": "antd-table" },
      columns?.map((col: any) =>
        React.createElement("div", { key: col.key, "data-testid": `column-${col.key}` }, col.title),
      ),
      dataSource?.map((row: any) =>
        React.createElement(
          "div",
          { key: row.key, "data-testid": `row-${row.key}` },
          columns?.map((col: any) => {
            const value = col.render ? col.render(row[col.dataIndex], row) : row[col.dataIndex];
            return React.createElement("div", { key: col.key }, value);
          }),
        ),
      ),
    );
  }
  (Table as any).displayName = "Table";

  function Segmented(props: any) {
    const { value, onChange, options, ...rest } = props;
    return React.createElement(
      "div",
      { ...rest, "data-testid": "antd-segmented" },
      options?.map((opt: any) =>
        React.createElement(
          "button",
          {
            key: opt.value,
            onClick: () => onChange?.(opt.value),
            "data-selected": value === opt.value,
          },
          opt.label,
        ),
      ),
    );
  }
  (Segmented as any).displayName = "AntdSegmented";

  function Tooltip(props: any) {
    const { title, children, ...rest } = props;
    return React.createElement("div", { ...rest, "data-testid": "antd-tooltip", title }, children);
  }
  (Tooltip as any).displayName = "AntdTooltip";

  return {
    ...actual,
    Select,
    Alert,
    Badge,
    Table,
    Segmented,
    Tooltip,
  };
});

vi.mock("@ant-design/icons", async () => {
  const React = await import("react");

  function Icon() {
    return React.createElement("span");
  }

  function LoadingOutlined(props: any) {
    return React.createElement("span", { "data-testid": "loading-icon", ...props });
  }

  return {
    GlobalOutlined: Icon,
    BankOutlined: Icon,
    TeamOutlined: Icon,
    ShoppingCartOutlined: Icon,
    TagsOutlined: Icon,
    RobotOutlined: Icon,
    LineChartOutlined: Icon,
    BarChartOutlined: Icon,
    ClockCircleOutlined: Icon,
    CalendarOutlined: Icon,
    InfoCircleOutlined: Icon,
    UserOutlined: Icon,
    LoadingOutlined,
  };
});

// Mock Tremor components
vi.mock("@tremor/react", async () => {
  const React = await import("react");
  const actual = await import("@tremor/react");

  function TabGroup({ children }: any) {
    return React.createElement("div", { "data-testid": "tremor-tab-group" }, children);
  }

  function TabList({ children }: any) {
    return React.createElement("div", { "data-testid": "tremor-tab-list" }, children);
  }

  function Tab({ children, ...props }: any) {
    return React.createElement("button", { ...props, "data-testid": "tremor-tab" }, children);
  }

  function TabPanels({ children }: any) {
    return React.createElement("div", { "data-testid": "tremor-tab-panels" }, children);
  }

  function TabPanel({ children }: any) {
    return React.createElement("div", { "data-testid": "tremor-tab-panel" }, children);
  }

  function Card({ children, ...props }: any) {
    return React.createElement("div", { ...props, "data-testid": "tremor-card" }, children);
  }

  function Grid({ children, numItems, ...props }: any) {
    return React.createElement("div", { ...props, "data-testid": "tremor-grid" }, children);
  }

  function Col({ children, numColSpan, ...props }: any) {
    return React.createElement("div", { ...props, "data-testid": "tremor-col" }, children);
  }

  function Title({ children, ...props }: any) {
    return React.createElement("h2", { ...props, "data-testid": "tremor-title" }, children);
  }

  function Text({ children, ...props }: any) {
    return React.createElement("p", { ...props, "data-testid": "tremor-text" }, children);
  }

  function BarChart({ data, valueFormatter, yAxisWidth, showLegend, customTooltip, ...props }: any) {
    return React.createElement("div", { ...props, "data-testid": "tremor-bar-chart" }, "Bar Chart");
  }

  function DonutChart({ data, ...props }: any) {
    return React.createElement("div", { ...props, "data-testid": "tremor-donut-chart" }, "Donut Chart");
  }

  function Button({ children, icon, onClick, ...props }: any) {
    return React.createElement(
      "button",
      { ...props, onClick, "data-testid": "tremor-button" },
      icon && React.createElement("span", { "data-testid": "tremor-button-icon" }),
      children,
    );
  }

  return {
    ...actual,
    TabGroup,
    TabList,
    Tab,
    TabPanels,
    TabPanel,
    Card,
    Grid,
    Col,
    Title,
    Text,
    BarChart,
    DonutChart,
    Button,
  };
});

describe("UsagePage", () => {
  const mockUserDailyActivityAggregatedCall = vi.mocked(networking.userDailyActivityAggregatedCall);
  const mockUserDailyActivityCall = vi.mocked(networking.userDailyActivityCall);
  const mockTagListCall = vi.mocked(networking.tagListCall);
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
    mockUserDailyActivityCall.mockClear();
    mockTagListCall.mockClear();
    mockUserDailyActivityAggregatedCall.mockResolvedValue(mockSpendData);
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
    expect(screen.getByText("1,500")).toBeInTheDocument();
    const successfulRequestLabelElements = screen.getAllByText("Successful Requests");
    expect(successfulRequestLabelElements.length).toBeGreaterThan(0);
    // Use getAllByText since this value appears in multiple places (metrics card + table)
    const successfulRequestElements = screen.getAllByText("1,450");
    expect(successfulRequestElements.length).toBeGreaterThan(0);
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
      expect(screen.getByText("Organization usage is a new feature.")).toBeInTheDocument();
      const entityUsageElements = screen.getAllByText("Entity Usage");
      expect(entityUsageElements.length).toBeGreaterThan(0);
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

  describe("admin user selector", () => {
    it("should render user selector for admin users in global view", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      // Admin should see the user selector select element with the placeholder attribute
      const userSelects = screen.getAllByRole("combobox");
      const userSelect = userSelects.find(
        (el) => el.getAttribute("placeholder") === "All Users (Global View)",
      );
      expect(userSelect).toBeDefined();
    });

    it("should format user options with alias when available", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

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
              users: [
                { user_id: "user-dup", user_alias: "DupUser", user_email: null },
              ],
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

      // Non-admin should not see the user selector
      const userSelects = screen.getAllByRole("combobox");
      const userSelect = userSelects.find(
        (el) => el.getAttribute("placeholder") === "All Users (Global View)",
      );
      expect(userSelect).toBeUndefined();
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

      // Should still render the data from the paginated fallback
      expect(screen.getByText("1,500")).toBeInTheDocument();
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

      mockUserDailyActivityCall
        .mockResolvedValueOnce(page1Data)
        .mockResolvedValueOnce(page2Data);

      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        // Both pages should have been fetched
        expect(mockUserDailyActivityCall).toHaveBeenCalledTimes(2);
      });

      // Verify first page call
      expect(mockUserDailyActivityCall).toHaveBeenCalledWith(
        "test-token",
        expect.any(Date),
        expect.any(Date),
        1,
        null,
      );

      // Verify second page call
      expect(mockUserDailyActivityCall).toHaveBeenCalledWith(
        "test-token",
        expect.any(Date),
        expect.any(Date),
        2,
        null,
      );
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

  describe("model view toggle", () => {
    it("should show Public Model Name view by default", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      // Default should be "groups" view showing "Top Public Model Names"
      expect(screen.getByText("Top Public Model Names")).toBeInTheDocument();
      expect(screen.getByText("Public Model Name")).toBeInTheDocument();
      expect(screen.getByText("Litellm Model Name")).toBeInTheDocument();
    });

    it("should switch to Litellm Model Name view on toggle click", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      // Click the "Litellm Model Name" toggle
      const litellmToggle = screen.getByText("Litellm Model Name");
      act(() => {
        fireEvent.click(litellmToggle);
      });

      // Title should change to "Top Litellm Models"
      await waitFor(() => {
        expect(screen.getByText("Top Litellm Models")).toBeInTheDocument();
      });
    });

    it("should switch back to Public Model Name view", async () => {
      renderWithProviders(<UsagePage {...defaultProps} />);

      await waitFor(() => {
        expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
      });

      // Switch to individual first
      const litellmToggle = screen.getByText("Litellm Model Name");
      act(() => {
        fireEvent.click(litellmToggle);
      });

      await waitFor(() => {
        expect(screen.getByText("Top Litellm Models")).toBeInTheDocument();
      });

      // Switch back to groups
      const publicToggle = screen.getByText("Public Model Name");
      act(() => {
        fireEvent.click(publicToggle);
      });

      await waitFor(() => {
        expect(screen.getByText("Top Public Model Names")).toBeInTheDocument();
      });
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
        expect(screen.getByText("Customer usage is a new feature.")).toBeInTheDocument();
      });

      // Click the close button
      const closeButton = screen.getByLabelText("Close");
      act(() => {
        fireEvent.click(closeButton);
      });

      await waitFor(() => {
        expect(screen.queryByText("Customer usage is a new feature.")).not.toBeInTheDocument();
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
        expect(screen.getByText("Agent usage (A2A) is a new feature.")).toBeInTheDocument();
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
});
