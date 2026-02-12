import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useCustomers } from "@/app/(dashboard)/hooks/customers/useCustomers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";
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
  const mockTagListCall = vi.mocked(networking.tagListCall);
  const mockUseCustomers = vi.mocked(useCustomers);
  const mockUseAgents = vi.mocked(useAgents);
  const mockUseAuthorized = vi.mocked(useAuthorized);
  const mockUseCurrentUser = vi.mocked(useCurrentUser);

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
    mockTagListCall.mockClear();
    mockUserDailyActivityAggregatedCall.mockResolvedValue(mockSpendData);
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
});
