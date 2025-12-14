import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useCustomers } from "@/app/(dashboard)/hooks/customers/useCustomers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import type { Organization } from "../../networking";
import * as networking from "../../networking";
import NewUsagePage from "./UsagePageView";

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

vi.mock("antd", async () => {
  const React = await import("react");

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

  return { Select, Alert, Badge };
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
  };
});

describe("NewUsage", () => {
  const mockUserDailyActivityAggregatedCall = vi.mocked(networking.userDailyActivityAggregatedCall);
  const mockTagListCall = vi.mocked(networking.tagListCall);
  const mockUseCustomers = vi.mocked(useCustomers);
  const mockUseAgents = vi.mocked(useAgents);
  const mockUseAuthorized = vi.mocked(useAuthorized);

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
    accessToken: "test-token",
    userRole: "Admin",
    userID: "user-123",
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
    premiumUser: true,
  };

  beforeEach(() => {
    mockUseAuthorized.mockReturnValue({
      token: "mock-token",
      accessToken: defaultProps.accessToken,
      userId: defaultProps.userID,
      userEmail: "test@example.com",
      userRole: defaultProps.userRole,
      premiumUser: defaultProps.premiumUser,
      disabledPersonalKeyCreation: false,
      showSSOBanner: false,
    });
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
    render(<NewUsagePage {...defaultProps} />);

    // Wait for data to be fetched
    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    // Check that key metrics are displayed
    expect(screen.getByText("Total Requests")).toBeInTheDocument();
    expect(screen.getByText("1,500")).toBeInTheDocument();
    expect(screen.getByText("Successful Requests")).toBeInTheDocument();
    // Use getAllByText since this value appears in multiple places (metrics card + table)
    const successfulRequestElements = screen.getAllByText("1,450");
    expect(successfulRequestElements.length).toBeGreaterThan(0);
  });

  it("should display usage metrics and charts", async () => {
    render(<NewUsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    // Check for usage metrics cards
    expect(screen.getByText("Total Requests")).toBeInTheDocument();
    expect(screen.getByText("Successful Requests")).toBeInTheDocument();
    expect(screen.getByText("Failed Requests")).toBeInTheDocument();
    expect(screen.getByText("Total Tokens")).toBeInTheDocument();

    // Check for chart titles
    expect(screen.getByText("Daily Spend")).toBeInTheDocument();
    expect(screen.getByText("Top Virtual Keys")).toBeInTheDocument();
  });

  it("should switch between usage views correctly", async () => {
    render(<NewUsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    // Default view should show Global Usage (for admin)
    expect(screen.getByText("Daily Spend")).toBeInTheDocument();

    // Switch to Team Usage view
    const usageSelect = screen.getByRole("combobox");
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
    render(<NewUsagePage {...defaultProps} organizations={mockOrganizations} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    const usageSelect = screen.getByRole("combobox");
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

    render(<NewUsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    const usageSelect = screen.getByRole("combobox");
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

    render(<NewUsagePage {...defaultProps} />);

    await waitFor(() => {
      expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    const usageSelect = screen.getByRole("combobox");
    act(() => {
      fireEvent.change(usageSelect, { target: { value: "agent" } });
    });

    await waitFor(() => {
      const entityUsageElements = screen.getAllByText("Entity Usage");
      expect(entityUsageElements.length).toBeGreaterThan(0);
    });
  });
});
