import { render, screen } from "@testing-library/react";
import React from "react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { ActivityMetrics, processActivityData, formatKeyLabel } from "./activity_metrics";
import { ModelActivityData, DailyData, KeyMetricWithMetadata } from "./UsagePage/types";
import { Team } from "./key_team_helpers/key_list";

beforeAll(() => {
  if (typeof window !== "undefined" && !window.ResizeObserver) {
    window.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as any;
  }
});

vi.mock("@tremor/react", () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Grid: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Text: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
  Title: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
  AreaChart: () => <div>AreaChart</div>,
  BarChart: () => <div>BarChart</div>,
}));

vi.mock("antd", () => {
  const CollapseComponent = ({ children }: { children: React.ReactNode }) => <div>{children}</div>;
  const PanelComponent = ({ children, header }: { children: React.ReactNode; header: React.ReactNode }) => (
    <div>
      <div>{header}</div>
      <div>{children}</div>
    </div>
  );
  PanelComponent.displayName = "Collapse.Panel";
  CollapseComponent.Panel = PanelComponent;
  return {
    Collapse: CollapseComponent,
  };
});

vi.mock("./UsagePage/utils/value_formatters", () => ({
  valueFormatter: (value: number) => value.toString(),
}));

vi.mock("./common_components/chartUtils", () => ({
  CustomTooltip: () => null,
  CustomLegend: () => <div>Legend</div>,
}));

vi.mock("@/utils/dataUtils", () => ({
  formatNumberWithCommas: (value: number, decimals?: number) => {
    return value.toFixed(decimals || 0);
  },
}));

describe("ActivityMetrics", () => {
  const mockModelMetrics: Record<string, ModelActivityData> = {
    "gpt-4": {
      label: "GPT-4",
      total_requests: 100,
      total_successful_requests: 95,
      total_failed_requests: 5,
      total_tokens: 50000,
      prompt_tokens: 30000,
      completion_tokens: 20000,
      total_spend: 100.5,
      total_cache_read_input_tokens: 1000,
      total_cache_creation_input_tokens: 500,
      top_api_keys: [],
      daily_data: [
        {
          date: "2025-01-01",
          metrics: {
            prompt_tokens: 30000,
            completion_tokens: 20000,
            total_tokens: 50000,
            api_requests: 100,
            spend: 100.5,
            successful_requests: 95,
            failed_requests: 5,
            cache_read_input_tokens: 1000,
            cache_creation_input_tokens: 500,
          },
        },
      ],
    },
  };

  it("should render", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} />);
    expect(screen.getByText("Overall Usage")).toBeInTheDocument();
  });

  it("should display prompt caching metrics when hidePromptCachingMetrics is false", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} hidePromptCachingMetrics={false} />);
    expect(screen.getByText("Prompt Caching Metrics")).toBeInTheDocument();
  });

  it("should hide prompt caching metrics when hidePromptCachingMetrics is true", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} hidePromptCachingMetrics={true} />);
    expect(screen.queryByText("Prompt Caching Metrics")).not.toBeInTheDocument();
  });
});

describe("processActivityData", () => {
  const mockDailyActivity: { results: DailyData[] } = {
    results: [
      {
        date: "2025-01-01",
        metrics: {
          spend: 100.5,
          prompt_tokens: 30000,
          completion_tokens: 20000,
          total_tokens: 50000,
          api_requests: 100,
          successful_requests: 95,
          failed_requests: 5,
          cache_read_input_tokens: 1000,
          cache_creation_input_tokens: 500,
        },
        breakdown: {
          models: {},
          model_groups: {},
          mcp_servers: {},
          providers: {},
          api_keys: {
            key1: {
              metrics: {
                spend: 50.25,
                prompt_tokens: 15000,
                completion_tokens: 10000,
                total_tokens: 25000,
                api_requests: 50,
                successful_requests: 47,
                failed_requests: 3,
                cache_read_input_tokens: 500,
                cache_creation_input_tokens: 250,
              },
              metadata: {
                key_alias: "test-key-1",
                team_id: "team1",
              },
            },
          },
          entities: {},
        },
      },
    ],
  };

  const mockTeams: Team[] = [
    {
      team_id: "team1",
      team_alias: "Test Team 1",
      models: [],
      max_budget: null,
      budget_duration: null,
      tpm_limit: null,
      rpm_limit: null,
      organization_id: "org1",
      created_at: "2025-01-01",
      keys: [],
      members_with_roles: [],
    },
  ];

  it("should process data for models key without teams parameter", () => {
    const result = processActivityData(mockDailyActivity, "models");

    expect(result).toEqual({});
  });

  it("should process data for api_keys key with teams parameter", () => {
    const result = processActivityData(mockDailyActivity, "api_keys", mockTeams);

    expect(result).toHaveProperty("key1");
    expect(result["key1"].label).toBe("test-key-1 (team: Test Team 1)");
    expect(result["key1"].total_requests).toBe(50);
    expect(result["key1"].total_spend).toBe(50.25);
  });

  it("should process data for api_keys key without teams parameter", () => {
    const result = processActivityData(mockDailyActivity, "api_keys");

    expect(result).toHaveProperty("key1");
    expect(result["key1"].label).toBe("test-key-1 (team_id: team1)");
  });
});

describe("formatKeyLabel", () => {
  const mockTeams: Team[] = [
    {
      team_id: "team1",
      team_alias: "Test Team 1",
      models: [],
      max_budget: null,
      budget_duration: null,
      tpm_limit: null,
      rpm_limit: null,
      organization_id: "org1",
      created_at: "2025-01-01",
      keys: [],
      members_with_roles: [],
    },
    {
      team_id: "team2",
      team_alias: "Test Team 2",
      models: [],
      max_budget: null,
      budget_duration: null,
      tpm_limit: null,
      rpm_limit: null,
      organization_id: "org2",
      created_at: "2025-01-01",
      keys: [],
      members_with_roles: [],
    },
  ];

  it("should return key_alias when no team_id is present", () => {
    const modelData: KeyMetricWithMetadata = {
      metrics: {
        spend: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        api_requests: 0,
        successful_requests: 0,
        failed_requests: 0,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      metadata: {
        key_alias: "test-key",
        team_id: null,
      },
    };

    const result = formatKeyLabel(modelData, "test-key", mockTeams);
    expect(result).toBe("test-key");
  });

  it("should return key_alias with team alias when team_id matches", () => {
    const modelData: KeyMetricWithMetadata = {
      metrics: {
        spend: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        api_requests: 0,
        successful_requests: 0,
        failed_requests: 0,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      metadata: {
        key_alias: "test-key",
        team_id: "team1",
      },
    };

    const result = formatKeyLabel(modelData, "test-key", mockTeams);
    expect(result).toBe("test-key (team: Test Team 1)");
  });

  it("should return key_alias with team_id when team is not found", () => {
    const modelData: KeyMetricWithMetadata = {
      metrics: {
        spend: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        api_requests: 0,
        successful_requests: 0,
        failed_requests: 0,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      metadata: {
        key_alias: "test-key",
        team_id: "nonexistent-team",
      },
    };

    const result = formatKeyLabel(modelData, "test-key", mockTeams);
    expect(result).toBe("test-key (team_id: nonexistent-team)");
  });

  it("should use key-hash fallback when key_alias is null", () => {
    const modelData: KeyMetricWithMetadata = {
      metrics: {
        spend: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        api_requests: 0,
        successful_requests: 0,
        failed_requests: 0,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      metadata: {
        key_alias: null,
        team_id: "team1",
      },
    };

    const result = formatKeyLabel(modelData, "actual-key", mockTeams);
    expect(result).toBe("key-hash-actual-key (team: Test Team 1)");
  });
});
