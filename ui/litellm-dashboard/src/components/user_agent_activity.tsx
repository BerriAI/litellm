import React, { useState, useEffect } from "react";
import { Bot, Users } from "lucide-react";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxClear,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
} from "@/components/ui/combobox";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { BarChart, STACKED_USAGE_PALETTE } from "@/components/shared/charts";
import { ChartSkeleton, Panel, Segmented } from "@/app/(dashboard)/usage/_components/components/overview/Primitives";
import { userAgentSummaryCall, tagDauCall, tagWauCall, tagMauCall, tagDistinctCall } from "./networking";
import PerUserUsage from "./per_user_usage";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";

type ActivityView = "active-users" | "per-user";
type ActivePeriod = "dau" | "wau" | "mau";

const VIEW_OPTIONS = [
  { value: "active-users", label: "Active users" },
  { value: "per-user", label: "Per user" },
] as const satisfies readonly { value: ActivityView; label: string }[];

const PERIOD_OPTIONS = [
  { value: "dau", label: "DAU" },
  { value: "wau", label: "WAU" },
  { value: "mau", label: "MAU" },
] as const satisfies readonly { value: ActivePeriod; label: string }[];

const SUMMARY_SLOTS = 4;

// New interfaces for the updated API response
interface TagActiveUsersResponse {
  tag: string;
  active_users: number;
  date: string;
  period_start?: string;
  period_end?: string;
}

interface ActiveUsersAnalyticsResponse {
  results: TagActiveUsersResponse[];
}

interface TagSummaryMetrics {
  tag: string;
  unique_users: number;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  total_tokens: number;
  total_spend: number;
}

interface TagSummaryResponse {
  results: TagSummaryMetrics[];
}

interface DistinctTagResponse {
  tag: string;
}

// Helper function to extract user agent from tag
const extractUserAgent = (tag: string): string => {
  if (tag.startsWith("User-Agent: ")) {
    return tag.replace("User-Agent: ", "");
  }
  return tag;
};

// Format numbers with K, M abbreviations
const formatAbbreviatedNumber = (value: number, decimalPlaces: number = 0): string => {
  if (value >= 1000000) return (value / 1000000).toFixed(decimalPlaces) + "M";
  if (value >= 1000) return (value / 1000).toFixed(decimalPlaces) + "K";
  return value.toFixed(decimalPlaces);
};

interface ActiveUserChart {
  value: ActivePeriod;
  title: string;
  loading: boolean;
  data: Record<string, unknown>[];
  index: string;
  tags: string[];
}

function ActiveUsersChart({ chart }: { chart: ActiveUserChart }) {
  if (chart.loading) return <ChartSkeleton className="h-72" />;
  if (chart.tags.length === 0) {
    return (
      <div className="flex h-72 items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">
        No active users in this period
      </div>
    );
  }
  return (
    <BarChart
      data={chart.data}
      index={chart.index}
      categories={chart.tags.map(extractUserAgent)}
      colors={STACKED_USAGE_PALETTE}
      valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
      yAxisWidth={48}
      maxBarSize={48}
      showLegend={true}
      stack={true}
      className="h-72"
    />
  );
}

interface UserAgentActivityProps {
  accessToken: string | null;
  userRole: string | null;
  dateValue: DateRangePickerValue;
  onDateChange?: (value: DateRangePickerValue) => void; // Optional - not used anymore
  /** Tags to pre-select, e.g. an agent's user-agent tags when opened from the Overview's Top agents. */
  initialTags?: readonly string[];
}

const UserAgentActivity: React.FC<UserAgentActivityProps> = ({
  accessToken,
  userRole,
  dateValue,
  onDateChange,
  initialTags,
}) => {
  const anchor = useComboboxAnchor();
  // Maximum number of categories to show in charts to prevent color palette overflow
  const MAX_CATEGORIES = 10;

  // Separate state for each endpoint
  const [dauData, setDauData] = useState<ActiveUsersAnalyticsResponse>({ results: [] });
  const [wauData, setWauData] = useState<ActiveUsersAnalyticsResponse>({ results: [] });
  const [mauData, setMauData] = useState<ActiveUsersAnalyticsResponse>({ results: [] });
  const [summaryData, setSummaryData] = useState<TagSummaryResponse>({ results: [] });

  const [userAgentFilter] = useState<string>("");

  // Tag filtering state
  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>(() => [...(initialTags ?? [])]);
  const [tagsLoading, setTagsLoading] = useState(false);

  // Separate loading states for each endpoint
  const [dauLoading, setDauLoading] = useState(false);
  const [wauLoading, setWauLoading] = useState(false);
  const [mauLoading, setMauLoading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);

  const [activityView, setActivityView] = useState<ActivityView>("active-users");
  const [period, setPeriod] = useState<ActivePeriod>("dau");

  // Use today's date as the end date for all API calls
  const today = new Date();

  const fetchAvailableTags = async () => {
    if (!accessToken) return;

    setTagsLoading(true);
    try {
      const data = await tagDistinctCall(accessToken);
      setAvailableTags(data.results.map((item: DistinctTagResponse) => item.tag));
    } catch (error) {
      console.error("Failed to fetch available tags:", error);
    } finally {
      setTagsLoading(false);
    }
  };

  const fetchDauData = async () => {
    if (!accessToken) return;

    setDauLoading(true);
    try {
      const data = await tagDauCall(
        accessToken,
        today,
        userAgentFilter || undefined,
        selectedTags.length > 0 ? selectedTags : undefined,
      );
      setDauData(data);
    } catch (error) {
      console.error("Failed to fetch DAU data:", error);
    } finally {
      setDauLoading(false);
    }
  };

  const fetchWauData = async () => {
    if (!accessToken) return;

    setWauLoading(true);
    try {
      const data = await tagWauCall(
        accessToken,
        today,
        userAgentFilter || undefined,
        selectedTags.length > 0 ? selectedTags : undefined,
      );
      setWauData(data);
    } catch (error) {
      console.error("Failed to fetch WAU data:", error);
    } finally {
      setWauLoading(false);
    }
  };

  const fetchMauData = async () => {
    if (!accessToken) return;

    setMauLoading(true);
    try {
      const data = await tagMauCall(
        accessToken,
        today,
        userAgentFilter || undefined,
        selectedTags.length > 0 ? selectedTags : undefined,
      );
      setMauData(data);
    } catch (error) {
      console.error("Failed to fetch MAU data:", error);
    } finally {
      setMauLoading(false);
    }
  };

  const fetchSummaryData = async () => {
    if (!accessToken || !dateValue.from || !dateValue.to) return;

    setSummaryLoading(true);
    try {
      const summary = await userAgentSummaryCall(
        accessToken,
        dateValue.from,
        dateValue.to,
        selectedTags.length > 0 ? selectedTags : undefined,
      );
      setSummaryData(summary);
    } catch (error) {
      console.error("Failed to fetch user agent summary data:", error);
    } finally {
      setSummaryLoading(false);
    }
  };

  // Effect to fetch available tags on mount
  useEffect(() => {
    fetchAvailableTags();
  }, [accessToken]);

  // Effect for DAU/WAU/MAU data (independent of date picker)
  useEffect(() => {
    if (!accessToken) return;

    const timeoutId = setTimeout(() => {
      fetchDauData();
      fetchWauData();
      fetchMauData();
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [accessToken, userAgentFilter, selectedTags]);

  // Effect for summary data (depends on date picker)
  useEffect(() => {
    if (!dateValue.from || !dateValue.to) return;

    const timeoutId = setTimeout(() => {
      fetchSummaryData();
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [accessToken, dateValue, selectedTags]);

  // Helper function to truncate user agent name (used with Ant Design Tooltip)
  const truncateUserAgent = (userAgent: string): string => {
    if (userAgent.length > 15) {
      return userAgent.substring(0, 15) + "...";
    }
    return userAgent;
  };

  // Get all user agents for each chart type based on their specific data
  const getAllTagsForData = (data: TagActiveUsersResponse[]) => {
    // Aggregate total active users per tag
    const tagTotals = data.reduce(
      (acc, item) => {
        acc[item.tag] = (acc[item.tag] || 0) + item.active_users;
        return acc;
      },
      {} as Record<string, number>,
    );

    // Sort by total active users and return all tags
    return Object.entries(tagTotals)
      .sort(([, a], [, b]) => b - a)
      .map(([tag]) => tag);
  };

  const allDauTags = getAllTagsForData(dauData.results).slice(0, MAX_CATEGORIES);
  const allWauTags = getAllTagsForData(wauData.results).slice(0, MAX_CATEGORIES);
  const allMauTags = getAllTagsForData(mauData.results).slice(0, MAX_CATEGORIES);

  // Prepare daily chart data (DAU) - always show last 7 days
  const generateDailyChartData = () => {
    const chartData: any[] = [];
    const endDate = new Date();

    // Generate all 7 days
    for (let i = 6; i >= 0; i--) {
      const date = new Date(endDate);
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split("T")[0]; // YYYY-MM-DD format

      const dayEntry: any = { date: dateStr };

      // Initialize all user agents to 0
      allDauTags.forEach((tag) => {
        const userAgent = extractUserAgent(tag);
        dayEntry[userAgent] = 0;
      });

      chartData.push(dayEntry);
    }

    // Fill in actual data
    dauData.results.forEach((item) => {
      const userAgent = extractUserAgent(item.tag);
      const dayEntry = chartData.find((d) => d.date === item.date);
      if (dayEntry) {
        dayEntry[userAgent] = item.active_users;
      }
    });

    return chartData;
  };

  const dailyChartData = generateDailyChartData();

  // Prepare weekly chart data (WAU) - always show all 7 weeks
  const generateWeeklyChartData = () => {
    const chartData: any[] = [];

    // Generate all 7 weeks (Week 1 through Week 7)
    for (let weekNum = 1; weekNum <= 7; weekNum++) {
      const weekEntry: any = { week: `Week ${weekNum}` };

      // Initialize all user agents to 0
      allWauTags.forEach((tag) => {
        const userAgent = extractUserAgent(tag);
        weekEntry[userAgent] = 0;
      });

      chartData.push(weekEntry);
    }

    // Fill in actual data
    wauData.results.forEach((item) => {
      const userAgent = extractUserAgent(item.tag);
      // Extract week number from the date field (e.g., "Week 1 (Jul 27)" -> "Week 1")
      const weekMatch = item.date.match(/Week (\d+)/);
      if (weekMatch) {
        const weekLabel = `Week ${weekMatch[1]}`;
        const weekEntry = chartData.find((d) => d.week === weekLabel);
        if (weekEntry) {
          weekEntry[userAgent] = item.active_users;
        }
      }
    });

    return chartData;
  };

  const weeklyChartData = generateWeeklyChartData();

  // Prepare monthly chart data (MAU) - always show all 7 months
  const generateMonthlyChartData = () => {
    const chartData: any[] = [];

    // Generate all 7 months (Month 1 through Month 7)
    for (let monthNum = 1; monthNum <= 7; monthNum++) {
      const monthEntry: any = { month: `Month ${monthNum}` };

      // Initialize all user agents to 0
      allMauTags.forEach((tag) => {
        const userAgent = extractUserAgent(tag);
        monthEntry[userAgent] = 0;
      });

      chartData.push(monthEntry);
    }

    // Fill in actual data
    mauData.results.forEach((item) => {
      const userAgent = extractUserAgent(item.tag);
      // Extract month number from the date field (e.g., "Month 1 (Jul)" -> "Month 1")
      const monthMatch = item.date.match(/Month (\d+)/);
      if (monthMatch) {
        const monthLabel = `Month ${monthMatch[1]}`;
        const monthEntry = chartData.find((d) => d.month === monthLabel);
        if (monthEntry) {
          monthEntry[userAgent] = item.active_users;
        }
      }
    });

    return chartData;
  };

  const monthlyChartData = generateMonthlyChartData();

  const activeUserCharts: readonly ActiveUserChart[] = [
    {
      value: "dau",
      title: "Daily Active Users - Last 7 Days",
      loading: dauLoading,
      data: dailyChartData,
      index: "date",
      tags: allDauTags,
    },
    {
      value: "wau",
      title: "Weekly Active Users - Last 7 Weeks",
      loading: wauLoading,
      data: weeklyChartData,
      index: "week",
      tags: allWauTags,
    },
    {
      value: "mau",
      title: "Monthly Active Users - Last 7 Months",
      loading: mauLoading,
      data: monthlyChartData,
      index: "month",
      tags: allMauTags,
    },
  ];

  const summaryRows = (summaryData.results || []).slice(0, SUMMARY_SLOTS);

  return (
    <div className="grid gap-3">
      <Panel
        icon={Bot}
        title="Summary by User Agent"
        subtitle="Performance metrics for different user agents"
        action={
          <div className="flex w-full items-center gap-2 sm:w-96">
            <label className="shrink-0 text-xs text-muted-foreground">Filter by User Agents</label>
            <Combobox
              multiple
              items={availableTags}
              value={selectedTags}
              onValueChange={(next: string[]) => setSelectedTags(next)}
            >
              <ComboboxChips render={<div ref={anchor} />} className="min-w-0 flex-1" aria-busy={tagsLoading}>
                <ComboboxValue>
                  {(selected: string[]) =>
                    selected.map((tag) => (
                      <ComboboxChip key={tag} aria-label={extractUserAgent(tag)}>
                        {truncateUserAgent(extractUserAgent(tag))}
                      </ComboboxChip>
                    ))
                  }
                </ComboboxValue>
                <ComboboxChipsInput placeholder="All User Agents" aria-label="All User Agents" />
                {selectedTags.length > 0 && <ComboboxClear aria-label="Clear user agent filter" />}
              </ComboboxChips>
              <ComboboxContent anchor={anchor}>
                <ComboboxEmpty>No user agents found</ComboboxEmpty>
                <ComboboxList>
                  {(tag: string) => {
                    const userAgent = extractUserAgent(tag);
                    return (
                      <ComboboxItem key={tag} value={tag} title={userAgent}>
                        {userAgent.length > 50 ? `${userAgent.substring(0, 50)}...` : userAgent}
                      </ComboboxItem>
                    );
                  }}
                </ComboboxList>
              </ComboboxContent>
            </Combobox>
          </div>
        }
      >
        {summaryLoading ? (
          <ChartSkeleton className="h-32" />
        ) : (
          <div className="grid grid-cols-1 overflow-hidden rounded-lg border sm:grid-cols-2 lg:grid-cols-4 lg:divide-x">
            {summaryRows.map((tag, index) => {
              const userAgent = extractUserAgent(tag.tag);
              return (
                <SummaryCell
                  key={index}
                  name={
                    <Tooltip>
                      <TooltipTrigger
                        render={
                          <h4 className="truncate text-sm font-medium text-foreground">
                            {truncateUserAgent(userAgent)}
                          </h4>
                        }
                      />
                      <TooltipContent side="top">{userAgent}</TooltipContent>
                    </Tooltip>
                  }
                  successRequests={formatAbbreviatedNumber(tag.successful_requests)}
                  totalTokens={formatAbbreviatedNumber(tag.total_tokens)}
                  totalCost={`$${formatAbbreviatedNumber(tag.total_spend, 4)}`}
                />
              );
            })}
            {Array.from({ length: Math.max(0, SUMMARY_SLOTS - summaryRows.length) }).map((_, index) => (
              <SummaryCell
                key={`empty-${index}`}
                name={<h4 className="truncate text-sm font-medium text-muted-foreground">No Data</h4>}
                successRequests="-"
                totalTokens="-"
                totalCost="-"
              />
            ))}
          </div>
        )}
      </Panel>

      {/* DAU/WAU/MAU vs Per User Usage. Every view stays mounted so switching does not reset its state. */}
      <Panel
        icon={Users}
        title={activityView === "active-users" ? "DAU, WAU & MAU per Agent" : "Per User Usage (Last 30 Days)"}
        subtitle={activityView === "active-users" ? "Active users across different time periods" : "Usage per end user"}
        action={
          <>
            {activityView === "active-users" && (
              <Segmented label="Active users period" value={period} options={PERIOD_OPTIONS} onChange={setPeriod} />
            )}
            <Segmented label="Activity view" value={activityView} options={VIEW_OPTIONS} onChange={setActivityView} />
          </>
        }
      >
        <div hidden={activityView !== "active-users"}>
          {activeUserCharts.map((chart) => (
            <div key={chart.value} hidden={period !== chart.value}>
              <h4 className="mb-2 text-xs text-muted-foreground">{chart.title}</h4>
              <ActiveUsersChart chart={chart} />
            </div>
          ))}
        </div>
        <div hidden={activityView !== "per-user"}>
          <PerUserUsage
            accessToken={accessToken}
            selectedTags={selectedTags}
            formatAbbreviatedNumber={formatAbbreviatedNumber}
          />
        </div>
      </Panel>
    </div>
  );
};

function SummaryCell({
  name,
  successRequests,
  totalTokens,
  totalCost,
}: {
  name: React.ReactNode;
  successRequests: string;
  totalTokens: string;
  totalCost: string;
}) {
  const rows = [
    { label: "Success Requests", value: successRequests },
    { label: "Total Tokens", value: totalTokens },
    { label: "Total Cost", value: totalCost },
  ];
  return (
    <div className="min-w-0 border-b px-4 py-3.5 last:border-b-0 lg:border-b-0">
      {name}
      <dl className="mt-3 grid gap-1.5">
        {rows.map((row) => (
          <div key={row.label} className="flex items-baseline justify-between gap-3">
            <dt className="text-xs text-muted-foreground">{row.label}</dt>
            <dd className="text-sm font-semibold tracking-tight tabular-nums text-foreground">{row.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export default UserAgentActivity;
