import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  Grid,
  BarChart,
  Metric,
  Subtitle,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
} from "@tremor/react";
import { Select, Tooltip, Switch, Table, Pagination, Input } from "antd";
const { Option } = Select;
const { Search } = Input;
import { userAgentSummaryCall, tagDauCall, tagWauCall, tagMauCall, tagDistinctCall, leaderboardCall, userDauCall, userWauCall, userMauCall } from "./networking";
import PerUserUsage from "./per_user_usage";
import { DateRangePickerValue } from "@tremor/react";
import { ChartLoader } from "./shared/chart_loader";
import { formatDate } from "./networking";

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

interface DistinctTagsResponse {
  results: DistinctTagResponse[];
}

interface LeaderboardUser {
  user_id: string;
  user_email: string | null;
  request_count: number;
}

interface LeaderboardResponse {
  results: LeaderboardUser[];
  total_count: number;
}

interface UserActiveUsersItem {
  date: string;
  active_users: number;
  period_start?: string;
  period_end?: string;
}

interface UserActiveUsersResponse {
  results: UserActiveUsersItem[];
}

interface UserAgentActivityProps {
  accessToken: string | null;
  userRole: string | null;
  dateValue: DateRangePickerValue;
  onDateChange?: (value: DateRangePickerValue) => void; // Optional - not used anymore
}

const UserAgentActivity: React.FC<UserAgentActivityProps> = ({ accessToken, userRole, dateValue, onDateChange }) => {
  // Maximum number of categories to show in charts to prevent color palette overflow
  const MAX_CATEGORIES = 10;

  // Separate state for each endpoint
  const [dauData, setDauData] = useState<ActiveUsersAnalyticsResponse>({ results: [] });
  const [wauData, setWauData] = useState<ActiveUsersAnalyticsResponse>({ results: [] });
  const [mauData, setMauData] = useState<ActiveUsersAnalyticsResponse>({ results: [] });
  const [summaryData, setSummaryData] = useState<TagSummaryResponse>({ results: [] });

  const [userAgentFilter, setUserAgentFilter] = useState<string>("");

  // Tag filtering state
  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [tagsLoading, setTagsLoading] = useState(false);

  // Separate loading states for each endpoint
  const [dauLoading, setDauLoading] = useState(false);
  const [wauLoading, setWauLoading] = useState(false);
  const [mauLoading, setMauLoading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [leaderboardLoading, setLeaderboardLoading] = useState(false);

  // Leaderboard state - now stores ALL users from backend
  const [leaderboardData, setLeaderboardData] = useState<LeaderboardResponse>({ results: [], total_count: 0 });
  const [leaderboardPage, setLeaderboardPage] = useState(1);
  const LEADERBOARD_PAGE_SIZE = 20;

  // Today's leaderboard data (separate from date-range leaderboard)
  const [todayLeaderboardData, setTodayLeaderboardData] = useState<LeaderboardResponse>({ results: [], total_count: 0 });

  // User-count based analytics state (not broken down by user-agent tag)
  const [userDauData, setUserDauData] = useState<UserActiveUsersResponse>({ results: [] });
  const [userWauData, setUserWauData] = useState<UserActiveUsersResponse>({ results: [] });
  const [userMauData, setUserMauData] = useState<UserActiveUsersResponse>({ results: [] });

  const [userDauLoading, setUserDauLoading] = useState(false);
  const [userWauLoading, setUserWauLoading] = useState(false);
  const [userMauLoading, setUserMauLoading] = useState(false);

  // Email search state for leaderboard
  const [leaderboardEmailSearch, setLeaderboardEmailSearch] = useState("");

  // MAU months selector state
  const [mauMonths, setMauMonths] = useState<number>(7);
  const MAU_MONTH_OPTIONS = [
    { value: 1, label: "Last 1 month" },
    { value: 2, label: "Last 2 months" },
    { value: 3, label: "Last 3 months" },
    { value: 4, label: "Last 4 months" },
    { value: 5, label: "Last 5 months" },
    { value: 6, label: "Last 6 months" },
    { value: 7, label: "Last 7 months" },
    { value: 8, label: "Last 8 months" },
    { value: 9, label: "Last 9 months" },
    { value: 10, label: "Last 10 months" },
    { value: 11, label: "Last 11 months" },
    { value: 12, label: "Last 12 months" },
  ];

  // Toggle state for filtering by hosted_vllm provider
  const [showHostedVllmOnly, setShowHostedVllmOnly] = useState(false);
  console.log('UserAgentActivity: Component mounted', { showHostedVllmOnly, CUSTOM_LLM_PROVIDER: 'hosted_vllm' });
  const CUSTOM_LLM_PROVIDER = "hosted_vllm"; // Provider name to filter by

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
        dateValue.from ? formatDate(dateValue.from) : undefined,
        dateValue.to ? formatDate(dateValue.to) : undefined,
        userAgentFilter || undefined,
        selectedTags.length > 0 ? selectedTags : undefined,
        showHostedVllmOnly ? CUSTOM_LLM_PROVIDER : undefined,
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
        userAgentFilter || undefined,
        selectedTags.length > 0 ? selectedTags : undefined,
        showHostedVllmOnly ? CUSTOM_LLM_PROVIDER : undefined,
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
        mauMonths,
        userAgentFilter || undefined,
        selectedTags.length > 0 ? selectedTags : undefined,
        showHostedVllmOnly ? CUSTOM_LLM_PROVIDER : undefined,
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
        showHostedVllmOnly ? CUSTOM_LLM_PROVIDER : undefined,
      );
      setSummaryData(summary);
    } catch (error) {
      console.error("Failed to fetch user agent summary data:", error);
    } finally {
      setSummaryLoading(false);
    }
  };

  const fetchLeaderboardData = async () => {
    if (!accessToken) return;

    setLeaderboardLoading(true);
    try {
      const data = await leaderboardCall(
        accessToken,
        dateValue.from ? formatDate(dateValue.from) : undefined,
        dateValue.to ? formatDate(dateValue.to) : undefined,
        showHostedVllmOnly ? CUSTOM_LLM_PROVIDER : undefined,
      );
      setLeaderboardData(data);
      // Reset to first page when data changes
      setLeaderboardPage(1);
    } catch (error) {
      console.error("Failed to fetch leaderboard data:", error);
    } finally {
      setLeaderboardLoading(false);
    }
  };

  const fetchTodayLeaderboardData = async () => {
    if (!accessToken) return;

    try {
      const today = new Date();
      const todayStr = formatDate(today);
      const data = await leaderboardCall(
        accessToken,
        todayStr,
        todayStr,
        showHostedVllmOnly ? CUSTOM_LLM_PROVIDER : undefined,
      );
      setTodayLeaderboardData(data);
    } catch (error) {
      console.error("Failed to fetch today's leaderboard data:", error);
    }
  };

  const fetchUserDauData = async () => {
    if (!accessToken) return;

    setUserDauLoading(true);
    try {
      const data = await userDauCall(
        accessToken,
        dateValue.from ? formatDate(dateValue.from) : undefined,
        dateValue.to ? formatDate(dateValue.to) : undefined,
        showHostedVllmOnly ? CUSTOM_LLM_PROVIDER : undefined,
      );
      setUserDauData(data);
    } catch (error) {
      console.error("Failed to fetch user DAU data:", error);
    } finally {
      setUserDauLoading(false);
    }
  };

  const fetchUserWauData = async () => {
    if (!accessToken) return;

    setUserWauLoading(true);
    try {
      const data = await userWauCall(
        accessToken,
        showHostedVllmOnly ? CUSTOM_LLM_PROVIDER : undefined,
      );
      setUserWauData(data);
    } catch (error) {
      console.error("Failed to fetch user WAU data:", error);
    } finally {
      setUserWauLoading(false);
    }
  };

  const fetchUserMauData = async () => {
    if (!accessToken) return;

    setUserMauLoading(true);
    try {
      const data = await userMauCall(
        accessToken,
        mauMonths,
        showHostedVllmOnly ? CUSTOM_LLM_PROVIDER : undefined,
      );
      setUserMauData(data);
    } catch (error) {
      console.error("Failed to fetch user MAU data:", error);
    } finally {
      setUserMauLoading(false);
    }
  };

  // Effect to fetch available tags on mount
  useEffect(() => {
    fetchAvailableTags();
  }, [accessToken]);

  // Effect for DAU/WAU/MAU data (depends on date picker for DAU, mauMonths for MAU)
  useEffect(() => {
    if (!accessToken) return;

    const timeoutId = setTimeout(() => {
      fetchDauData();
      fetchWauData();
      fetchMauData();
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [accessToken, userAgentFilter, selectedTags, showHostedVllmOnly, dateValue, mauMonths]);

  // Effect for summary data (depends on date picker)
  useEffect(() => {
    if (!dateValue.from || !dateValue.to) return;

    const timeoutId = setTimeout(() => {
      fetchSummaryData();
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [accessToken, dateValue, selectedTags, showHostedVllmOnly]);

  // Effect for leaderboard data
  useEffect(() => {
    if (!accessToken) return;

    const timeoutId = setTimeout(() => {
      fetchLeaderboardData();
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [accessToken, showHostedVllmOnly, dateValue]);

  // Effect for today's leaderboard data (always fetches for today only)
  useEffect(() => {
    if (!accessToken) return;

    const timeoutId = setTimeout(() => {
      fetchTodayLeaderboardData();
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [accessToken, showHostedVllmOnly]);

  // Effect for user-count based analytics (User DAU/WAU/MAU)
  useEffect(() => {
    if (!accessToken) return;

    const timeoutId = setTimeout(() => {
      fetchUserDauData();
      fetchUserWauData();
      fetchUserMauData();
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [accessToken, showHostedVllmOnly, dateValue, mauMonths]);

  // Reset to first page when search changes
  useEffect(() => {
    setLeaderboardPage(1);
  }, [leaderboardEmailSearch]);

  // Filter leaderboard data by email search (client-side)
  const filteredLeaderboardResults = leaderboardData.results.filter((user) => {
    if (!leaderboardEmailSearch) return true;
    const searchLower = leaderboardEmailSearch.toLowerCase();
    return (
      user.user_email?.toLowerCase().includes(searchLower) ||
      user.user_id.toLowerCase().includes(searchLower)
    );
  });

  // Get paginated results
  const paginatedLeaderboardResults = filteredLeaderboardResults.slice(
    (leaderboardPage - 1) * LEADERBOARD_PAGE_SIZE,
    leaderboardPage * LEADERBOARD_PAGE_SIZE
  );

  // Helper function to extract user agent from tag
  const extractUserAgent = (tag: string): string => {
    if (tag.startsWith("User-Agent: ")) {
      return tag.replace("User-Agent: ", "");
    }
    return tag;
  };

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

  // Prepare daily chart data (DAU) - based on selected date range
  const generateDailyChartData = () => {
    const chartData: any[] = [];

    if (!dateValue.from || !dateValue.to) return chartData;

    const startDate = new Date(dateValue.from);
    const endDate = new Date(dateValue.to);

    // Calculate number of days in range
    const dayDiff = Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));

    // Generate all days in the range
    for (let i = 0; i <= dayDiff; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
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

  // Prepare weekly chart data (WAU) - generate all 7 weeks
  const generateWeeklyChartData = () => {
    const chartData: any[] = [];

    // Generate all 7 week labels (most recent first)
    // Backend format: "Week X (Mon DD)" where X = 1 (earliest) to 7 (most recent)
    const allWeekLabels: string[] = [];
    for (let i = 0; i < 7; i++) {
      const now = new Date();
      // Add 1 day like backend does
      const endDate = new Date(now);
      endDate.setDate(now.getDate() - (i * 7) + 1);
      const startDate = new Date(endDate);
      startDate.setDate(endDate.getDate() - 6);

      // Backend format: "Week {7-i} ({startMonth} {startDay})"
      // Week 1 = earliest (7 weeks ago), Week 7 = most recent
      const startMonth = startDate.toLocaleDateString('en-US', { month: 'short' });
      const startDay = startDate.getDate();
      const weekNum = 7 - i;

      allWeekLabels.push(`Week ${weekNum} (${startMonth} ${startDay})`);
    }

    // Generate entries for each week (all 7 weeks, even if empty)
    allWeekLabels.forEach((weekLabel) => {
      const weekEntry: any = { week: weekLabel };

      // Initialize all user agents to 0
      allWauTags.forEach((tag) => {
        const userAgent = extractUserAgent(tag);
        weekEntry[userAgent] = 0;
      });

      chartData.push(weekEntry);
    });

    // Fill in actual data from API response
    wauData.results.forEach((item) => {
      const userAgent = extractUserAgent(item.tag);
      // Match by date label (API format: "Week 1 (Jan 15)")
      const weekEntry = chartData.find((d) => d.week === item.date);
      if (weekEntry) {
        weekEntry[userAgent] = item.active_users;
      }
    });

    return chartData;
  };

  const weeklyChartData = generateWeeklyChartData();

  // Prepare monthly chart data (MAU) - generate all months based on selection
  const generateMonthlyChartData = () => {
      const chartData: any[] = [];
      const now = new Date();
        
        const allMonthLabels: string[] = [];
        for (let i = 0; i < mauMonths; i++) {
          const d = new Date(now.getFullYear(), now.getMonth() - i, 1);
          // Use 'short' for month and 'numeric' for year for matching backend format
          // Backend uses: TO_CHAR(..., 'Mon YYYY')
      const monthName = d.toLocaleString('default', { month: 'short', year: 'numeric' }); 
          allMonthLabels.push(monthName);
        }

    // Generate entries for each month (all months, even if empty)
    allMonthLabels.forEach((monthLabel) => {
      const monthEntry: any = { month: monthLabel };

      // Initialize all user agents to 0
      allMauTags.forEach((tag) => {
        const userAgent = extractUserAgent(tag);
        monthEntry[userAgent] = 0;
      });

      chartData.push(monthEntry);
    });

    // Fill in actual data from API response
    mauData.results.forEach((item) => {
      const userAgent = extractUserAgent(item.tag);
      const monthEntry = chartData.find((d) => d.month === item.date);
      if (monthEntry) {
        monthEntry[userAgent] = item.active_users;
      }
    });

    return chartData;
  };

  const monthlyChartData = generateMonthlyChartData();

  // User-count based chart data generation
  const generateUserDauChartData = () => {
    const chartData: any[] = [];

    if (!dateValue.from || !dateValue.to) return chartData;
    if (!userDauData.results || userDauData.results.length === 0) {
      // Return empty data with all dates initialized to 0
      const startDate = new Date(dateValue.from);
      const endDate = new Date(dateValue.to);
      const dayDiff = Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));

      for (let i = 0; i <= dayDiff; i++) {
        const date = new Date(startDate);
        date.setUTCDate(date.getUTCDate() + i);
        const dateStr = date.toISOString().split("T")[0];
        chartData.push({ date: dateStr, "Active Users": 0 });
      }
      return chartData;
    }

    const startDate = new Date(dateValue.from);
    const endDate = new Date(dateValue.to);
    const dayDiff = Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));

    // Generate all days in the range with 0 initialization
    for (let i = 0; i <= dayDiff; i++) {
      const date = new Date(startDate);
      date.setUTCDate(date.getUTCDate() + i);
      const dateStr = date.toISOString().split("T")[0];
      chartData.push({ date: dateStr, "Active Users": 0 });
    }

    // Fill in actual data
    userDauData.results.forEach((item) => {
      const dayEntry = chartData.find((d) => d.date === item.date);
      if (dayEntry) {
        dayEntry["Active Users"] = item.active_users;
      }
    });

    return chartData;
  };

  const generateUserWauChartData = () => {
    const chartData: any[] = [];

    // Build chart data directly from API response dates to avoid timezone/format mismatches
    if (userWauData.results && userWauData.results.length > 0) {
      userWauData.results.forEach((item) => {
        chartData.push({
          week: item.date,
          "Active Users": item.active_users,
        });
      });
    }

    return chartData;
  };

  const generateUserMauChartData = () => {
    const chartData: any[] = [];

    // Build chart data directly from API response dates to avoid timezone/format mismatches
    if (userMauData.results && userMauData.results.length > 0) {
      userMauData.results.forEach((item) => {
        chartData.push({
          month: item.date,
          "Active Users": item.active_users,
        });
      });
    }

    return chartData;
  };

  const userDauChartData = generateUserDauChartData();
  const userWauChartData = generateUserWauChartData();
  const userMauChartData = generateUserMauChartData();

  // Format numbers with K, M abbreviations
  const formatAbbreviatedNumber = (value: number, decimalPlaces: number = 0): string => {
    if (value >= 100000000) {
      return (value / 1000000).toFixed(decimalPlaces) + "M";
    } else if (value >= 10000000) {
      return (value / 1000000).toFixed(decimalPlaces) + "M";
    } else if (value >= 1000000) {
      return (value / 1000000).toFixed(decimalPlaces) + "M";
    } else if (value >= 10000) {
      return (value / 1000).toFixed(decimalPlaces) + "K";
    } else if (value >= 1000) {
      return (value / 1000).toFixed(decimalPlaces) + "K";
    } else {
      return value.toFixed(decimalPlaces);
    }
  };

  // Debug: Log component state
  console.log('UserAgentActivity: Component state', {
    showHostedVllmOnly,
    CUSTOM_LLM_PROVIDER,
    mountTime: new Date().toISOString()
  });

  return (
    <div className="space-y-6 mt-6">
      {/* User Trends Section - User count based analytics (not broken down by user-agent tag) */}
      <Card>
        <div className="mb-6 flex items-start justify-between">
          {/* Left side: Titles */}
          <div>
            <Title>User Trends</Title>
            <Subtitle>Unique user activity over time (not broken down by user-agent)</Subtitle>
          </div>

          {/* Right side: Toggle */}
          <div className="flex items-center gap-2">
            <Switch
              checked={showHostedVllmOnly}
              onChange={setShowHostedVllmOnly}
              size="small"
            />
            <Text className="text-sm whitespace-nowrap">
              Self-hosted models only
            </Text>
          </div>
        </div>

        <TabGroup>
          <TabList className="mb-6">
            <Tab>Daily</Tab>
            <Tab>Weekly</Tab>
            <Tab>Monthly</Tab>
          </TabList>

          <TabPanels>
            {/* Daily User Trends */}
            <TabPanel>
              <div className="flex gap-6">
                {/* Left: Today's Active Users Card */}
                <Card className="w-64 flex-shrink-0">
                  <Title className="text-base">Today&apos;s Active Users</Title>
                  <Subtitle className="text-xs text-gray-500">Independent of date range</Subtitle>
                  {userDauLoading ? (
                    <ChartLoader isDateChanging={false} />
                  ) : (
                    <>
                      <Metric className="text-4xl mt-2">
                        {formatAbbreviatedNumber(todayLeaderboardData.total_count || 0)}
                      </Metric>
                      <Text className="mt-1">users today</Text>
                    </>
                  )}
                </Card>

                {/* Right: Bar Chart */}
                <div className="flex-1">
                  <div className="mb-4">
                    <Title className="text-lg">
                      Daily Active Users
                      {dateValue.from && dateValue.to && (
                        <span className="text-gray-500 font-normal ml-2">
                          ({formatDate(dateValue.from)} to {formatDate(dateValue.to)})
                        </span>
                      )}
                    </Title>
                  </div>
                  {userDauLoading ? (
                    <ChartLoader isDateChanging={false} />
                  ) : (
                    <div className="flex-1 overflow-x-auto pb-2">
                      <BarChart
                        data={userDauChartData}
                        index="date"
                        categories={["Active Users"]}
                        valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                        yAxisWidth={60}
                        showLegend={true}
                        tickGap={5}
                      />
                    </div>
                  )}
                </div>
              </div>
            </TabPanel>

            {/* Weekly User Trends */}
            <TabPanel>
              <div className="mb-4">
                <Title className="text-lg">Weekly Active Users (Last 7 Weeks)</Title>
              </div>
              {userWauLoading ? (
                <ChartLoader isDateChanging={false} />
              ) : (
                <BarChart
                  data={userWauChartData}
                  index="week"
                  categories={["Active Users"]}
                  valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                  yAxisWidth={60}
                  showLegend={true}
                />
              )}
            </TabPanel>

            {/* Monthly User Trends */}
            <TabPanel>
              <div className="mb-4 flex items-center gap-4">
                <Title className="text-lg">Monthly Active Users</Title>
                <Select
                  value={mauMonths}
                  onChange={setMauMonths}
                  style={{ width: 160 }}
                  size="small"
                >
                  {MAU_MONTH_OPTIONS.map((opt) => (
                    <Option key={opt.value} value={opt.value}>{opt.label}</Option>
                  ))}
                </Select>
              </div>
              {userMauLoading ? (
                <ChartLoader isDateChanging={false} />
              ) : (
                <BarChart
                  data={userMauChartData}
                  index="month"
                  categories={["Active Users"]}
                  valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                  yAxisWidth={60}
                  showLegend={true}
                  tickGap={5}
                />
              )}
            </TabPanel>
          </TabPanels>
        </TabGroup>
      </Card>
      {/* Leaderboard Section Card */}
      <Card>
        <div className="space-y-4">
          <div className="flex justify-between items-start">
            <div>
              <Title>Top Active Users</Title>
              <Subtitle>
                Most active users by request count
                {dateValue.from && dateValue.to && (
                  <span className="text-gray-500 ml-1">
                    ({formatDate(dateValue.from)} to {formatDate(dateValue.to)})
                  </span>
                )}
              </Subtitle>
            </div>
            <Search
              placeholder="Search by email or user ID"
              allowClear
              value={leaderboardEmailSearch}
              onChange={(e) => setLeaderboardEmailSearch(e.target.value)}
              style={{ width: 350 }}
              size="middle"
            />
          </div>

          {leaderboardLoading ? (
            <ChartLoader isDateChanging={false} />
          ) : (
            <>
              <Table
                dataSource={paginatedLeaderboardResults.map((user, index) => ({
                  key: user.user_id,
                  rank: (leaderboardPage - 1) * LEADERBOARD_PAGE_SIZE + index + 1,
                  user_id: user.user_id,
                  user_email: user.user_email || "-",
                  request_count: user.request_count,
                }))}
                columns={[
                  {
                    title: "Rank",
                    dataIndex: "rank",
                    key: "rank",
                    width: 60,
                    render: (rank: number) => (
                      <span className={`font-bold ${rank <= 3 ? "text-yellow-600" : ""}`}>
                        {rank <= 3 ? ["🥇", "🥈", "🥉"][rank - 1] : `#${rank}`}
                      </span>
                    ),
                  },
                  {
                    title: "User ID",
                    dataIndex: "user_id",
                    key: "user_id",
                    ellipsis: true,
                  },
                  {
                    title: "Email",
                    dataIndex: "user_email",
                    key: "user_email",
                    ellipsis: true,
                  },
                  {
                    title: "Requests",
                    dataIndex: "request_count",
                    key: "request_count",
                    width: 120,
                    render: (count: number) => formatAbbreviatedNumber(count),
                  },
                ]}
                pagination={false}
                size="small"
                scroll={{ y: 400 }}
              />
              {filteredLeaderboardResults.length > 0 && (
                <div className="flex justify-between items-center mt-4">
                  <Text className="text-sm text-gray-500">
                    Showing {paginatedLeaderboardResults.length} of {filteredLeaderboardResults.length} users
                    {leaderboardEmailSearch && filteredLeaderboardResults.length !== leaderboardData.total_count && (
                      <span className="ml-2">
                        (filtered from {leaderboardData.total_count} total)
                      </span>
                    )}
                  </Text>
                  <Pagination
                    current={leaderboardPage}
                    pageSize={LEADERBOARD_PAGE_SIZE}
                    total={filteredLeaderboardResults.length}
                    onChange={(page) => setLeaderboardPage(page)}
                    showSizeChanger={false}
                    showTotal={(total, range) => `${range[0]}-${range[1]} of ${total} users`}
                  />
                </div>
              )}
              {filteredLeaderboardResults.length === 0 && leaderboardData.total_count > 0 && (
                <div className="text-center py-4">
                  <Text className="text-gray-500">No users match your search</Text>
                </div>
              )}
            </>
          )}
        </div>
      </Card>
      {/* Summary Section Card */}
      <Card>
        <div className="space-y-6">
          <div className="flex justify-between items-start">
            <div>
              <Title>Summary by User Agent</Title>
              <Subtitle>Performance metrics for different user agents</Subtitle>
            </div>

            <div className="flex items-center gap-6">

              {/* User Agent Filter */}
              <div className="w-80">
                <Text className="text-sm font-medium block mb-2">Filter by User Agents</Text>
                <Select
                  mode="multiple"
                  placeholder="All User Agents"
                  value={selectedTags}
                  onChange={setSelectedTags}
                  style={{ width: "100%" }}
                  showSearch={true}
                  allowClear={true}
                  loading={tagsLoading}
                  optionFilterProp="label"
                  className="rounded-md"
                  maxTagCount="responsive"
                >
                  {availableTags.map((tag) => {
                    const userAgent = extractUserAgent(tag);
                    const displayName = userAgent.length > 50 ? `${userAgent.substring(0, 50)}...` : userAgent;
                    return (
                      <Select.Option key={tag} value={tag} label={displayName} title={userAgent}>
                        {displayName}
                      </Select.Option>
                    );
                  })}
                </Select>
              </div>
            </div>
          </div>

          {/* Date Range Picker is controlled by parent component */}

          {/* Top 4 User Agents Cards */}
          {summaryLoading ? (
            <ChartLoader isDateChanging={false} />
          ) : (
            <Grid numItems={4} className="gap-4">
              {(summaryData.results || []).slice(0, 4).map((tag, index) => {
                const userAgent = extractUserAgent(tag.tag);
                const displayName = truncateUserAgent(userAgent);
                return (
                  <Card key={index}>
                    <Tooltip title={userAgent} placement="top">
                      <Title className="truncate">{displayName}</Title>
                    </Tooltip>
                    <div className="mt-4 space-y-3">
                      <div>
                        <Text className="text-sm text-gray-600">Success Requests</Text>
                        <Metric className="text-lg">{formatAbbreviatedNumber(tag.successful_requests)}</Metric>
                      </div>
                      <div>
                        <Text className="text-sm text-gray-600">Total Tokens</Text>
                        <Metric className="text-lg">{formatAbbreviatedNumber(tag.total_tokens)}</Metric>
                      </div>
                      <div>
                        <Text className="text-sm text-gray-600">Users</Text>
                        <Metric className="text-lg">{formatAbbreviatedNumber(tag.unique_users)}</Metric>
                      </div>
                    </div>
                  </Card>
                );
              })}
              {/* Fill remaining slots if less than 4 agents */}
              {Array.from({ length: Math.max(0, 4 - (summaryData.results || []).length) }).map((_, index) => (
                <Card key={`empty-${index}`}>
                  <Title>No Data</Title>
                  <div className="mt-4 space-y-3">
                    <div>
                      <Text className="text-sm text-gray-600">Success Requests</Text>
                      <Metric className="text-lg">-</Metric>
                    </div>
                    <div>
                      <Text className="text-sm text-gray-600">Total Tokens</Text>
                      <Metric className="text-lg">-</Metric>
                    </div>
                    <div>
                      <Text className="text-sm text-gray-600">Users</Text>
                      <Metric className="text-lg">-</Metric>
                    </div>
                  </div>
                </Card>
              ))}
            </Grid>
          )}
        </div>
      </Card>

      {/* DAU/WAU/MAU per Agent Section */}
      <Card>
        <div className="mb-6">
          <Title>DAU, WAU & MAU per Agent</Title>
          <Subtitle>Active users across different time periods</Subtitle>
        </div>

        <TabGroup>
          <TabList className="mb-6">
            <Tab>DAU</Tab>
            <Tab>WAU</Tab>
            <Tab>MAU</Tab>
          </TabList>

          <TabPanels>
            <TabPanel>
              <div className="mb-4">
                <Title className="text-lg">
                  Daily Active Users Agent
                  {dateValue.from && dateValue.to && (
                    <span className="text-gray-500 font-normal ml-2">
                      ({formatDate(dateValue.from)} to {formatDate(dateValue.to)})
                    </span>
                  )}
                </Title>
              </div>
              {dauLoading ? (
                <ChartLoader isDateChanging={false} />
              ) : (
                <div className="flex-1 overflow-x-auto pb-2">
                  <BarChart
                    data={dailyChartData}
                    index="date"
                    categories={allDauTags.map(extractUserAgent)}
                    valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                    yAxisWidth={60}
                    showLegend={true}
                    stack={true}
                    tickGap={5}
                  />
                </div>
              )}
            </TabPanel>

            <TabPanel>
              <div className="mb-4">
                <Title className="text-lg">Weekly Active Users Agent(Last 7 Weeks) </Title>
              </div>
              {wauLoading ? (
                <ChartLoader isDateChanging={false} />
              ) : (
                <BarChart
                  data={weeklyChartData}
                  index="week"
                  categories={allWauTags.map(extractUserAgent)}
                  valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                  yAxisWidth={60}
                  showLegend={true}
                  stack={true}
                />
              )}
            </TabPanel>

            <TabPanel>
              <div className="mb-4 flex items-center gap-4">
                <Title className="text-lg">Monthly Active Users Agent</Title>
                <Select
                  value={mauMonths}
                  onChange={setMauMonths}
                  style={{ width: 160 }}
                  size="small"
                >
                  {MAU_MONTH_OPTIONS.map((opt) => (
                    <Option key={opt.value} value={opt.value}>{opt.label}</Option>
                  ))}
                </Select>
              </div>
              {mauLoading ? (
                <ChartLoader isDateChanging={false} />
              ) : (
                <BarChart
                  data={monthlyChartData}
                  index="month"
                  categories={allMauTags.map(extractUserAgent)}
                  valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                  yAxisWidth={60}
                  showLegend={true}
                  stack={true}
                  tickGap={5}
                />
              )}
            </TabPanel>
          </TabPanels>
        </TabGroup>
      </Card>
    </div>
  );
};

export default UserAgentActivity;
