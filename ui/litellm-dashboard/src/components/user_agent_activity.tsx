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
import { Select, Tooltip } from "antd";
import { userAgentSummaryCall, tagDauCall, tagWauCall, tagMauCall, tagDistinctCall } from "./networking";
import PerUserUsage from "./per_user_usage";
import { DateRangePickerValue } from "@tremor/react";
import { ChartLoader } from "./shared/chart_loader";

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

  return (
    <div className="space-y-6 mt-6">
      {/* Summary Section Card */}
      <Card>
        <div className="space-y-6">
          <div className="flex justify-between items-start">
            <div>
              <Title>Summary by User Agent</Title>
              <Subtitle>Performance metrics for different user agents</Subtitle>
            </div>

            {/* User Agent Filter */}
            <div className="w-96">
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
                        <Text className="text-sm text-gray-600">Total Cost</Text>
                        <Metric className="text-lg">${formatAbbreviatedNumber(tag.total_spend, 4)}</Metric>
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
                      <Text className="text-sm text-gray-600">Total Cost</Text>
                      <Metric className="text-lg">-</Metric>
                    </div>
                  </div>
                </Card>
              ))}
            </Grid>
          )}
        </div>
      </Card>

      {/* Main TabGroup for DAU/WAU/MAU vs Per User Usage */}
      <Card>
        <TabGroup>
          <TabList className="mb-6">
            <Tab>DAU/WAU/MAU</Tab>
            <Tab>Per User Usage (Last 30 Days)</Tab>
          </TabList>

          <TabPanels>
            {/* DAU/WAU/MAU Tab Panel */}
            <TabPanel>
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
                      <Title className="text-lg">Daily Active Users - Last 7 Days</Title>
                    </div>
                    {dauLoading ? (
                      <ChartLoader isDateChanging={false} />
                    ) : (
                      <BarChart
                        data={dailyChartData}
                        index="date"
                        categories={allDauTags.map(extractUserAgent)}
                        valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                        yAxisWidth={60}
                        showLegend={true}
                        stack={true}
                      />
                    )}
                  </TabPanel>

                  <TabPanel>
                    <div className="mb-4">
                      <Title className="text-lg">Weekly Active Users - Last 7 Weeks</Title>
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
                    <div className="mb-4">
                      <Title className="text-lg">Monthly Active Users - Last 7 Months</Title>
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
                      />
                    )}
                  </TabPanel>
                </TabPanels>
              </TabGroup>
            </TabPanel>

            {/* Per User Usage Tab Panel */}
            <TabPanel>
              <PerUserUsage
                accessToken={accessToken}
                selectedTags={selectedTags}
                formatAbbreviatedNumber={formatAbbreviatedNumber}
              />
            </TabPanel>
          </TabPanels>
        </TabGroup>
      </Card>
    </div>
  );
};

export default UserAgentActivity;
