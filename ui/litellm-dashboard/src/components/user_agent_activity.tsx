import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  Grid,
  Col,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  BarChart,
  DonutChart,
  Metric,
  Subtitle,
  Select,
  SelectItem,
  Button,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
} from "@tremor/react";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { userAgentSummaryCall, tagDauCall, tagWauCall, tagMauCall } from "./networking";
import AdvancedDatePicker from "./shared/advanced_date_picker";
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

interface UserAgentActivityProps {
  accessToken: string | null;
  userRole: string | null;
}

const UserAgentActivity: React.FC<UserAgentActivityProps> = ({
  accessToken,
  userRole,
}) => {
  // Separate state for each endpoint
  const [dauData, setDauData] = useState<ActiveUsersAnalyticsResponse>({ results: [] });
  const [wauData, setWauData] = useState<ActiveUsersAnalyticsResponse>({ results: [] });
  const [mauData, setMauData] = useState<ActiveUsersAnalyticsResponse>({ results: [] });
  const [summaryData, setSummaryData] = useState<TagSummaryResponse>({ results: [] });

  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    to: new Date(),
  });

  const [userAgentFilter, setUserAgentFilter] = useState<string>("");
  
  // Separate loading states for each endpoint
  const [dauLoading, setDauLoading] = useState(false);
  const [wauLoading, setWauLoading] = useState(false);
  const [mauLoading, setMauLoading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);
  
  const [isDateChanging, setIsDateChanging] = useState(false);

  // Use today's date as the end date for all API calls
  const today = new Date();

  const fetchDauData = async () => {
    if (!accessToken) return;

    setDauLoading(true);
    try {
      const data = await tagDauCall(
        accessToken,
        today,
        userAgentFilter || undefined
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
        userAgentFilter || undefined
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
        userAgentFilter || undefined
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
      const summary = await userAgentSummaryCall(accessToken, dateValue.from, dateValue.to);
      setSummaryData(summary);
    } catch (error) {
      console.error("Failed to fetch user agent summary data:", error);
    } finally {
      setSummaryLoading(false);
      setIsDateChanging(false);
    }
  };

  // Super responsive date change handler
  const handleDateChange = (newValue: DateRangePickerValue) => {
    // Instant visual feedback
    setIsDateChanging(true);
    setSummaryLoading(true);

    // Update date immediately for UI responsiveness
    setDateValue(newValue);
  };

  // Effect for DAU/WAU/MAU data (independent of date picker)
  useEffect(() => {
    if (!accessToken) return;

    const timeoutId = setTimeout(() => {
      fetchDauData();
      fetchWauData();
      fetchMauData();
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [accessToken, userAgentFilter]);

  // Effect for summary data (depends on date picker)
  useEffect(() => {
    if (!dateValue.from || !dateValue.to) return;

    const timeoutId = setTimeout(() => {
      fetchSummaryData();
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [accessToken, dateValue]);

  // Helper function to extract user agent from tag
  const extractUserAgent = (tag: string): string => {
    if (tag.startsWith("User-Agent: ")) {
      return tag.replace("User-Agent: ", "");
    }
    return tag;
  };

  // Helper function to truncate user agent name with tooltip
  const truncateUserAgent = (userAgent: string): string => {
    if (userAgent.length > 10) {
      return userAgent.substring(0, 10) + "...";
    }
    return userAgent;
  };

  // Get unique user agents for charts
  const getAllUniqueTags = () => {
    const allTags = new Set<string>();
    dauData.results.forEach(item => allTags.add(item.tag));
    wauData.results.forEach(item => allTags.add(item.tag));
    mauData.results.forEach(item => allTags.add(item.tag));
    return Array.from(allTags).slice(0, 3); // Top 3 tags
  };

  const uniqueTags = getAllUniqueTags();

  // Prepare daily chart data (DAU)
  const dailyChartData = dauData.results.reduce((acc, item) => {
    const existingDate = acc.find(d => d.date === item.date);
    const userAgent = extractUserAgent(item.tag);
    
    if (existingDate) {
      existingDate[userAgent] = item.active_users;
    } else {
      const newDateEntry: any = { 
        date: item.date,
        [userAgent]: item.active_users
      };
      acc.push(newDateEntry);
    }
    return acc;
  }, [] as any[]);

  // Sort daily data by date
  dailyChartData.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  // Prepare weekly chart data (WAU)
  const weeklyChartData = wauData.results.reduce((acc, item) => {
    const userAgent = extractUserAgent(item.tag);
    const weekLabel = item.period_start && item.period_end 
      ? `${item.period_start} to ${item.period_end}` 
      : item.date;
    
    const existingWeek = acc.find(d => d.week === weekLabel);
    if (existingWeek) {
      existingWeek[userAgent] = item.active_users;
    } else {
      const newWeekEntry: any = { 
        week: weekLabel,
        [userAgent]: item.active_users
      };
      acc.push(newWeekEntry);
    }
    return acc;
  }, [] as any[]);

  // Prepare monthly chart data (MAU)
  const monthlyChartData = mauData.results.reduce((acc, item) => {
    const userAgent = extractUserAgent(item.tag);
    const monthLabel = item.period_start && item.period_end 
      ? `${item.period_start} to ${item.period_end}` 
      : item.date;
    
    const existingMonth = acc.find(d => d.month === monthLabel);
    if (existingMonth) {
      existingMonth[userAgent] = item.active_users;
    } else {
      const newMonthEntry: any = { 
        month: monthLabel,
        [userAgent]: item.active_users
      };
      acc.push(newMonthEntry);
    }
    return acc;
  }, [] as any[]);

  // Format numbers with K, M abbreviations
  const formatAbbreviatedNumber = (value: number, decimalPlaces: number = 0): string => {
    if (value >= 100000000) {
      return (value / 1000000).toFixed(decimalPlaces) + 'M';
    } else if (value >= 10000000) {
      return (value / 1000000).toFixed(decimalPlaces) + 'M';
    } else if (value >= 1000000) {
      return (value / 1000000).toFixed(decimalPlaces) + 'M';
    } else if (value >= 10000) {
      return (value / 1000).toFixed(decimalPlaces) + 'K';
    } else if (value >= 1000) {
      return (value / 1000).toFixed(decimalPlaces) + 'K';
    } else {
      return value.toFixed(decimalPlaces);
    }
  };

  return (
    <div className="space-y-6">
      {/* Date Range Picker */}
      <Grid numItems={2} className="gap-2 w-full">
        <Col>
          <AdvancedDatePicker
            value={dateValue}
            onValueChange={handleDateChange}
          />
        </Col>
        <Col>
          <Select
            value={userAgentFilter}
            onValueChange={setUserAgentFilter}
            placeholder="Filter by User Agent"
          >
            <SelectItem value="">All User Agents</SelectItem>
            <SelectItem value="curl">curl</SelectItem>
            <SelectItem value="litellm">litellm</SelectItem>
            <SelectItem value="python">python</SelectItem>
            <SelectItem value="postman">postman</SelectItem>
            <SelectItem value="axios">axios</SelectItem>
            <SelectItem value="fetch">fetch</SelectItem>
          </Select>
        </Col>
      </Grid>

      {/* Top 4 User Agents Cards */}
      {summaryLoading ? (
        <Card>
          <ChartLoader isDateChanging={isDateChanging} />
        </Card>
      ) : (
        <Grid numItems={4} className="gap-4">
          {(summaryData.results || []).slice(0, 4).map((tag, index) => {
            const userAgent = extractUserAgent(tag.tag);
            const displayName = truncateUserAgent(userAgent);
            return (
              <Card key={index}>
                <Title className="truncate" title={userAgent}>
                  {displayName}
                </Title>
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

      {/* Main TabGroup for DAU/WAU/MAU vs Per User Usage */}
      <Card>
        <TabGroup>
          <TabList className="mb-6">
            <Tab>DAU/WAU/MAU</Tab>
            <Tab>Per User Usage</Tab>
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
                        categories={uniqueTags.map(extractUserAgent).slice(0, 3)}
                        colors={["blue", "green", "orange"]}
                        valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                        yAxisWidth={60}
                        showLegend={true}
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
                        categories={uniqueTags.map(extractUserAgent).slice(0, 3)}
                        colors={["blue", "green", "orange"]}
                        valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                        yAxisWidth={60}
                        showLegend={true}
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
                        categories={uniqueTags.map(extractUserAgent).slice(0, 3)}
                        colors={["blue", "green", "orange"]}
                        valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                        yAxisWidth={60}
                        showLegend={true}
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
                dateValue={dateValue}
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