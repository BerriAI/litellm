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
import { userAgentAnalyticsCall, userAgentSummaryCall } from "./networking";
import UsageDatePicker from "./shared/usage_date_picker";
import PerUserUsage from "./per_user_usage";
import { DateRangePickerValue } from "@tremor/react";

interface UserAgentMetrics {
  dau: number;
  wau: number;
  mau: number;
  successful_requests: number;
  failed_requests: number;
  total_requests: number;
  completed_tokens: number;
  total_tokens: number;
  spend: number;
}

interface UserAgentActivityData {
  date: string;
  tag: string;
  user_agent: string;
  metrics: UserAgentMetrics;
}

interface UserAgentAnalyticsResponse {
  results: UserAgentActivityData[];
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
}

interface UserAgentSummaryData {
  total_user_agents: number;
  total_requests: number;
  total_successful_requests: number;
  total_failed_requests: number;
  total_tokens: number;
  total_spend: number;
  top_user_agents: Array<{
    user_agent: string;
    requests: number;
    successful_requests: number;
    failed_requests: number;
    tokens: number;
    spend: number;
  }>;
}

interface UserAgentActivityProps {
  accessToken: string | null;
  userRole: string | null;
}

const UserAgentActivity: React.FC<UserAgentActivityProps> = ({
  accessToken,
  userRole,
}) => {
  const [analyticsData, setAnalyticsData] = useState<UserAgentAnalyticsResponse>({
    results: [],
    total_count: 0,
    page: 1,
    page_size: 50,
    total_pages: 0,
  });
  
  const [summaryData, setSummaryData] = useState<UserAgentSummaryData>({
    total_user_agents: 0,
    total_requests: 0,
    total_successful_requests: 0,
    total_failed_requests: 0,
    total_tokens: 0,
    total_spend: 0,
    top_user_agents: [],
  });

  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    to: new Date(),
  });

  const [userAgentFilter, setUserAgentFilter] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);

  const fetchData = async () => {
    if (!accessToken || !dateValue.from || !dateValue.to) return;

    setLoading(true);
    try {
      const [analytics, summary] = await Promise.all([
        userAgentAnalyticsCall(
          accessToken,
          dateValue.from,
          dateValue.to,
          currentPage,
          50,
          userAgentFilter || undefined
        ),
        userAgentSummaryCall(accessToken, dateValue.from, dateValue.to),
      ]);

      setAnalyticsData(analytics);
      setSummaryData(summary);
    } catch (error) {
      console.error("Failed to fetch user agent data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [accessToken, dateValue, userAgentFilter, currentPage]);

  const handleNextPage = () => {
    if (currentPage < analyticsData.total_pages) {
      setCurrentPage(currentPage + 1);
    }
  };

  const handlePrevPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  // Aggregate data by user agent for charts
  const aggregatedByUserAgent = analyticsData.results.reduce((acc, item) => {
    const ua = item.user_agent || "Unknown";
    if (!acc[ua]) {
      acc[ua] = {
        user_agent: ua,
        total_requests: 0,
        successful_requests: 0,
        failed_requests: 0,
        total_tokens: 0,
        spend: 0,
        dau: 0,
        wau: 0,
        mau: 0,
      };
    }
    acc[ua].total_requests += item.metrics.total_requests;
    acc[ua].successful_requests += item.metrics.successful_requests;
    acc[ua].failed_requests += item.metrics.failed_requests;
    acc[ua].total_tokens += item.metrics.total_tokens;
    acc[ua].spend += item.metrics.spend;
    // For user counts, take the maximum to avoid double counting
    acc[ua].dau = Math.max(acc[ua].dau, item.metrics.dau);
    acc[ua].wau = Math.max(acc[ua].wau, item.metrics.wau);
    acc[ua].mau = Math.max(acc[ua].mau, item.metrics.mau);
    return acc;
  }, {} as Record<string, any>);

  const chartData = Object.values(aggregatedByUserAgent).sort(
    (a: any, b: any) => b.total_requests - a.total_requests
  );

  const successRateData = summaryData.top_user_agents.map((ua) => ({
    user_agent: ua.user_agent,
    success_rate: ua.successful_requests / (ua.requests || 1) * 100,
    total_requests: ua.requests,
  }));

  // Get unique user agents for chart
  const uniqueUserAgents = Array.from(
    new Set(analyticsData.results.map(item => item.user_agent || "Unknown"))
  ).slice(0, 3); // Top 3 user agents

  // Prepare daily chart data (DAU)
  const dailyChartData = analyticsData.results.reduce((acc, item) => {
    const existingDate = acc.find(d => d.date === item.date);
    if (existingDate) {
      existingDate[item.user_agent || "Unknown"] = item.metrics.dau;
    } else {
      const newDateEntry: any = { 
        date: item.date,
        [item.user_agent || "Unknown"]: item.metrics.dau
      };
      acc.push(newDateEntry);
    }
    return acc;
  }, [] as any[]);

  // Prepare weekly chart data (WAU)
  const weeklyChartData = analyticsData.results.reduce((acc, item) => {
    const existingDate = acc.find(d => d.week === item.date);
    if (existingDate) {
      existingDate[item.user_agent || "Unknown"] = item.metrics.wau;
    } else {
      const newDateEntry: any = { 
        week: `Week ${acc.length + 1}`,
        [item.user_agent || "Unknown"]: item.metrics.wau
      };
      acc.push(newDateEntry);
    }
    return acc;
  }, [] as any[]);

  // Prepare monthly chart data (MAU)
  const monthlyChartData = analyticsData.results.reduce((acc, item) => {
    const existingDate = acc.find(d => d.month === item.date);
    if (existingDate) {
      existingDate[item.user_agent || "Unknown"] = item.metrics.mau;
    } else {
      const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"];
      const newDateEntry: any = { 
        month: monthNames[acc.length % 7] || `Month ${acc.length + 1}`,
        [item.user_agent || "Unknown"]: item.metrics.mau
      };
      acc.push(newDateEntry);
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
          <UsageDatePicker
            value={dateValue}
            onValueChange={(value) => {
              setDateValue(value);
              setCurrentPage(1); // Reset to first page when date changes
            }}
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
      <Grid numItems={4} className="gap-4">
        {summaryData.top_user_agents.slice(0, 4).map((ua, index) => (
          <Card key={index}>
            <Title className="truncate" title={ua.user_agent}>
              {ua.user_agent}
            </Title>
            <div className="mt-4 space-y-3">
              <div>
                <Text className="text-sm text-gray-600">Success Requests</Text>
                <Metric className="text-lg">{formatAbbreviatedNumber(ua.successful_requests)}</Metric>
              </div>
              <div>
                <Text className="text-sm text-gray-600">Total Tokens</Text>
                <Metric className="text-lg">{formatAbbreviatedNumber(ua.tokens)}</Metric>
              </div>
              <div>
                <Text className="text-sm text-gray-600">Total Cost</Text>
                <Metric className="text-lg">${formatAbbreviatedNumber(ua.spend, 4)}</Metric>
              </div>
            </div>
        </Card>
        ))}
        {/* Fill remaining slots if less than 4 agents */}
        {Array.from({ length: Math.max(0, 4 - summaryData.top_user_agents.length) }).map((_, index) => (
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
                    <BarChart
                      data={dailyChartData}
                      index="date"
                      categories={uniqueUserAgents.slice(0, 3)}
                      colors={["blue", "green", "orange"]}
                      valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                      yAxisWidth={60}
                      showLegend={true}
                    />
                  </TabPanel>
                  
                  <TabPanel>
                    <div className="mb-4">
                      <Title className="text-lg">Weekly Active Users - Last 4 Weeks</Title>
                    </div>
                    <BarChart
                      data={weeklyChartData}
                      index="week"
                      categories={uniqueUserAgents.slice(0, 3)}
                      colors={["blue", "green", "orange"]}
                      valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                      yAxisWidth={60}
                      showLegend={true}
                    />
                  </TabPanel>
                  
                  <TabPanel>
                    <div className="mb-4">
                      <Title className="text-lg">Monthly Active Users - Last 7 Months</Title>
                    </div>
                    <BarChart
                      data={monthlyChartData}
                      index="month"
                      categories={uniqueUserAgents.slice(0, 3)}
                      colors={["blue", "green", "orange"]}
                      valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                      yAxisWidth={60}
                      showLegend={true}
                    />
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