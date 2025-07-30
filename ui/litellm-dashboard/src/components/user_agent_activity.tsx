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
} from "@tremor/react";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { userAgentAnalyticsCall, userAgentSummaryCall } from "./networking";
import UsageDatePicker from "./shared/usage_date_picker";
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

      {/* Charts */}
      <Grid numItems={1} className="gap-4">
        <Card>
          <Title>Success Generations by User Agent</Title>
          <BarChart
            data={chartData.slice(0, 10)} // Top 10
            index="user_agent"
            categories={["successful_requests", "total_tokens"]}
            colors={["green", "blue"]}
            valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
            yAxisWidth={60}
            showLegend={true}
          />
        </Card>
      </Grid>

      {/* Top User Agents Table */}
      <Card>
        <Title>Top User Agents</Title>
        <Table className="mt-4">
          <TableHead>
            <TableRow>
              <TableHeaderCell>User Agent</TableHeaderCell>
              <TableHeaderCell className="text-right">Requests</TableHeaderCell>
              <TableHeaderCell className="text-right">Success Rate</TableHeaderCell>
              <TableHeaderCell className="text-right">Tokens</TableHeaderCell>
              <TableHeaderCell className="text-right">Spend</TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {summaryData.top_user_agents.map((ua, index) => (
              <TableRow key={index}>
                <TableCell className="font-medium">{ua.user_agent}</TableCell>
                <TableCell className="text-right">
                  {formatNumberWithCommas(ua.requests)}
                </TableCell>
                <TableCell className="text-right">
                  <span
                    className={
                      (ua.successful_requests / ua.requests) * 100 > 95
                        ? "text-green-600"
                        : (ua.successful_requests / ua.requests) * 100 > 90
                        ? "text-yellow-600"
                        : "text-red-600"
                    }
                  >
                    {((ua.successful_requests / ua.requests) * 100).toFixed(1)}%
                  </span>
                </TableCell>
                <TableCell className="text-right">
                  {formatNumberWithCommas(ua.tokens)}
                </TableCell>
                <TableCell className="text-right">
                  ${formatNumberWithCommas(ua.spend, 2)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>

      {/* Daily Activity Table */}
      <Card>
        <div className="flex justify-between items-center mb-4">
          <Title>Daily User Agent Activity</Title>
          <div className="flex items-center space-x-2">
            <Button
              variant="secondary"
              size="xs"
              onClick={handlePrevPage}
              disabled={currentPage <= 1 || loading}
            >
              Previous
            </Button>
            <Text className="text-sm">
              Page {currentPage} of {analyticsData.total_pages}
            </Text>
            <Button
              variant="secondary"
              size="xs"
              onClick={handleNextPage}
              disabled={currentPage >= analyticsData.total_pages || loading}
            >
              Next
            </Button>
          </div>
        </div>
        
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>Date</TableHeaderCell>
              <TableHeaderCell>User Agent</TableHeaderCell>
              <TableHeaderCell className="text-right">DAU</TableHeaderCell>
              <TableHeaderCell className="text-right">WAU</TableHeaderCell>
              <TableHeaderCell className="text-right">MAU</TableHeaderCell>
              <TableHeaderCell className="text-right">Requests</TableHeaderCell>
              <TableHeaderCell className="text-right">Success Rate</TableHeaderCell>
              <TableHeaderCell className="text-right">Tokens</TableHeaderCell>
              <TableHeaderCell className="text-right">Spend</TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {analyticsData.results.map((item, index) => (
              <TableRow key={index}>
                <TableCell>{item.date}</TableCell>
                <TableCell className="font-medium">
                  {item.user_agent || "Unknown"}
                </TableCell>
                <TableCell className="text-right">
                  {formatNumberWithCommas(item.metrics.dau)}
                </TableCell>
                <TableCell className="text-right">
                  {formatNumberWithCommas(item.metrics.wau)}
                </TableCell>
                <TableCell className="text-right">
                  {formatNumberWithCommas(item.metrics.mau)}
                </TableCell>
                <TableCell className="text-right">
                  <div>
                    <div>{formatNumberWithCommas(item.metrics.total_requests)}</div>
                    <div className="text-xs text-gray-500">
                      {formatNumberWithCommas(item.metrics.successful_requests)} success
                    </div>
                  </div>
                </TableCell>
                <TableCell className="text-right">
                  <span
                    className={
                      (item.metrics.successful_requests / item.metrics.total_requests) * 100 > 95
                        ? "text-green-600"
                        : (item.metrics.successful_requests / item.metrics.total_requests) * 100 > 90
                        ? "text-yellow-600"
                        : "text-red-600"
                    }
                  >
                    {item.metrics.total_requests > 0
                      ? ((item.metrics.successful_requests / item.metrics.total_requests) * 100).toFixed(1)
                      : "0"}%
                  </span>
                </TableCell>
                <TableCell className="text-right">
                  <div>
                    <div>{formatNumberWithCommas(item.metrics.total_tokens)}</div>
                    <div className="text-xs text-gray-500">
                      {formatNumberWithCommas(item.metrics.completed_tokens)} completed
                    </div>
                  </div>
                </TableCell>
                <TableCell className="text-right">
                  ${formatNumberWithCommas(item.metrics.spend, 4)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        
        {analyticsData.results.length === 0 && !loading && (
          <div className="text-center py-8">
            <Text>No user agent activity data found for the selected period.</Text>
          </div>
        )}
        
        {loading && (
          <div className="text-center py-8">
            <Text>Loading...</Text>
          </div>
        )}
      </Card>
    </div>
  );
};

export default UserAgentActivity;