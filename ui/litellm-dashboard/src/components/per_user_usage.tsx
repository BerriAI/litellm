import React, { useState, useEffect } from "react";
import {
  Title,
  Subtitle,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  BarChart,
  Text,
  Button,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
} from "@tremor/react";
import { perUserAnalyticsCall } from "./networking";

interface PerUserMetrics {
  user_id: string;
  user_email: string | null;
  user_agent: string | null;
  successful_requests: number;
  failed_requests: number;
  total_requests: number;
  total_tokens: number;
  spend: number;
}

interface PerUserAnalyticsResponse {
  results: PerUserMetrics[];
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
}

interface PerUserUsageProps {
  accessToken: string | null;
  selectedTags: string[];
  formatAbbreviatedNumber: (value: number, decimalPlaces?: number) => string;
}

const PerUserUsage: React.FC<PerUserUsageProps> = ({ accessToken, selectedTags, formatAbbreviatedNumber }) => {
  // Maximum number of user agent categories to show in charts to prevent color palette overflow
  const MAX_USER_AGENTS = 8;
  const [perUserData, setPerUserData] = useState<PerUserAnalyticsResponse>({
    results: [],
    total_count: 0,
    page: 1,
    page_size: 50,
    total_pages: 0,
  });

  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);

  const fetchPerUserData = async () => {
    if (!accessToken) return;

    setLoading(true);
    try {
      const response = await perUserAnalyticsCall(
        accessToken,
        currentPage,
        50,
        selectedTags.length > 0 ? selectedTags : undefined,
      );
      setPerUserData(response);
    } catch (error) {
      console.error("Failed to fetch per-user data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPerUserData();
  }, [accessToken, selectedTags, currentPage]);

  const handleNextPage = () => {
    if (currentPage < perUserData.total_pages) {
      setCurrentPage(currentPage + 1);
    }
  };

  const handlePrevPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  return (
    <div className="mb-6">
      <Title>Per User Usage</Title>
      <Subtitle>Individual developer usage metrics</Subtitle>

      <TabGroup>
        <TabList className="mb-6">
          <Tab>User Details</Tab>
          <Tab>Usage Distribution</Tab>
        </TabList>

        <TabPanels>
          {/* Tab 1: Existing User Details Table */}
          <TabPanel>
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>User ID</TableHeaderCell>
                  <TableHeaderCell>User Email</TableHeaderCell>
                  <TableHeaderCell>User Agent</TableHeaderCell>
                  <TableHeaderCell className="text-right">Success Generations</TableHeaderCell>
                  <TableHeaderCell className="text-right">Total Tokens</TableHeaderCell>
                  <TableHeaderCell className="text-right">Failed Requests</TableHeaderCell>
                  <TableHeaderCell className="text-right">Total Cost</TableHeaderCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {perUserData.results.slice(0, 10).map((item: PerUserMetrics, index: number) => (
                  <TableRow key={index}>
                    <TableCell>
                      <Text className="font-medium">{item.user_id}</Text>
                    </TableCell>
                    <TableCell>
                      <Text>{item.user_email || "N/A"}</Text>
                    </TableCell>
                    <TableCell>
                      <Text>{item.user_agent || "Unknown"}</Text>
                    </TableCell>
                    <TableCell className="text-right">
                      <Text>{formatAbbreviatedNumber(item.successful_requests)}</Text>
                    </TableCell>
                    <TableCell className="text-right">
                      <Text>{formatAbbreviatedNumber(item.total_tokens)}</Text>
                    </TableCell>
                    <TableCell className="text-right">
                      <Text>{formatAbbreviatedNumber(item.failed_requests)}</Text>
                    </TableCell>
                    <TableCell className="text-right">
                      <Text>${formatAbbreviatedNumber(item.spend, 4)}</Text>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>

            {perUserData.results.length > 10 && (
              <div className="mt-4 flex justify-between items-center">
                <Text className="text-sm text-gray-500">Showing 10 of {perUserData.total_count} results</Text>
                <div className="flex gap-2">
                  <Button size="sm" variant="secondary" onClick={handlePrevPage} disabled={currentPage === 1}>
                    Previous
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={handleNextPage}
                    disabled={currentPage >= perUserData.total_pages}
                  >
                    Next
                  </Button>
                </div>
              </div>
            )}
          </TabPanel>

          {/* Tab 2: Usage Distribution Histogram */}
          <TabPanel>
            <div className="mb-4">
              <Title className="text-lg">User Usage Distribution</Title>
              <Subtitle>Number of users by successful request frequency</Subtitle>
            </div>

            <BarChart
              data={(() => {
                // Get top user agents by frequency first
                const userAgentCounts = new Map<string, number>();
                perUserData.results.forEach((item: PerUserMetrics) => {
                  const agent = item.user_agent || "Unknown";
                  userAgentCounts.set(agent, (userAgentCounts.get(agent) || 0) + 1);
                });

                const topUserAgents = Array.from(userAgentCounts.entries())
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, MAX_USER_AGENTS)
                  .map(([agent]) => agent);

                // Categorize users by successful request count and user agent
                const categories = {
                  "1-9 requests": { range: [1, 9], agents: {} as Record<string, number> },
                  "10-99 requests": { range: [10, 99], agents: {} as Record<string, number> },
                  "100-999 requests": { range: [100, 999], agents: {} as Record<string, number> },
                  "1K-9.9K requests": { range: [1000, 9999], agents: {} as Record<string, number> },
                  "10K-99.9K requests": { range: [10000, 99999], agents: {} as Record<string, number> },
                  "100K+ requests": { range: [100000, Infinity], agents: {} as Record<string, number> },
                };

                // Count users in each category by user agent (only for top user agents)
                perUserData.results.forEach((item: PerUserMetrics) => {
                  const successCount = item.successful_requests;
                  const userAgent = item.user_agent || "Unknown";

                  // Only process if this is one of the top user agents
                  if (topUserAgents.includes(userAgent)) {
                    Object.entries(categories).forEach(([categoryName, category]) => {
                      if (successCount >= category.range[0] && successCount <= category.range[1]) {
                        if (!category.agents[userAgent]) {
                          category.agents[userAgent] = 0;
                        }
                        category.agents[userAgent]++;
                      }
                    });
                  }
                });

                // Convert to chart data format for stacked bar chart
                return Object.entries(categories).map(([categoryName, category]) => {
                  const dataPoint: Record<string, any> = { category: categoryName };

                  // Add count for each top user agent
                  topUserAgents.forEach((agent) => {
                    dataPoint[agent] = category.agents[agent] || 0;
                  });

                  return dataPoint;
                });
              })()}
              index="category"
              categories={(() => {
                // Count user agents by frequency and get top ones
                const userAgentCounts = new Map<string, number>();
                perUserData.results.forEach((item: PerUserMetrics) => {
                  const agent = item.user_agent || "Unknown";
                  userAgentCounts.set(agent, (userAgentCounts.get(agent) || 0) + 1);
                });

                // Sort by frequency (most common first) and limit to top MAX_USER_AGENTS
                return Array.from(userAgentCounts.entries())
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, MAX_USER_AGENTS)
                  .map(([agent]) => agent);
              })()}
              colors={["blue", "green", "orange", "red", "purple", "yellow", "pink", "indigo"]}
              valueFormatter={(value: number) => `${value} users`}
              yAxisWidth={80}
              showLegend={true}
              stack={true}
            />
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default PerUserUsage;
