import React, { useCallback, useState, useEffect } from "react";
import { BarChart } from "@tremor/react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
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

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);

  const fetchPerUserData = useCallback(async () => {
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
  }, [accessToken, currentPage, selectedTags]);

  useEffect(() => {
    fetchPerUserData();
  }, [fetchPerUserData]);

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
      <h2 className="text-2xl font-semibold">Per User Usage</h2>
      <p className="text-muted-foreground">
        Individual developer usage metrics
      </p>

      <Tabs defaultValue="details">
        <TabsList className="mb-6">
          <TabsTrigger value="details">User Details</TabsTrigger>
          <TabsTrigger value="distribution">Usage Distribution</TabsTrigger>
        </TabsList>

        {/* Tab 1: Existing User Details Table */}
        <TabsContent value="details">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>User ID</TableHead>
                  <TableHead>User Email</TableHead>
                  <TableHead>User Agent</TableHead>
                  <TableHead className="text-right">Success Generations</TableHead>
                  <TableHead className="text-right">Total Tokens</TableHead>
                  <TableHead className="text-right">Failed Requests</TableHead>
                  <TableHead className="text-right">Total Cost</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {perUserData.results
                  .slice(0, 10)
                  .map((item: PerUserMetrics, index: number) => (
                    <TableRow key={index}>
                      <TableCell>
                        <span className="font-medium">{item.user_id}</span>
                      </TableCell>
                      <TableCell>
                        <span>{item.user_email || "N/A"}</span>
                      </TableCell>
                      <TableCell>
                        <span>{item.user_agent || "Unknown"}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span>
                          {formatAbbreviatedNumber(item.successful_requests)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span>
                          {formatAbbreviatedNumber(item.total_tokens)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span>
                          {formatAbbreviatedNumber(item.failed_requests)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span>
                          ${formatAbbreviatedNumber(item.spend, 4)}
                        </span>
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>

            {perUserData.results.length > 10 && (
              <div className="mt-4 flex justify-between items-center">
                <span className="text-sm text-muted-foreground">
                  Showing 10 of {perUserData.total_count} results
                </span>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={handlePrevPage}
                    disabled={currentPage === 1}
                  >
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
        </TabsContent>

        {/* Tab 2: Usage Distribution Histogram */}
        <TabsContent value="distribution">
            <div className="mb-4">
              <h3 className="text-lg font-semibold">
                User Usage Distribution
              </h3>
              <p className="text-muted-foreground">
                Number of users by successful request frequency
              </p>
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
                return Object.entries(categories).map(
                  ([categoryName, category]) => {
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    const dataPoint: Record<string, any> = {
                      category: categoryName,
                    };

                    // Add count for each top user agent
                    topUserAgents.forEach((agent) => {
                      dataPoint[agent] = category.agents[agent] || 0;
                    });

                    return dataPoint;
                  },
                );
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
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PerUserUsage;
