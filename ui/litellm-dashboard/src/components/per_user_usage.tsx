import React, { useState, useEffect, useCallback } from "react";
import type { ColumnDef, OnChangeFn, PaginationState } from "@tanstack/react-table";
import { BarChart } from "@/components/shared/charts";
import { DataTable } from "@/components/shared/DataTable";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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

  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 50 });
  const [pagedTags, setPagedTags] = useState(selectedTags);

  if (pagedTags !== selectedTags) {
    setPagedTags(selectedTags);
    setPagination((prev) => (prev.pageIndex === 0 ? prev : { ...prev, pageIndex: 0 }));
  }

  useEffect(() => {
    if (!accessToken) return;

    let stale = false;
    perUserAnalyticsCall(
      accessToken,
      pagination.pageIndex + 1,
      pagination.pageSize,
      pagedTags.length > 0 ? pagedTags : undefined,
    )
      .then((response) => {
        if (stale) return;
        setPerUserData(response);
      })
      .catch((error) => console.error("Failed to fetch per-user data:", error));

    return () => {
      stale = true;
    };
  }, [accessToken, pagedTags, pagination]);

  const handlePaginationChange = useCallback<OnChangeFn<PaginationState>>((updaterOrValue) => {
    setPagination((prev) => {
      const next = typeof updaterOrValue === "function" ? updaterOrValue(prev) : updaterOrValue;
      return next.pageSize === prev.pageSize ? next : { pageIndex: 0, pageSize: next.pageSize };
    });
  }, []);

  const columns: ColumnDef<PerUserMetrics>[] = [
    {
      header: "User ID",
      accessorKey: "user_id",
      cell: ({ row }) => <span className="font-medium">{row.original.user_id}</span>,
    },
    {
      header: "User Email",
      accessorKey: "user_email",
      cell: ({ row }) => row.original.user_email || "N/A",
    },
    {
      header: "User Agent",
      accessorKey: "user_agent",
      cell: ({ row }) => row.original.user_agent || "Unknown",
    },
    {
      header: "Success Generations",
      accessorKey: "successful_requests",
      meta: { numeric: true },
      cell: ({ row }) => formatAbbreviatedNumber(row.original.successful_requests),
    },
    {
      header: "Total Tokens",
      accessorKey: "total_tokens",
      meta: { numeric: true },
      cell: ({ row }) => formatAbbreviatedNumber(row.original.total_tokens),
    },
    {
      header: "Failed Requests",
      accessorKey: "failed_requests",
      meta: { numeric: true },
      cell: ({ row }) => formatAbbreviatedNumber(row.original.failed_requests),
    },
    {
      header: "Total Cost",
      accessorKey: "spend",
      meta: { numeric: true },
      cell: ({ row }) => `$${formatAbbreviatedNumber(row.original.spend, 4)}`,
    },
  ];

  return (
    <div className="mb-6">
      <h3 className="text-lg font-medium text-foreground">Per User Usage</h3>
      <p className="text-sm text-muted-foreground">Individual developer usage metrics</p>

      <Tabs defaultValue="details">
        <TabsList variant="line" className="mb-6 h-auto w-full justify-start rounded-none border-b p-0">
          <TabsTrigger value="details" className="flex-none rounded-none px-4 py-2">
            User Details
          </TabsTrigger>
          <TabsTrigger value="distribution" className="flex-none rounded-none px-4 py-2">
            Usage Distribution
          </TabsTrigger>
        </TabsList>

        {/* Tab 1: Existing User Details Table */}
        <TabsContent value="details" keepMounted>
          <DataTable
            columns={columns}
            data={perUserData.results}
            getRowId={(row) => row.user_id}
            paginationMode="server"
            pagination={pagination}
            onPaginationChange={handlePaginationChange}
            rowCount={perUserData.total_count}
            noDataMessage="No per-user usage data"
            size="compact"
          />
        </TabsContent>

        {/* Tab 2: Usage Distribution Histogram */}
        <TabsContent value="distribution" keepMounted>
          <div className="mb-4">
            <h4 className="text-lg font-medium text-foreground">User Usage Distribution</h4>
            <p className="text-sm text-muted-foreground">Number of users by successful request frequency</p>
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
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PerUserUsage;
