/**
 * User Usage Bar Chart Component
 *
 * Displays top N users in a horizontal bar chart
 */

import { BarChart, Card, Title } from "@tremor/react";
import { Segmented } from "antd";
import React from "react";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { ChartLoader } from "../../../shared/chart_loader";
import { UserMetrics } from "./types";

interface UserUsageBarChartProps {
  topUsers: UserMetrics[];
  loading: boolean;
  topN: number;
  onTopNChange: (topN: number) => void;
  sortBy: "spend" | "requests" | "tokens";
}

const valueFormatters = {
  spend: (value: number) => `$${formatNumberWithCommas(value, 2)}`,
  requests: (value: number) => formatNumberWithCommas(value, 0),
  tokens: (value: number) => formatNumberWithCommas(value, 0),
};

export const UserUsageBarChart: React.FC<UserUsageBarChartProps> = ({
  topUsers,
  loading,
  topN,
  onTopNChange,
  sortBy,
}) => {
  const chartData = topUsers.map((user) => ({
    name: user.user_email || user.user_id,
    value: user[sortBy],
  }));

  const valueFormatter = valueFormatters[sortBy];

  const getTitle = () => {
    const sortLabel = sortBy === "spend" ? "Spend" : sortBy === "requests" ? "Requests" : "Tokens";
    return `Top ${topN} Users by ${sortLabel}`;
  };

  return (
    <Card>
      <div className="flex justify-between items-center mb-4">
        <Title>{getTitle()}</Title>
        <Segmented
          options={[
            { label: "10", value: 10 },
            { label: "25", value: 25 },
            { label: "50", value: 50 },
          ]}
          value={topN}
          onChange={(value) => onTopNChange(value as number)}
        />
      </div>

      {loading ? (
        <ChartLoader isDateChanging={loading} />
      ) : chartData.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          No user data available for the selected date range and filters.
        </div>
      ) : (
        <BarChart
          data={chartData}
          index="name"
          categories={["value"]}
          colors={["cyan"]}
          valueFormatter={valueFormatter}
          layout="vertical"
          yAxisWidth={250}
          showLegend={false}
          style={{ height: Math.min(chartData.length * 52, 600) }}
          customTooltip={({ payload, active }) => {
            if (!active || !payload?.[0]) return null;
            const data = payload[0].payload;
            const user = topUsers.find(
              (u) => (u.user_email || u.user_id) === data.name
            );
            if (!user) return null;

            return (
              <div className="bg-white p-4 shadow-lg rounded-lg border">
                <p className="font-bold">{user.user_email || user.user_id}</p>
                <p className="text-cyan-500">
                  Spend: ${formatNumberWithCommas(user.spend, 2)}
                </p>
                <p className="text-gray-600">
                  Requests: {formatNumberWithCommas(user.requests, 0)}
                </p>
                <p className="text-green-600">
                  Successful: {formatNumberWithCommas(user.successful_requests, 0)}
                </p>
                <p className="text-red-600">
                  Failed: {formatNumberWithCommas(user.failed_requests, 0)}
                </p>
                <p className="text-gray-600">
                  Tokens: {formatNumberWithCommas(user.tokens, 0)}
                </p>
                <p className="text-gray-600">Days Active: {user.days_active}</p>
                {user.tags && user.tags.length > 0 && (
                  <p className="text-gray-600 text-sm mt-2">
                    Tags: {user.tags.join(", ")}
                  </p>
                )}
              </div>
            );
          }}
        />
      )}
    </Card>
  );
};
