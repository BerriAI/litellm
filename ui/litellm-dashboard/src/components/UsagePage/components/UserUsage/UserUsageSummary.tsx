/**
 * User Usage Summary Component
 *
 * Displays summary cards with total users, spend, requests, etc.
 */

import { Card, Col, Grid, Text, Title } from "@tremor/react";
import React from "react";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { UserUsageSummary as UserUsageSummaryType } from "./types";

interface UserUsageSummaryProps {
  summary: UserUsageSummaryType | null;
  loading: boolean;
}

export const UserUsageSummary: React.FC<UserUsageSummaryProps> = ({
  summary,
  loading,
}) => {
  if (loading || !summary) {
    return (
      <Grid numItems={5} className="gap-4">
        {[...Array(5)].map((_, i) => (
          <Card key={i}>
            <div className="animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-24 mb-2"></div>
              <div className="h-8 bg-gray-200 rounded w-16"></div>
            </div>
          </Card>
        ))}
      </Grid>
    );
  }

  return (
    <Grid numItems={5} className="gap-4">
      {/* Total Users */}
      <Card>
        <Title>Total Users</Title>
        <Text className="text-2xl font-bold mt-2">
          {formatNumberWithCommas(summary.total_users, 0)}
        </Text>
      </Card>

      {/* Total Spend */}
      <Card>
        <Title>Total Spend</Title>
        <Text className="text-2xl font-bold mt-2 text-cyan-600">
          ${formatNumberWithCommas(summary.total_spend, 2)}
        </Text>
      </Card>

      {/* Average per User */}
      <Card>
        <Title>Avg per User</Title>
        <Text className="text-2xl font-bold mt-2">
          ${formatNumberWithCommas(summary.avg_spend_per_user, 2)}
        </Text>
      </Card>

      {/* Power Users */}
      <Card>
        <Title>Power Users (&gt;$200)</Title>
        <Text className="text-2xl font-bold mt-2 text-green-600">
          {formatNumberWithCommas(summary.power_users_count, 0)}
        </Text>
        <Text className="text-sm text-gray-500 mt-1">
          {summary.total_users > 0
            ? `${((summary.power_users_count / summary.total_users) * 100).toFixed(1)}% of total`
            : "0%"}
        </Text>
      </Card>

      {/* Low Users */}
      <Card>
        <Title>Low Users (&lt;$10)</Title>
        <Text className="text-2xl font-bold mt-2 text-orange-600">
          {formatNumberWithCommas(summary.low_users_count, 0)}
        </Text>
        <Text className="text-sm text-gray-500 mt-1">
          {summary.total_users > 0
            ? `${((summary.low_users_count / summary.total_users) * 100).toFixed(1)}% of total`
            : "0%"}
        </Text>
      </Card>
    </Grid>
  );
};
