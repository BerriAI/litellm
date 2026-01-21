/**
 * User Usage Component
 *
 * Admin view for user usage analytics with filtering, sorting, and pagination.
 * Shows top users bar chart, summary cards, and paginated user table.
 */

import { DateRangePickerValue } from "@tremor/react";
import React, { useCallback, useEffect, useState } from "react";
import { adminUsersDailyActivityCall } from "../../../networking";
import { UserUsageBarChart } from "./UserUsageBarChart";
import { UserUsageFilters } from "./UserUsageFilters";
import { UserUsageSummary } from "./UserUsageSummary";
import { UserUsageTable } from "./UserUsageTable";
import {
  UserUsageFiltersState,
  UserUsageResponse,
} from "./types";

interface UserUsageProps {
  accessToken: string | null;
  dateValue: DateRangePickerValue;
}

const UserUsage: React.FC<UserUsageProps> = ({ accessToken, dateValue }) => {
  const [data, setData] = useState<UserUsageResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [topN, setTopN] = useState(10);

  const [filters, setFilters] = useState<UserUsageFiltersState>({
    tagFilters: [],
    minSpend: null,
    maxSpend: null,
    sortBy: "spend",
    sortOrder: "desc",
  });

  const fetchData = useCallback(async () => {
    if (!accessToken || !dateValue.from || !dateValue.to) return;

    setLoading(true);

    try {
      const response = await adminUsersDailyActivityCall(
        accessToken,
        dateValue.from,
        dateValue.to,
        page,
        filters.tagFilters.length > 0 ? filters.tagFilters : null,
        filters.minSpend,
        filters.maxSpend,
        filters.sortBy,
        filters.sortOrder,
        topN
      );

      setData(response);
    } catch (error) {
      console.error("Error fetching user usage data:", error);
    } finally {
      setLoading(false);
    }
  }, [
    accessToken,
    dateValue.from,
    dateValue.to,
    page,
    pageSize,
    topN,
    filters,
  ]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleFiltersChange = (newFilters: Partial<UserUsageFiltersState>) => {
    setFilters((prev) => ({ ...prev, ...newFilters }));
    setPage(1); // Reset to first page when filters change
  };

  const handlePageChange = (newPage: number) => {
    setPage(newPage);
  };

  const handlePageSizeChange = (newPageSize: number) => {
    setPageSize(newPageSize);
    setPage(1); // Reset to first page when page size changes
  };

  const handleTopNChange = (newTopN: number) => {
    setTopN(newTopN);
  };

  return (
    <div className="space-y-6">
      {/* Filters */}
      <UserUsageFilters
        filters={filters}
        onFiltersChange={handleFiltersChange}
        loading={loading}
      />

      {/* Top Users Bar Chart */}
      <UserUsageBarChart
        topUsers={data?.top_users || []}
        loading={loading}
        topN={topN}
        onTopNChange={handleTopNChange}
        sortBy={filters.sortBy}
      />

      {/* Summary Cards */}
      <UserUsageSummary summary={data?.summary || null} loading={loading} />

      {/* Paginated User Table */}
      <UserUsageTable
        users={data?.users || []}
        pagination={data?.pagination || null}
        loading={loading}
        page={page}
        pageSize={pageSize}
        onPageChange={handlePageChange}
        onPageSizeChange={handlePageSizeChange}
        sortBy={filters.sortBy}
        sortOrder={filters.sortOrder}
        onSortChange={(sortBy, sortOrder) =>
          handleFiltersChange({ sortBy, sortOrder })
        }
      />
    </div>
  );
};

export default UserUsage;
