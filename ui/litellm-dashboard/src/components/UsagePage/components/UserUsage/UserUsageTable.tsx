/**
 * User Usage Table Component
 *
 * Displays paginated user list with sorting and pagination controls
 */

import {
  Badge,
  Card,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Title,
} from "@tremor/react";
import { Select } from "antd";
import React from "react";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { UserMetrics, UserUsagePagination } from "./types";

interface UserUsageTableProps {
  users: UserMetrics[];
  pagination: UserUsagePagination | null;
  loading: boolean;
  page: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (pageSize: number) => void;
  sortBy: "spend" | "requests" | "tokens";
  sortOrder: "asc" | "desc";
  onSortChange: (
    sortBy: "spend" | "requests" | "tokens",
    sortOrder: "asc" | "desc"
  ) => void;
}

export const UserUsageTable: React.FC<UserUsageTableProps> = ({
  users,
  pagination,
  loading,
  page,
  pageSize,
  onPageChange,
  onPageSizeChange,
  sortBy,
  sortOrder,
  onSortChange,
}) => {
  const handleSortClick = (column: "spend" | "requests" | "tokens") => {
    if (sortBy === column) {
      // Toggle sort order
      onSortChange(column, sortOrder === "asc" ? "desc" : "asc");
    } else {
      // Change sort column (default to desc)
      onSortChange(column, "desc");
    }
  };

  const getSortIcon = (column: "spend" | "requests" | "tokens") => {
    if (sortBy !== column) return "⇅";
    return sortOrder === "asc" ? "↑" : "↓";
  };

  const renderPagination = () => {
    if (!pagination) return null;

    const { page: currentPage, total_pages, total_count } = pagination;
    const startItem = (currentPage - 1) * pageSize + 1;
    const endItem = Math.min(currentPage * pageSize, total_count);

    const pages = [];
    const maxPagesToShow = 7;
    const halfRange = Math.floor(maxPagesToShow / 2);

    let startPage = Math.max(1, currentPage - halfRange);
    let endPage = Math.min(total_pages, currentPage + halfRange);

    // Adjust if we're near the start or end
    if (currentPage <= halfRange) {
      endPage = Math.min(total_pages, maxPagesToShow);
    } else if (currentPage + halfRange >= total_pages) {
      startPage = Math.max(1, total_pages - maxPagesToShow + 1);
    }

    // Add first page if not visible
    if (startPage > 1) {
      pages.push(
        <button
          key={1}
          onClick={() => onPageChange(1)}
          className="px-3 py-1 border rounded hover:bg-gray-50"
        >
          1
        </button>
      );
      if (startPage > 2) {
        pages.push(
          <span key="ellipsis1" className="px-2">
            ...
          </span>
        );
      }
    }

    // Add page numbers
    for (let i = startPage; i <= endPage; i++) {
      pages.push(
        <button
          key={i}
          onClick={() => onPageChange(i)}
          className={`px-3 py-1 border rounded ${
            i === currentPage
              ? "bg-cyan-500 text-white"
              : "hover:bg-gray-50"
          }`}
        >
          {i}
        </button>
      );
    }

    // Add last page if not visible
    if (endPage < total_pages) {
      if (endPage < total_pages - 1) {
        pages.push(
          <span key="ellipsis2" className="px-2">
            ...
          </span>
        );
      }
      pages.push(
        <button
          key={total_pages}
          onClick={() => onPageChange(total_pages)}
          className="px-3 py-1 border rounded hover:bg-gray-50"
        >
          {total_pages}
        </button>
      );
    }

    return (
      <div className="flex items-center justify-between mt-4">
        <div className="text-sm text-gray-600">
          Showing {startItem}-{endItem} of {total_count} users
        </div>

        <div className="flex items-center gap-4">
          {/* Page Size Selector */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">Items per page:</span>
            <Select
              value={pageSize}
              onChange={onPageSizeChange}
              style={{ width: 80 }}
              options={[
                { label: "25", value: 25 },
                { label: "50", value: 50 },
                { label: "100", value: 100 },
              ]}
            />
          </div>

          {/* Pagination Buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => onPageChange(currentPage - 1)}
              disabled={currentPage === 1}
              className="px-3 py-1 border rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              ← Previous
            </button>

            <div className="flex gap-1">{pages}</div>

            <button
              onClick={() => onPageChange(currentPage + 1)}
              disabled={currentPage === total_pages}
              className="px-3 py-1 border rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next →
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <Card>
      <div className="flex justify-between items-center mb-4">
        <Title>👥 All Users</Title>
        {pagination && (
          <span className="text-sm text-gray-600">
            {pagination.total_count} total users
          </span>
        )}
      </div>

      {loading ? (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-500 mx-auto"></div>
          <p className="text-gray-500 mt-4">Loading users...</p>
        </div>
      ) : users.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          No users found for the selected filters.
        </div>
      ) : (
        <>
          <div className="overflow-x-auto">
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>User Email</TableHeaderCell>
                  <TableHeaderCell>
                    <button
                      onClick={() => handleSortClick("spend")}
                      className="flex items-center gap-1 hover:text-cyan-600"
                    >
                      Spend {getSortIcon("spend")}
                    </button>
                  </TableHeaderCell>
                  <TableHeaderCell>
                    <button
                      onClick={() => handleSortClick("requests")}
                      className="flex items-center gap-1 hover:text-cyan-600"
                    >
                      Requests {getSortIcon("requests")}
                    </button>
                  </TableHeaderCell>
                  <TableHeaderCell>
                    <button
                      onClick={() => handleSortClick("tokens")}
                      className="flex items-center gap-1 hover:text-cyan-600"
                    >
                      Tokens {getSortIcon("tokens")}
                    </button>
                  </TableHeaderCell>
                  <TableHeaderCell>Success Rate</TableHeaderCell>
                  <TableHeaderCell>Days Active</TableHeaderCell>
                  <TableHeaderCell>Tags</TableHeaderCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {users.map((user) => {
                  const successRate =
                    user.requests > 0
                      ? (user.successful_requests / user.requests) * 100
                      : 0;

                  return (
                    <TableRow key={user.user_id}>
                      <TableCell>
                        <div>
                          <div className="font-medium">
                            {user.user_email || user.user_id}
                          </div>
                          {user.user_email && (
                            <div className="text-xs text-gray-500">
                              {user.user_id}
                            </div>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        <span className="font-medium text-cyan-600">
                          ${formatNumberWithCommas(user.spend, 2)}
                        </span>
                      </TableCell>
                      <TableCell>
                        {formatNumberWithCommas(user.requests, 0)}
                        <div className="text-xs text-gray-500">
                          <span className="text-green-600">
                            {formatNumberWithCommas(user.successful_requests, 0)}
                          </span>
                          {" / "}
                          <span className="text-red-600">
                            {formatNumberWithCommas(user.failed_requests, 0)}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        {formatNumberWithCommas(user.tokens, 0)}
                      </TableCell>
                      <TableCell>
                        <Badge
                          color={
                            successRate >= 99
                              ? "green"
                              : successRate >= 95
                              ? "yellow"
                              : "red"
                          }
                        >
                          {successRate.toFixed(1)}%
                        </Badge>
                      </TableCell>
                      <TableCell>{user.days_active}</TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {user.tags && user.tags.length > 0 ? (
                            user.tags.slice(0, 2).map((tag, idx) => (
                              <Badge key={idx} size="xs">
                                {tag.replace("User-Agent:", "")}
                              </Badge>
                            ))
                          ) : (
                            <span className="text-gray-400 text-sm">-</span>
                          )}
                          {user.tags && user.tags.length > 2 && (
                            <Badge size="xs" color="gray">
                              +{user.tags.length - 2}
                            </Badge>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>

          {renderPagination()}
        </>
      )}
    </Card>
  );
};
